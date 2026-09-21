"""CLI runner for the CUAD RAG Evaluation Suite.

Modes
─────
  retrieval   Vector retrieval metrics only (Recall@K, Precision@K, MRR, nDCG).
              Uses a shared DB pool + OpenAI embedding client.
              Embedding API calls are made; no LLM generation → cheap.

  answer      Full agent pipeline + structured LLM judge (correctness + faithfulness).
              Each case = 1 agent run + 1 gpt-4o-mini judge call → moderate cost.
              Logfire is ON by default for answer runs.

  all         Runs retrieval then answer in sequence.

Quick start
───────────
  # Smoke test — 5 cases, retrieval only, no Logfire
  uv run python -m evals.run_eval --mode retrieval --smoke --no-logfire

  # Answer eval, Governing Law only, 10 cases/category
  uv run python -m evals.run_eval --mode answer --categories "Governing Law"

  # Full retrieval benchmark, all categories, K=5
  uv run python -m evals.run_eval --mode retrieval --k 5 --limit 50

  # Full benchmark, all metrics, Logfire on
  uv run python -m evals.run_eval --mode all --limit 50 --answer-limit 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Auto-load .env from the project root
# Uses os.environ.setdefault — already-exported shell vars always win.
# ---------------------------------------------------------------------------
_ENV_FILE = Path(__file__).parent.parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

import logfire
import pydantic_core
from openai import AsyncOpenAI

from evals.answer_cases import AnswerInput, build_answer_dataset
from evals.answer_evaluators import AnswerOutput
from evals.retrieval_cases import RetrievalInput, build_retrieval_dataset
from src.agent import run_agent_with_details
from src.database import database_connect

logger = logging.getLogger("evals.run_eval")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Retrieval SQL (shared pool variant — avoids creating a new pool per case)
# ---------------------------------------------------------------------------

_SQL_WITH_CAT = """
    SELECT contract_id, document_name, chunk_id, page, text, category,
           (embedding <-> $1) AS distance
    FROM contracts
    WHERE $3 = ANY(category)
    ORDER BY embedding <-> $1
    LIMIT $2
"""

_SQL_NO_CAT = """
    SELECT contract_id, document_name, chunk_id, page, text, category,
           (embedding <-> $1) AS distance
    FROM contracts
    ORDER BY embedding <-> $1
    LIMIT $2
"""


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------


def make_retrieval_task(pool, openai_client: AsyncOpenAI):
    """Factory returning an async retrieval task bound to shared pool + client.

    Using a factory avoids creating a new DB pool and OpenAI client per case,
    which would be prohibitively expensive for large eval runs.
    """

    async def run_retrieval(inp: RetrievalInput) -> List[Dict[str, Any]]:
        """Embed query and run pgvector similarity search."""
        embedding = await openai_client.embeddings.create(
            input=inp.query,
            model="text-embedding-3-small",
        )
        emb_json = pydantic_core.to_json(embedding.data[0].embedding).decode()

        if inp.category:
            rows = await pool.fetch(_SQL_WITH_CAT, emb_json, inp.k, inp.category)
        else:
            rows = await pool.fetch(_SQL_NO_CAT, emb_json, inp.k)

        return [
            {
                "chunk_id": row["chunk_id"],
                "document_name": row["document_name"],
                "text": row["text"],
                "category": list(row["category"]) if row["category"] else [],
                "distance": float(row["distance"]),
            }
            for row in rows
        ]

    return run_retrieval


async def run_answer(inp: AnswerInput) -> AnswerOutput:
    """Task function: run the full agent pipeline and return answer + context.

    Extracts the retrieved context from retrieve_contracts tool call results
    only (not retrieve_products), since we're evaluating contract clause QA.
    """
    result = await run_agent_with_details(inp.question)

    retrieved_context = "\n\n---\n\n".join(
        tc["result"]
        for tc in result.get("tool_calls", [])
        if tc.get("result") and tc.get("tool_name") == "retrieve_contracts"
    )

    return AnswerOutput(
        answer_text=result["output"],
        retrieved_context=retrieved_context,
    )


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------


async def run_retrieval_eval(
    categories: Optional[List[str]],
    limit: int,
    k: int,
    smoke: bool,
    save_results: bool,
) -> Dict[str, Any]:
    """Build and run the retrieval evaluation dataset."""
    effective_limit = 5 if smoke else limit
    logger.info(
        "Building retrieval dataset — categories=%s  limit=%d/cat  k=%d",
        categories or "all",
        effective_limit,
        k,
    )

    dataset = await build_retrieval_dataset(
        categories=categories,
        limit_per_category=effective_limit,
        k=k,
    )
    n_cases = len(dataset.cases)
    logger.info("Retrieval dataset ready: %d cases", n_cases)

    if n_cases == 0:
        logger.warning(
            "No retrieval cases built. Is the contracts table populated? "
            "Run `make backend-contracts` first."
        )
        return {"mode": "retrieval", "case_count": 0}

    async with database_connect(False) as pool:
        openai_client = AsyncOpenAI()
        task_fn = make_retrieval_task(pool, openai_client)
        report = await dataset.evaluate(task_fn)

    report.print(include_input=False, include_output=False)

    if save_results:
        _save_report(report, mode="retrieval")

    return {"mode": "retrieval", "case_count": n_cases}


async def run_answer_eval(
    categories: Optional[List[str]],
    limit: int,
    smoke: bool,
    save_results: bool,
) -> Dict[str, Any]:
    """Build and run the answer quality evaluation dataset."""
    effective_limit = 5 if smoke else limit
    logger.info(
        "Building answer dataset — categories=%s  limit=%d/cat",
        categories or "all",
        effective_limit,
    )

    dataset = build_answer_dataset(
        categories=categories,
        limit_per_category=effective_limit,
    )
    n_cases = len(dataset.cases)
    logger.info("Answer dataset ready: %d cases", n_cases)

    if n_cases == 0:
        logger.warning("No answer cases built — check your CUAD annotations file.")
        return {"mode": "answer", "case_count": 0}

    # Warn before full (non-smoke) runs about LLM cost
    if not smoke and n_cases > 20:
        print(
            f"\n⚠️  About to run {n_cases} agent calls + {n_cases} judge calls "
            f"(gpt-4o-mini). Press Ctrl-C within 5 s to abort...\n"
        )
        await asyncio.sleep(5)

    report = await dataset.evaluate(run_answer)
    report.print(include_input=True, include_output=True)

    if save_results:
        _save_report(report, mode="answer")

    return {"mode": "answer", "case_count": n_cases}


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------


def _save_report(report: Any, mode: str) -> None:
    """Persist a pydantic_evals report to evals/results/ as timestamped JSON."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = RESULTS_DIR / f"{timestamp}_{mode}.json"
    try:
        if hasattr(report, "model_dump"):
            data = report.model_dump()
        else:
            data = {"repr": str(report)}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\n💾 Results saved → {path}")
    except Exception as exc:
        logger.warning("Could not save results snapshot: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CUAD RAG Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--mode",
        choices=["retrieval", "answer", "all"],
        default="retrieval",
        help="Evaluation mode (default: retrieval)",
    )
    p.add_argument(
        "--categories",
        default=None,
        metavar="CAT1,CAT2",
        help='Comma-separated category filter, e.g. "Governing Law,Audit Rights"',
    )
    p.add_argument(
        "--limit",
        type=int,
        default=50,
        metavar="N",
        help="Max cases per category for retrieval eval (default: 50)",
    )
    p.add_argument(
        "--answer-limit",
        type=int,
        default=10,
        metavar="N",
        dest="answer_limit",
        help="Max cases per category for answer eval (default: 10)",
    )
    p.add_argument(
        "--k",
        type=int,
        default=8,
        metavar="K",
        help="Retrieval depth K (default: 8, matches production LIMIT 8)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: run only 5 cases per mode to validate setup cheaply",
    )
    p.add_argument(
        "--logfire",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Upload results to Logfire Datasets & Experiments. "
            "Default: ON for answer mode, OFF for retrieval-only."
        ),
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving JSON snapshot to evals/results/",
    )
    return p.parse_args()


async def _main(args: argparse.Namespace) -> None:
    categories = (
        [c.strip() for c in args.categories.split(",")]
        if args.categories
        else None
    )

    # Logfire: on by default for answer eval, off by default for retrieval-only
    use_logfire = args.logfire
    if use_logfire is None:
        use_logfire = args.mode in ("answer", "all")

    if use_logfire:
        logfire.configure()
        logger.info("Logfire configured — results will upload to Datasets & Experiments")

    save = not args.no_save
    summaries: List[Dict[str, Any]] = []

    if args.mode in ("retrieval", "all"):
        s = await run_retrieval_eval(
            categories=categories,
            limit=args.limit,
            k=args.k,
            smoke=args.smoke,
            save_results=save,
        )
        summaries.append(s)

    if args.mode in ("answer", "all"):
        s = await run_answer_eval(
            categories=categories,
            limit=args.answer_limit,
            smoke=args.smoke,
            save_results=save,
        )
        summaries.append(s)

    print("\n" + "═" * 50)
    print("  Evaluation Complete")
    print("═" * 50)
    for s in summaries:
        print(f"  {s['mode']:12s}  {s['case_count']} cases")
    print()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)s  %(message)s",
    )
    args = _parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
