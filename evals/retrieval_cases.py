"""Build retrieval evaluation cases from CUAD annotations + live DB chunks.

Evaluation design — Oracle / Evidence Retrieval Benchmark
──────────────────────────────────────────────────────────
Query     = annotation text (the gold clause span verbatim)
Category  = annotation category (passed as metadata filter to the retrieval tool)
Ground truth = chunk_ids from the DB whose text overlaps the gold span

Why gold text as the query?
  This is the hardest meaningful test: if the vector store cannot surface the
  chunk containing the very clause text that was ingested, something is wrong
  with chunking, embedding, or distance thresholds.  It gives an upper-bound
  ("oracle") estimate of achievable recall — any realistic NL query will score
  equal to or lower than this.

Relevance labelling — three signals (descending priority)
────────────────────────────────────────────────────────
  1. Character-offset overlap  (DB chunks store ingestion offsets via
     chunk_start / chunk_end columns; exact same logic as ingestion.py)
  2. Normalised substring containment  (fallback when offset columns absent)
  3. Token overlap ratio ≥ 0.50        (soft fallback)

All three signals are checked per chunk; a chunk is "relevant" if ANY signal fires.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional

import asyncpg
from pydantic_evals import Case, Dataset

from evals.cuad_loader import (
    Annotation,
    is_relevant_chunk,
    load_annotations,
    sample_annotations,
)
from src.database import database_connect

logger = logging.getLogger("evals.retrieval_cases")


# ---------------------------------------------------------------------------
# Input / output types for the retrieval task
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalInput:
    """Input to the retrieval task function passed by pydantic_evals."""

    query: str          # gold annotation text used as the query
    category: str       # CUAD clause category (used as metadata filter)
    document_name: str  # source document (carried for analysis; not sent to retrieval)
    gold_text: str      # alias of query; available to evaluators via ctx.inputs
    k: int = 8          # number of results to retrieve


@dataclass(frozen=True)
class RelevanceLabels:
    """Pre-computed ground-truth for a single retrieval case."""

    chunk_ids: FrozenSet[int]  # DB chunk_ids that are relevant to the annotation
    document_name: str         # carried for debugging / per-document analysis


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

# Query includes chunk_start / chunk_end when available (Stage 2+ schema).
# COALESCE guards against older schemas where those columns don't exist yet.
_FETCH_CHUNKS_SQL = """
    SELECT
        chunk_id,
        document_name,
        text,
        category,
        COALESCE(chunk_start, NULL) AS chunk_start,
        COALESCE(chunk_end,   NULL) AS chunk_end
    FROM contracts
    ORDER BY document_name, chunk_id
"""

# Fallback for schemas without chunk_start/chunk_end columns
_FETCH_CHUNKS_FALLBACK_SQL = """
    SELECT chunk_id, document_name, text, category
    FROM contracts
    ORDER BY document_name, chunk_id
"""


async def _fetch_all_chunks(pool: asyncpg.Pool) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch all contract chunks from DB, grouped by document_name.

    Tries to retrieve chunk_start/chunk_end (offset-based relevance signal).
    Falls back gracefully if those columns are absent.

    Returns:
        Dict[document_name → List[chunk_dict]]
    """
    try:
        rows = await pool.fetch(_FETCH_CHUNKS_SQL)
        has_offsets = True
    except Exception:
        logger.info(
            "chunk_start/chunk_end columns not found; using text-only relevance signals"
        )
        rows = await pool.fetch(_FETCH_CHUNKS_FALLBACK_SQL)
        has_offsets = False

    chunks_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        doc = row["document_name"]
        entry: Dict[str, Any] = {
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "category": list(row["category"]) if row["category"] else [],
        }
        if has_offsets:
            entry["chunk_start"] = row["chunk_start"]
            entry["chunk_end"] = row["chunk_end"]
        chunks_by_doc.setdefault(doc, []).append(entry)

    logger.info(
        "Fetched chunks for %d documents (offsets available: %s)",
        len(chunks_by_doc),
        has_offsets,
    )
    return chunks_by_doc


# ---------------------------------------------------------------------------
# Relevance labelling
# ---------------------------------------------------------------------------


def _label_relevant_chunks(
    annotation: Annotation,
    doc_chunks: List[Dict[str, Any]],
    token_threshold: float = 0.5,
) -> FrozenSet[int]:
    """Return chunk_ids that are relevant to the given annotation.

    Passes chunk_start/chunk_end to is_relevant_chunk when available so the
    offset-overlap signal (strongest) fires first.
    """
    relevant: set = set()
    for chunk in doc_chunks:
        relevant_flag, _ = is_relevant_chunk(
            annotation=annotation,
            chunk_text=chunk["text"],
            chunk_start=chunk.get("chunk_start"),
            chunk_end=chunk.get("chunk_end"),
            token_threshold=token_threshold,
        )
        if relevant_flag:
            relevant.add(chunk["chunk_id"])
    return frozenset(relevant)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


async def build_retrieval_dataset(
    categories: Optional[List[str]] = None,
    limit_per_category: int = 50,
    k: int = 8,
    token_threshold: float = 0.5,
) -> Dataset:
    """Build a pydantic_evals Dataset for retrieval evaluation.

    Steps:
      1. Load CUAD annotations and sample up to limit_per_category per category.
      2. Query DB once for ALL chunks, grouped by document name.
      3. For each annotation, label relevant chunk_ids via multi-signal matching.
      4. Build Case objects and return a ready-to-run Dataset.

    Annotations are skipped (with a warning) when:
      - The document has no chunks in the DB (not yet ingested)
      - No chunk in the document passes the relevance threshold (chunking gap)

    Args:
        categories:          Optional category filter.
        limit_per_category:  Max annotations per category (default 50).
        k:                   Retrieval depth K stored on RetrievalInput.
        token_threshold:     Min token overlap to mark a chunk relevant (default 0.5).

    Returns:
        pydantic_evals Dataset[RetrievalInput, List[Dict], RelevanceLabels]
    """
    # Import here to avoid circular imports at module level
    from evals.metrics import MRR, NDCGAtK, PrecisionAtK, RecallAtK  # noqa: PLC0415

    # 1. Sample annotations
    data = load_annotations()
    sampled = sample_annotations(
        data, categories=categories, limit_per_category=limit_per_category
    )
    logger.info("Sampled %d annotations across categories", len(sampled))

    # 2. Fetch all chunks once
    async with database_connect(False) as pool:
        chunks_by_doc = await _fetch_all_chunks(pool)

    # 3. Build cases
    cases: List[Case] = []
    skipped_no_doc = 0
    skipped_no_match = 0

    for doc_name, ann in sampled:
        doc_chunks = chunks_by_doc.get(doc_name, [])
        if not doc_chunks:
            skipped_no_doc += 1
            continue

        relevant_ids = _label_relevant_chunks(ann, doc_chunks, token_threshold)
        if not relevant_ids:
            skipped_no_match += 1
            continue

        # Sanitise case name (pydantic_evals requires unique names)
        safe_doc = doc_name[:28].replace(" ", "_")
        safe_cat = ann.category.replace(" ", "_")
        case_name = f"{safe_doc}__{safe_cat}__{ann.start_char}"

        cases.append(
            Case(
                name=case_name,
                inputs=RetrievalInput(
                    query=ann.text,
                    category=ann.category,
                    document_name=doc_name,
                    gold_text=ann.text,
                    k=k,
                ),
                expected_output=RelevanceLabels(
                    chunk_ids=relevant_ids,
                    document_name=doc_name,
                ),
                metadata={"category": ann.category, "doc_name": doc_name},
            )
        )

    logger.info(
        "Retrieval dataset: %d cases built, %d skipped (no doc in DB), "
        "%d skipped (no chunk match)",
        len(cases),
        skipped_no_doc,
        skipped_no_match,
    )
    if skipped_no_doc:
        logger.warning(
            "%d annotations skipped — their documents are not ingested yet. "
            "Run `make backend-contracts` first.",
            skipped_no_doc,
        )

    return Dataset(
        name="cuad-retrieval-eval",
        cases=cases,
        evaluators=[
            RecallAtK(k=k),
            PrecisionAtK(k=k),
            MRR(k=k),
            NDCGAtK(k=k),
        ],
    )
