"""
Token & Cost Estimator for Agentic RAG Data Ingestion
======================================================

Run this script BEFORE ingesting data to understand how many OpenAI
embedding tokens will be consumed and the approximate cost.

Covers both data sources:
  • Product Catalog  (sample_products.csv)
  • Legal Contracts  (data/contracts/*.txt)

Usage:
    uv run estimate_tokens.py                        # estimate both sources
    uv run estimate_tokens.py --source products      # products only
    uv run estimate_tokens.py --source contracts      # contracts only
    uv run estimate_tokens.py --contracts-dir /path   # custom contracts path
    uv run estimate_tokens.py --csv /path/to/file.csv # custom CSV path
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path

try:
    import tiktoken
except ImportError:
    print(
        "ERROR: 'tiktoken' is required to run this script.\n"
        "       Install it with:  uv pip install tiktoken\n"
        "       Or:               pip install tiktoken",
        file=sys.stderr,
    )
    sys.exit(1)


logger = logging.getLogger("agentic_rag.token_estimator")

# ---------------------------------------------------------------------------
# Constants — keep in sync with src/ingestion.py
# ---------------------------------------------------------------------------

DEFAULT_CSV_PATH = Path(__file__).parent / "sample_products.csv"
DEFAULT_CONTRACTS_DIR = Path(__file__).parent / "data" / "contracts"

EMBEDDING_MODEL = "text-embedding-3-small"

# Chunking parameters (mirrors src/ingestion.py)
CHUNK_SIZE = 1000  # approximate tokens (chars / 4)
CHUNK_OVERLAP = 200  # overlap in tokens

# OpenAI pricing (USD per 1M tokens) — update if pricing changes
# Source: https://openai.com/api/pricing/
EMBEDDING_COST_PER_1M_TOKENS = 0.02  # text-embedding-3-small


# ---------------------------------------------------------------------------
# Public helpers — used by main.py / app.py imports
# ---------------------------------------------------------------------------


def get_encoder() -> tiktoken.Encoding:
    """Returns the tiktoken encoder for the embedding model (cl100k_base)."""
    return tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _product_embedding_content(row: dict) -> str:
    """Replicates ProductData.embedding_content() from src/models.py."""
    return "\n\n".join(
        (
            f"ProductID: {row['ProductID']}",
            f"ProductName: {row['ProductName']}",
            f"ProductBrand: {row['ProductBrand']}",
            f"Gender: {row['Gender']}",
            f"PrimaryColor: {row['PrimaryColor']}",
            f"Description: {row['Description']}",
        )
    )


def _estimate_chunk_count(file_chars: int) -> int:
    """Mathematically estimates the number of chunks for a contract file.

    Uses the same CHUNK_SIZE / CHUNK_OVERLAP parameters as
    src/ingestion.chunk_text() without reimplementing the paragraph-aware
    splitting logic.  Good enough for cost estimation.
    """
    chars_per_chunk = CHUNK_SIZE * 4  # ≈ 4000 chars
    if file_chars <= chars_per_chunk:
        return 1
    # Each subsequent chunk advances by (chunk_size - overlap) * 4 chars
    stride = (CHUNK_SIZE - CHUNK_OVERLAP) * 4  # ≈ 3200 chars
    return math.ceil((file_chars - chars_per_chunk) / stride) + 1


# ---------------------------------------------------------------------------
# Estimation routines
# ---------------------------------------------------------------------------


def estimate_products(csv_path: Path, enc: tiktoken.Encoding) -> dict:
    """Estimate tokens for product catalog embeddings."""
    if not csv_path.exists():
        return {
            "source": "Products",
            "file": str(csv_path),
            "error": f"CSV file not found: {csv_path}",
        }

    rows = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total_tokens = 0
    per_product: list[dict] = []

    for row in rows:
        content = _product_embedding_content(row)
        tokens = len(enc.encode(content))
        total_tokens += tokens
        per_product.append(
            {
                "product_id": row["ProductID"],
                "product_name": row["ProductName"],
                "text_chars": len(content),
                "tokens": tokens,
            }
        )

    cost_usd = (total_tokens / 1_000_000) * EMBEDDING_COST_PER_1M_TOKENS

    return {
        "source": "Products",
        "file": str(csv_path),
        "product_count": len(rows),
        "embedding_calls": len(rows),
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost_usd,
        "details": per_product,
    }


def estimate_contracts(contracts_dir: Path, enc: tiktoken.Encoding) -> dict:
    """Estimate tokens for contract embeddings.

    Reads raw text files and uses mathematical estimation for chunk counts
    and overlap-adjusted token totals — no chunking logic is reimplemented here.
    """
    if not contracts_dir.exists():
        return {
            "source": "Contracts",
            "directory": str(contracts_dir),
            "error": f"Contracts directory not found: {contracts_dir}",
        }

    txt_files = sorted(contracts_dir.glob("*.txt"))
    if not txt_files:
        return {
            "source": "Contracts",
            "directory": str(contracts_dir),
            "error": "No .txt files found in contracts directory",
        }

    # Overlap factor: each chunk re-embeds CHUNK_OVERLAP tokens of the previous chunk
    overlap_factor = CHUNK_SIZE / (CHUNK_SIZE - CHUNK_OVERLAP)  # 1.25

    total_tokens = 0
    total_chunks = 0
    per_file: list[dict] = []

    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8")
        raw_tokens = len(enc.encode(text))
        num_chunks = _estimate_chunk_count(len(text))

        # Single-chunk files have no overlap; multi-chunk files inflate by overlap_factor
        if num_chunks == 1:
            embedding_tokens = raw_tokens
        else:
            embedding_tokens = int(raw_tokens * overlap_factor)

        total_tokens += embedding_tokens
        total_chunks += num_chunks
        per_file.append(
            {
                "file_name": txt_file.name,
                "document_name": txt_file.name,
                "raw_chars": len(text),
                "raw_tokens": raw_tokens,
                "estimated_chunks": num_chunks,
                "embedding_tokens": embedding_tokens,
            }
        )

    cost_usd = (total_tokens / 1_000_000) * EMBEDDING_COST_PER_1M_TOKENS

    return {
        "source": "Contracts",
        "directory": str(contracts_dir),
        "file_count": len(txt_files),
        "total_chunks": total_chunks,
        "embedding_calls": total_chunks,
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost_usd,
        "details": per_file,
    }


# ---------------------------------------------------------------------------
# Summary formatter — importable by main.py / app.py for logging
# ---------------------------------------------------------------------------


def format_estimate_summary(results: list[dict]) -> str:
    """Returns a compact human-readable summary string for logging/UI display."""
    lines = [
        f"📊 Token Estimate  (model: {EMBEDDING_MODEL}, "
        f"${EMBEDDING_COST_PER_1M_TOKENS}/1M tokens)"
    ]
    grand_tokens = 0
    grand_cost = 0.0
    grand_calls = 0

    for r in results:
        if "error" in r:
            lines.append(f"  ⚠️  {r['source']}: {r['error']}")
            continue

        calls = r.get("embedding_calls", 0)
        tokens = r.get("total_tokens", 0)
        cost = r.get("estimated_cost_usd", 0.0)

        if r["source"] == "Products":
            lines.append(
                f"  🛍️  Products: {r['product_count']} items → "
                f"{calls:,} API calls, {tokens:,} tokens, ${cost:.6f}"
            )
        elif r["source"] == "Contracts":
            lines.append(
                f"  📄  Contracts: {r['file_count']} files → ~{r['total_chunks']:,} chunks, "
                f"{calls:,} API calls, {tokens:,} tokens, ${cost:.6f}"
            )

        grand_tokens += tokens
        grand_cost += cost
        grand_calls += calls

    lines.append(
        f"  ── TOTAL: {grand_calls:,} API calls, "
        f"{grand_tokens:,} tokens, ${grand_cost:.6f}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pretty-print report (standalone CLI output)
# ---------------------------------------------------------------------------


def _hr(char: str = "─", width: int = 70) -> str:
    return char * width


def print_report(results: list[dict]) -> None:
    """Render a human-readable estimation report to stdout."""

    print()
    print(_hr("═"))
    print("  🔢  TOKEN & COST ESTIMATION REPORT")
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  Price: ${EMBEDDING_COST_PER_1M_TOKENS:.4f} / 1M tokens")
    print(_hr("═"))

    grand_total_tokens = 0
    grand_total_cost = 0.0
    grand_total_api_calls = 0

    for result in results:
        print()
        print(f"  📦 Source: {result['source']}")
        print(_hr())

        if "error" in result:
            print(f"  ⚠️  {result['error']}")
            continue

        path_key = "file" if "file" in result else "directory"
        print(f"  Path:            {result[path_key]}")

        if result["source"] == "Products":
            print(f"  Products:        {result['product_count']}")
            print(f"  Embedding Calls: {result['embedding_calls']}")
            print(f"  Total Tokens:    {result['total_tokens']:,}")
            print(f"  Est. Cost:       ${result['estimated_cost_usd']:.6f}")
            print()

            # Per-product detail table
            print(f"  {'ID':<8} {'Name':<50} {'Chars':>7} {'Tokens':>8}")
            print(f"  {'─' * 8} {'─' * 50} {'─' * 7} {'─' * 8}")
            for p in result["details"]:
                name = p["product_name"][:48]
                print(
                    f"  {p['product_id']:<8} {name:<50} {p['text_chars']:>7,} {p['tokens']:>8,}"
                )

        elif result["source"] == "Contracts":
            print(f"  Files:           {result['file_count']}")
            print(f"  Total Chunks:    ~{result['total_chunks']:,}")
            print(f"  Embedding Calls: ~{result['embedding_calls']:,}")
            print(f"  Total Tokens:    ~{result['total_tokens']:,}")
            print(f"  Est. Cost:       ~${result['estimated_cost_usd']:.6f}")
            print()

            # Per-file summary (top 15 + truncation notice)
            details = result["details"]
            show_count = min(len(details), 15)
            print(f"  {'File':<60} {'Chars':>9} {'~Chunks':>8} {'~Tokens':>9}")
            print(f"  {'─' * 60} {'─' * 9} {'─' * 8} {'─' * 9}")
            for f in details[:show_count]:
                fname = f["file_name"][:58]
                print(
                    f"  {fname:<60} {f['raw_chars']:>9,} {f['estimated_chunks']:>8,} {f['embedding_tokens']:>9,}"
                )
            if len(details) > show_count:
                remaining = len(details) - show_count
                remaining_tokens = sum(
                    d["embedding_tokens"] for d in details[show_count:]
                )
                print(
                    f"  ... and {remaining} more files (~{remaining_tokens:,} tokens)"
                )

        grand_total_tokens += result.get("total_tokens", 0)
        grand_total_cost += result.get("estimated_cost_usd", 0.0)
        grand_total_api_calls += result.get("embedding_calls", 0)

    # Grand totals
    print()
    print(_hr("═"))
    print("  📊 GRAND TOTAL")
    print(_hr())
    print(f"  Total API Calls:    ~{grand_total_api_calls:,}")
    print(f"  Total Tokens:       ~{grand_total_tokens:,}")
    print(f"  Estimated Cost:     ~${grand_total_cost:.6f}")
    print(_hr("═"))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate OpenAI embedding tokens & cost before data ingestion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        choices=["products", "contracts", "both"],
        default="both",
        help="Which data source to estimate (default: both)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Path to the product CSV file (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--contracts-dir",
        type=Path,
        default=DEFAULT_CONTRACTS_DIR,
        help=f"Path to the contracts directory (default: {DEFAULT_CONTRACTS_DIR})",
    )
    args = parser.parse_args()

    # Load the tokenizer for the embedding model
    print("Loading tokenizer (cl100k_base for text-embedding-3-small)...")
    enc = get_encoder()

    results: list[dict] = []

    if args.source in ("products", "both"):
        print(f"Scanning products CSV: {args.csv}")
        results.append(estimate_products(args.csv, enc))

    if args.source in ("contracts", "both"):
        print(f"Scanning contracts directory: {args.contracts_dir}")
        results.append(estimate_contracts(args.contracts_dir, enc))

    print_report(results)


if __name__ == "__main__":
    main()
