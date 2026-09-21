import asyncio
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("agentic_rag.main")

from estimate_tokens import (
    DEFAULT_CONTRACTS_DIR,
    DEFAULT_CSV_PATH,
    estimate_contracts,
    estimate_products,
    format_estimate_summary,
    get_encoder,
)
from src.agent import run_agent
from src.ingestion import build_contracts_db, build_search_db
from src.tracing import configure_langfuse, flush_langfuse

# Eagerly configure the Langfuse client so it is ready before any agent runs
configure_langfuse()


def _log_token_estimate(results: list) -> None:
    """Log the token estimation summary and print to stdout."""
    summary = format_estimate_summary(results)
    logger.info("Pre-ingestion token estimate:\n%s", summary)
    print(f"\n{summary}\n")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        if action == "build":
            enc = get_encoder()
            _log_token_estimate([estimate_products(DEFAULT_CSV_PATH, enc)])
            asyncio.run(build_search_db())
        elif action == "build-contracts":
            enc = get_encoder()
            _log_token_estimate([estimate_contracts(DEFAULT_CONTRACTS_DIR, enc)])
            asyncio.run(build_contracts_db())
        elif action == "search":
            if len(sys.argv) == 3:
                q = sys.argv[2]
            else:
                q = "Show me casual cotton shirts under ₹1000"
            logger.info("Question processing... %s", q)
            answer = asyncio.run(run_agent(q))
            print(f"\nAnswer:\n{answer}")
        else:
            print(
                "Usage: uv run main.py build|build-contracts|search [query]",
                file=sys.stderr,
            )
            sys.exit(1)
    finally:
        flush_langfuse()
