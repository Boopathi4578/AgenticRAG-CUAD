"""Central Langfuse tracing configuration for the Agentic RAG Catalogue Tool.

Langfuse v4 uses `@observe()` as the primary tracing decorator and
`get_client()` for the singleton client.  This module re-exports both so
the rest of the codebase only needs to import from one place.

"""

import logging
import os

from langfuse import Langfuse, get_client, observe 

logger = logging.getLogger("agentic_rag.tracing")

_configured: bool = False


def configure_langfuse() -> None:
    """Configure the global Langfuse client from environment variables.

    Call once at application startup (before any @observe-decorated function
    runs).  Subsequent calls are no-ops.

    Environment variables:
        LANGFUSE_PUBLIC_KEY  – project public key (from cloud.langfuse.com)
        LANGFUSE_SECRET_KEY  – project secret key
        LANGFUSE_HOST        – server URL (defaults to Langfuse Cloud)
    """
    global _configured
    if _configured:
        return

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not (public_key and secret_key):
        logger.warning(
            "LANGFUSE_PUBLIC_KEY and/or LANGFUSE_SECRET_KEY are not set. "
            "Traces will NOT be sent to Langfuse. "
            "Add them to your .env file to enable cloud observability."
        )

    # Langfuse v4 reads LANGFUSE_* env vars automatically when calling
    # get_client(); we call Langfuse(...) explicitly so we can pass host.
    Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    logger.info("Langfuse client configured (host=%s)", host)
    _configured = True


def get_langfuse() -> Langfuse:
    """Return the global Langfuse client singleton (initialising if needed)."""
    configure_langfuse()
    return get_client()


def flush_langfuse() -> None:
    """Flush queued Langfuse events.  Call on application shutdown."""
    try:
        client = get_client()
        client.flush()
        logger.debug("Langfuse events flushed.")
    except Exception:
        pass
