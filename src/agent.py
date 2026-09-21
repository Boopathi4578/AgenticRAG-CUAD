import difflib
import json
import logging
from typing import Any

import pydantic_core
from langfuse.openai import AsyncOpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from src.database import database_connect
from src.models import Deps
from src.tracing import get_langfuse, observe

logger = logging.getLogger("agentic_rag.agent")


async def _resolve_category(pool, requested: str) -> tuple[str, bool]:
    """Fuzzy-match a requested category string against all valid categories in the DB.

    Returns:
        (resolved_category, was_corrected) where `was_corrected` is True if the
        original string was different from the best match found.
    """
    rows = await pool.fetch(
        "SELECT DISTINCT cat FROM contracts, unnest(category) AS cat ORDER BY cat"
    )
    valid_categories = [r["cat"] for r in rows if r["cat"]]

    if not valid_categories:
        logger.warning(
            '_resolve_category: no categories found in DB, using requested value as-is: "%s"',
            requested,
        )
        return requested, False

    # Exact match (case-sensitive) — fast path
    if requested in valid_categories:
        return requested, False

    # Case-insensitive exact match
    lower_map = {cat.lower(): cat for cat in valid_categories}
    if requested.lower() in lower_map:
        resolved = lower_map[requested.lower()]
        logger.info(
            '🔧 [Category Resolved] Case-insensitive match: "%s" → "%s"',
            requested,
            resolved,
        )
        return resolved, True

    # Fuzzy match — finds the closest valid category by character similarity
    matches = difflib.get_close_matches(requested, valid_categories, n=1, cutoff=0.6)
    if matches:
        resolved = matches[0]
        logger.info(
            '🔧 [Category Resolved] Fuzzy match: "%s" → "%s" (similarity cutoff=0.6)',
            requested,
            resolved,
        )
        return resolved, True

    # No match found — log a warning; the query will return 0 results rather than wrong results
    logger.warning(
        '⚠️ [Category Unresolved] No close match found for "%s" among %d valid categories. '
        "Will query as-is (expect 0 results). Valid categories: %s",
        requested,
        len(valid_categories),
        valid_categories,
    )
    return requested, False


SYSTEM_PROMPT = """\
You are a helpful assistant with access to two data sources:

1. **Product Catalog** — an e-commerce catalog of clothing, shoes, accessories, etc.
   Use the `retrieve_products` tool to search for products by description, color, brand, etc.

2. **Legal Contracts** — a collection of business contracts (agency, franchise, licensing,
   collaboration agreements, etc.).
   Use the `retrieve_contracts` tool to search for contract clauses, terms, or specific agreements.
   You can optionally filter by clause category (e.g., "Anti-Assignment", "Change Of Control",
   "Governing Law", "Termination For Convenience", "Non-Compete", "Exclusivity", "Audit Rights",
   "Ip Ownership Assignment", etc.) when the user asks for clauses of a specific type.

Always use the appropriate tool based on the user's query. You may call both tools if the query
could span both data sources. Provide clear, well-structured answers based on the retrieved data.
"""

_agent: Agent | None = None


def get_agent() -> Agent:
    """Lazily initializes and returns the Pydantic AI Agent with retrieval tool attached."""
    global _agent
    if _agent is not None:
        return _agent

    agent = Agent("openai:gpt-4o-mini", deps_type=Deps, system_prompt=SYSTEM_PROMPT)

    @agent.tool
    async def retrieve_products(context: RunContext[Deps], search_query: str) -> str:
        """Retrieve relevant products from the e-commerce catalog based on similarity search.

        Use this tool when the user asks about products, clothing, shoes, accessories,
        brands, prices, or anything related to shopping.
        """
        logger.info(
            'Tool [retrieve_products] invoked with search_query="%s"', search_query
        )
        lf = get_langfuse()
        with lf.start_as_current_observation(
            name="tool_retrieve_products",
            as_type="tool",
            input={"search_query": search_query},
        ) as span:
            try:
                embedding = await context.deps.openai.embeddings.create(
                    input=search_query,
                    model="text-embedding-3-small",
                )
                embedding_json = pydantic_core.to_json(
                    embedding.data[0].embedding
                ).decode()

                # Fetching top 8 products (closest vectors) based on Euclidean distance
                rows = await context.deps.pool.fetch(
                    "SELECT productid, productname, description FROM products "
                    "ORDER BY embedding <-> $1 LIMIT 8",
                    embedding_json,
                )

                logger.info(
                    "Tool [retrieve_products] retrieved %d product rows", len(rows)
                )
                span.update(output={"retrieved_count": len(rows)})

                return "\n\n".join(
                    f"# {row['productid']}\nProduct Details:{row['productname']}\n\n{row['description']}\n"
                    for row in rows
                )
            except Exception as exc:
                span.update(
                    output={"error": str(exc)},
                    level="ERROR",
                    status_message=str(exc),
                )
                raise

    @agent.tool
    async def retrieve_contracts(
        context: RunContext[Deps],
        search_query: str,
        category: str | None = None,
    ) -> str:
        """Retrieve relevant legal contract excerpts based on similarity search with optional category filtering.

        Args:
            search_query: The search query string describing clauses, terms, or topics to find.
            category: Optional clause category to filter by (e.g. 'Anti-Assignment',
                'Change Of Control', 'Governing Law', 'Termination For Convenience',
                'Non-Compete', 'Exclusivity', 'Ip Ownership Assignment', 'Audit Rights',
                'Cap On Liability', 'Revenue/Profit Sharing', etc.).

        Use this tool when the user asks about contracts, agreements, legal terms,
        clauses, licensing, franchising, collaboration, or any business/legal topic.
        """
        filter_active = bool(category)
        if filter_active:
            logger.info(
                '🔍 [Metadata Filtering ACTIVE] Tool [retrieve_contracts] filtering by category="%s" (search_query="%s")',
                category,
                search_query,
            )
        else:
            logger.info(
                '🔍 [Metadata Filtering INACTIVE] Tool [retrieve_contracts] unconstrained similarity search (search_query="%s")',
                search_query,
            )

        lf = get_langfuse()
        with lf.start_as_current_observation(
            name="tool_retrieve_contracts",
            as_type="tool",
            input={
                "search_query": search_query,
                "category": category,
                "metadata_filter_active": filter_active,
            },
        ) as span:
            try:
                embedding = await context.deps.openai.embeddings.create(
                    input=search_query,
                    model="text-embedding-3-small",
                )
                embedding_json = pydantic_core.to_json(
                    embedding.data[0].embedding
                ).decode()

                if category:
                    # Validate & fuzzy-correct the LLM-provided category before querying
                    category, was_corrected = await _resolve_category(
                        context.deps.pool, category
                    )
                    if was_corrected:
                        logger.info(
                            'Category auto-corrected to "%s" via fuzzy matching',
                            category,
                        )

                    logger.info(
                        '⚡ Executing GIN-indexed array containment query: WHERE $2 = ANY(category) [category="%s"]',
                        category,
                    )
                    rows = await context.deps.pool.fetch(
                        """
                        SELECT contract_id, document_name, chunk_id, page, text, category,
                               (embedding <-> $1) AS distance
                        FROM contracts
                        WHERE $2 = ANY(category)
                        ORDER BY embedding <-> $1
                        LIMIT 8
                        """,
                        embedding_json,
                        category,
                    )
                else:
                    logger.info(
                        "⚡ Executing standard vector distance query without category filter"
                    )
                    rows = await context.deps.pool.fetch(
                        """
                        SELECT contract_id, document_name, chunk_id, page, text, category,
                               (embedding <-> $1) AS distance
                        FROM contracts
                        ORDER BY embedding <-> $1
                        LIMIT 8
                        """,
                        embedding_json,
                    )

                logger.info(
                    '📊 [Metadata Filtering Results] Retrieved %d contract chunks (filter_category="%s")',
                    len(rows),
                    category or "None",
                )
                for idx, row in enumerate(rows, 1):
                    has_target = category in row["category"] if category else True
                    logger.info(
                        '  ├─ [%d/%d] doc="%s" chunk=%d page=%s dist=%.4f cats=%s %s',
                        idx,
                        len(rows),
                        row["document_name"],
                        row["chunk_id"],
                        row["page"] or "N/A",
                        float(row["distance"]),
                        row["category"],
                        "✅ MATCH" if has_target else "❌ MISMATCH",
                    )

                span.update(
                    output={
                        "retrieved_count": len(rows),
                        "category": category,
                        "filter_active": filter_active,
                    }
                )

                return "\n\n".join(
                    f"# {row['document_name']} (chunk {row['chunk_id']}, page {row['page'] or 'N/A'}) "
                    f"[Categories: {', '.join(row['category'])}]\n{row['text']}\n"
                    for row in rows
                )
            except Exception as exc:
                span.update(
                    output={"error": str(exc)},
                    level="ERROR",
                    status_message=str(exc),
                )
                raise

    _agent = agent
    return _agent


async def run_agent(question: str) -> str:
    """Executes the Pydantic AI agent against the catalog database."""
    details = await run_agent_with_details(question)
    return details["output"]


@observe(name="run_agent_with_details", as_type="agent")
async def run_agent_with_details(question: str) -> dict[str, Any]:
    """Executes the Pydantic AI agent and returns answer, tool execution trace, and usage."""
    logger.info('Starting agent execution for question="%s"', question)

    lf = get_langfuse()
    lf.set_current_trace_io(input={"question": question})

    openai = AsyncOpenAI()

    async with database_connect(False) as pool:
        deps = Deps(openai=openai, pool=pool)
        agent_instance = get_agent()
        run_result = await agent_instance.run(question, deps=deps)

    output_text = run_result.output
    usage = run_result.usage

    # Extract tool calls and return data.   
    tool_calls: list[dict[str, Any]] = []
    call_map: dict[str, dict[str, Any]] = {}  

    for msg in run_result.all_messages():   
        if hasattr(msg, "parts"):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    tool_data = {
                        "tool_name": part.tool_name,
                        "args": part.args if isinstance(part.args, dict) else {},
                        "tool_call_id": getattr(part, "tool_call_id", None),
                        "result": None,
                    }
                    if isinstance(part.args, str):
                        try:
                            tool_data["args"] = json.loads(part.args)
                        except Exception:
                            tool_data["args"] = {"raw": part.args}

                    tool_calls.append(tool_data)
                    if tool_data["tool_call_id"]:
                        call_map[tool_data["tool_call_id"]] = tool_data
                elif isinstance(part, ToolReturnPart):
                    call_id = getattr(part, "tool_call_id", None)
                    if call_id and call_id in call_map:
                        call_map[call_id]["result"] = part.content
                    elif tool_calls:
                        for tc in reversed(tool_calls):
                            if (
                                tc["tool_name"] == part.tool_name
                                and tc["result"] is None
                            ):
                                tc["result"] = part.content
                                break

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    usage_dict = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "requests": getattr(usage, "requests", 0) or 1,
        "tool_calls_count": len(tool_calls),
    }

    logger.info(
        'Agent completed question="%s" with %d tool calls and %d total tokens',
        question,
        len(tool_calls),
        usage_dict["total_tokens"],
    )

    # Update trace output and usage in Langfuse
    lf.set_current_trace_io(output={"answer": output_text})
    lf.update_current_span(
        metadata={
            "model": "openai:gpt-4o-mini",
            "tool_calls_count": len(tool_calls),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    )

    return {
        "output": output_text,
        "tool_calls": tool_calls,
        "usage": usage_dict,
    }


@observe(name="search_products_direct", as_type="retriever")
async def search_products_direct(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Performs direct vector similarity search on the products table."""
    logger.info('Direct product search: query="%s", limit=%d', query, limit)

    lf = get_langfuse()
    lf.set_current_trace_io(input={"query": query, "limit": limit})

    openai = AsyncOpenAI()
    embedding = await openai.embeddings.create(
        input=query,
        model="text-embedding-3-small",
    )
    embedding_json = pydantic_core.to_json(embedding.data[0].embedding).decode()

    async with database_connect(False) as pool:
        rows = await pool.fetch(
            """
            SELECT productid, productname, productbrand, gender, priceinr, description, primarycolor,
                   (embedding <-> $1) AS distance
            FROM products
            ORDER BY embedding <-> $1
            LIMIT $2
            """,
            embedding_json,
            limit,
        )

    results = [
        {
            "productid": r["productid"],
            "productname": r["productname"],
            "productbrand": r["productbrand"],
            "gender": r["gender"],
            "priceinr": float(r["priceinr"]),
            "description": r["description"],
            "primarycolor": r["primarycolor"],
            "distance": float(r["distance"]),
        }
        for r in rows
    ]
    logger.info("Direct product search returned %d items", len(results))
    lf.set_current_trace_io(output={"result_count": len(results)})
    return results


@observe(name="search_contracts_direct", as_type="retriever")
async def search_contracts_direct(
    query: str,
    limit: int = 8,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Performs direct vector similarity search on the contracts table with optional category filtering."""
    filter_active = bool(category)
    if filter_active:
        logger.info(
            '🔍 [Metadata Filtering ACTIVE] Direct contracts search: category="%s", query="%s", limit=%d',
            category,
            query,
            limit,
        )
    else:
        logger.info(
            '🔍 [Metadata Filtering INACTIVE] Direct contracts search: unconstrained query="%s", limit=%d',
            query,
            limit,
        )

    lf = get_langfuse()
    lf.set_current_trace_io(
        input={"query": query, "limit": limit, "category": category}
    )

    openai = AsyncOpenAI()
    embedding = await openai.embeddings.create(
        input=query,
        model="text-embedding-3-small",
    )
    embedding_json = pydantic_core.to_json(embedding.data[0].embedding).decode()

    async with database_connect(False) as pool:
        if category:
            logger.info(
                '⚡ Executing GIN-indexed array containment query: WHERE $3 = ANY(category) [category="%s"]',
                category,
            )
            rows = await pool.fetch(
                """
                SELECT contract_id, document_name, chunk_id, page, text, category,
                       (embedding <-> $1) AS distance
                FROM contracts
                WHERE $3 = ANY(category)
                ORDER BY embedding <-> $1
                LIMIT $2
                """,
                embedding_json,
                limit,
                category,
            )
        else:
            logger.info(
                "⚡ Executing standard vector distance query without category filter"
            )
            rows = await pool.fetch(
                """
                SELECT contract_id, document_name, chunk_id, page, text, category,
                       (embedding <-> $1) AS distance
                FROM contracts
                ORDER BY embedding <-> $1
                LIMIT $2
                """,
                embedding_json,
                limit,
            )

    results = [
        {
            "contract_id": r["contract_id"],
            "document_name": r["document_name"],
            "chunk_id": r["chunk_id"],
            "page": r["page"],
            "text": r["text"],
            "category": list(r["category"]) if r["category"] else [],
            "distance": float(r["distance"]),
        }
        for r in rows
    ]

    logger.info(
        '📊 [Metadata Filtering Results] Direct search returned %d items (category_filter="%s")',
        len(results),
        category or "None",
    )
    for idx, r in enumerate(results, 1):
        has_target = category in r["category"] if category else True
        logger.info(
            '  ├─ [%d/%d] doc="%s" chunk=%d page=%s dist=%.4f cats=%s %s',
            idx,
            len(results),
            r["document_name"],
            r["chunk_id"],
            r["page"] or "N/A",
            r["distance"],
            r["category"],
            "✅ MATCH" if has_target else "❌ MISMATCH",
        )

    lf.set_current_trace_io(
        output={"result_count": len(results), "category": category}
    )
    return results


async def get_contract_categories() -> list[str]:
    """Returns a sorted list of all distinct contract categories currently in the database."""
    async with database_connect(False) as pool:
        has_cat = await pool.fetchval(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'contracts' AND column_name = 'category'"
        )
        if not has_cat:
            return []
        rows = await pool.fetch(
            "SELECT DISTINCT cat FROM contracts, unnest(category) AS cat ORDER BY cat"
        )
        return [r["cat"] for r in rows if r["cat"]]
