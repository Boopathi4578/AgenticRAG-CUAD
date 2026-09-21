import asyncio
import logging
import os
import time
from collections.abc import Coroutine
from typing import Any


import streamlit as st
from dotenv import load_dotenv

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Load environment variables
load_dotenv()

# Initialise Langfuse tracing client
from src.tracing import configure_langfuse
configure_langfuse()

from estimate_tokens import (
    DEFAULT_CONTRACTS_DIR,
    DEFAULT_CSV_PATH,
    estimate_contracts,
    estimate_products,
    get_encoder,
)
from src.agent import (
    get_contract_categories,
    run_agent_with_details,
    search_contracts_direct,
    search_products_direct,
)
from src.database import check_db_status, initialize_database
from src.ingestion import build_contracts_db, build_search_db


def run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Safely runs an async coroutine within Streamlit's execution thread."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(coro)).result()
    else:
        return asyncio.run(coro)


# --- Page Configuration ---
st.set_page_config(
    page_title="Agentic RAG — Catalogue & Contracts",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .status-badge-online {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .status-badge-offline {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .metric-pill {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
    }
    .tool-box {
        background: #f8f9fa;
        border-left: 4px solid #4f46e5;
        padding: 10px 14px;
        border-radius: 4px;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .product-card {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .contract-card {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
        background-color: #fafafa;
    }
    .category-badge {
        background-color: #ede9fe;
        color: #5b21b6;
        border: 1px solid #ddd6fe;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        display: inline-block;
        margin: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_status" not in st.session_state:
    st.session_state.db_status = None
if "last_status_check" not in st.session_state:
    st.session_state.last_status_check = 0


def refresh_db_status():
    """Fetches the latest database connectivity and row counts."""
    with st.spinner("Checking database status..."):
        status = run_async(check_db_status())
        st.session_state.db_status = status
        st.session_state.last_status_check = time.time()
    return status


# Initial check if not yet loaded or TTL expired
if st.session_state.db_status is None or (
    time.time() - st.session_state.last_status_check > 30
):
    refresh_db_status()

db_status = st.session_state.db_status or {}
is_online = db_status.get("status") == "online"

# --- Sidebar: System Status & Management ---
with st.sidebar:
    st.title("⚙️ System Status")

    # Online / Offline Badge
    if is_online:
        st.markdown(
            '<div class="status-badge-online">🟢 Database Online</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge-offline">🔴 Database Offline</div>',
            unsafe_allow_html=True,
        )

    st.caption(f"Database: `{db_status.get('database_name', 'agentic_rag')}`")

    if st.button("🔄 Refresh Status", use_container_width=True):
        db_status = refresh_db_status()
        is_online = db_status.get("status") == "online"
        st.rerun()

    if not is_online and db_status.get("error"):
        st.warning(f"**Issue:** {db_status.get('error')}")

    # Database Statistics
    st.markdown("### 📊 Data Assets")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Products", db_status.get("product_count", 0))
    with col2:
        st.metric("Contracts", db_status.get("contract_count", 0))
    with col3:
        st.metric("Categories", db_status.get("category_count", 0))

    top_cats = db_status.get("top_categories", [])
    if top_cats:
        with st.expander("Top CUAD Categories", expanded=False):
            for tc in top_cats:
                st.caption(f"• **{tc['category']}**: {tc['count']} chunks")

    # Environment Diagnostics
    st.markdown("### 🔑 Environment")
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_langfuse = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))

    st.write("OpenAI API Key:", "✅ Configured" if has_openai else "❌ Missing")
    st.write(
        "Langfuse Tracing:", "✅ Connected" if has_langfuse else "⚪ Keys Not Set"
    )

    if not has_openai:
        st.error(
            "Missing `OPENAI_API_KEY` in `.env`. Retrieval & Agent queries require OpenAI embeddings & LLM."
        )

    # Database Management Actions
    st.markdown("### 🛠️ Database Setup")
    with st.expander("Database & Ingestion Controls", expanded=not is_online):
        if not db_status.get("database_exists") or not db_status.get(
            "tables_initialized"
        ):
            if st.button(
                "Initialize DB & Schema", use_container_width=True, type="primary"
            ):
                try:
                    with st.spinner("Creating database & executing schema.sql..."):
                        run_async(initialize_database())
                    st.success("Database & tables initialized!")
                    refresh_db_status()
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to initialize: {e}")

        if st.button("📥 Ingest Sample Products", use_container_width=True):
            try:
                enc = get_encoder()
                estimate = estimate_products(DEFAULT_CSV_PATH, enc)
                st.markdown("##### 🔢 Token Estimate")
                m1, m2, m3 = st.columns(3)
                m1.metric("Products", estimate["product_count"])
                m2.metric("Tokens", f"{estimate['total_tokens']:,}")
                m3.metric("Est. Cost", f"${estimate['estimated_cost_usd']:.4f}")
                with st.spinner("Ingesting sample_products.csv into pgvector..."):
                    run_async(build_search_db())
                st.success("Products ingested successfully!")
                refresh_db_status()
                st.rerun()
            except Exception as e:
                st.error(f"Product ingestion error: {e}")

        if st.button("📥 Ingest Legal Contracts", use_container_width=True):
            try:
                enc = get_encoder()
                estimate = estimate_contracts(DEFAULT_CONTRACTS_DIR, enc)
                st.markdown("##### 🔢 Token Estimate")
                m1, m2 = st.columns(2)
                m1.metric("Files", estimate["file_count"])
                m2.metric("~Chunks", f"{estimate['total_chunks']:,}")
                m3, m4 = st.columns(2)
                m3.metric("~Tokens", f"{estimate['total_tokens']:,}")
                m4.metric("~Est. Cost", f"${estimate['estimated_cost_usd']:.4f}")
                with st.spinner(
                    "Ingesting contracts from data/contracts/ into pgvector..."
                ):
                    run_async(build_contracts_db())
                st.success("Contracts ingested successfully!")
                refresh_db_status()
                st.rerun()
            except Exception as e:
                st.error(f"Contract ingestion error: {e}")

    # Chat controls
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "Agentic RAG powered by Pydantic AI, OpenAI GPT-4o-mini, PostgreSQL pgvector & Logfire."
    )


# --- Main Window Header ---
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("🛍️ Agentic RAG Catalogue & Contracts Tool")
    st.caption(
        "Intelligent Multi-Source RAG with autonomous tool-routing between E-Commerce Catalog and Legal Contracts"
    )

with header_col2:
    st.write("")
    if is_online:
        st.markdown(
            f'<div style="text-align: right; margin-top: 10px;"><span class="status-badge-online">🟢 DB Online ({db_status.get("product_count", 0)} products, {db_status.get("contract_count", 0)} contracts)</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="text-align: right; margin-top: 10px;"><span class="status-badge-offline">🔴 DB Offline</span></div>',
            unsafe_allow_html=True,
        )

# --- Navigation Tabs ---
tab_chat, tab_explorer, tab_about = st.tabs(
    [
        "💬 AI Agent Chat",
        "🔎 Vector Search Explorer",
        "ℹ️ System Info & Architecture",
    ]
)

# ==============================================================================
# TAB 1: AI Agent Chat
# ==============================================================================
with tab_chat:
    # Notice banner if database is offline or empty
    if not is_online:
        st.info(
            "💡 **Database is currently offline or uninitialized.** "
            "Use the **'Database Setup'** section in the left sidebar to initialize the database and ingest products or contracts."
        )
    elif (
        db_status.get("product_count", 0) == 0
        and db_status.get("contract_count", 0) == 0
    ):
        st.warning(
            "⚠️ Both products and contracts tables are empty. Click **'Ingest Sample Products'** or **'Ingest Legal Contracts'** in the sidebar to populate vector data."
        )

    # Sample query suggestion buttons
    st.markdown("**Sample Queries to Try:**")
    suggest_col1, suggest_col2, suggest_col3, suggest_col4 = st.columns(4)

    sample_query = None
    with suggest_col1:
        if st.button("🛍️ *'Black running shoes under 5000 INR'*", width="stretch"):
            sample_query = "Find black running shoes under 5000 INR with their brand and description."
    with suggest_col2:
        if st.button("📄 *'Anti-Assignment clauses'*", width="stretch"):
            sample_query = "Search only clauses of category = 'Anti-Assignment'"
    with suggest_col3:
        if st.button("📄 *'Termination terms in agreements'*", width="stretch"):
            sample_query = "What are the termination conditions and notice periods in agency agreements?"
    with suggest_col4:
        if st.button("🔀 *'Contracts for sportswear/shoes'*", width="stretch"):
            sample_query = "Are there any contracts or agreements related to sportswear, athletic endorsements, or shoe brands?"

    # Display past conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool_calls"):
                hist_total_chunks = 0
                for tc in msg["tool_calls"]:
                    raw = tc.get("result") or ""
                    hist_total_chunks += max(1, raw.count("\n# ")) if raw else 0

                with st.expander(
                    f"🔍 Agent Tool Calls & Retrieved Sources — {hist_total_chunks} chunk(s) via {len(msg['tool_calls'])} tool call(s)",
                    expanded=False,
                ):
                    for idx, tc in enumerate(msg["tool_calls"], 1):
                        tool_name = tc.get("tool_name", "tool")
                        args = tc.get("args", {})
                        query_arg = args.get("search_query", str(args))
                        cat_arg = args.get("category")
                        raw_result = tc.get("result") or ""
                        chunk_count = max(1, raw_result.count("\n# ")) if raw_result else 0
                        st.markdown(f"**Step {idx}: `{tool_name}`** — {chunk_count} chunk(s) retrieved")
                        if cat_arg:
                            st.caption(
                                f'Search Query: *"{query_arg}"* • 🏷️ **Category Filter:** `{cat_arg}`'
                            )
                            st.info(
                                f"⚡ **Metadata Filter Applied:** PostgreSQL GIN query `WHERE '{cat_arg}' = ANY(category)`"
                            )
                        else:
                            st.caption(f'Search Query: *"{query_arg}"*')
                        if raw_result:
                            st.text_area(
                                f"Retrieved Context ({tool_name}) — {chunk_count} chunk(s)",
                                raw_result,
                                height=200,
                                key=f"hist_tool_{msg.get('id', 0)}_{idx}",
                            )
            if msg.get("usage"):
                u = msg["usage"]
                st.caption(
                    f"⏱️ Tokens: {u.get('total_tokens', 0)} ({u.get('input_tokens', 0)} in, {u.get('output_tokens', 0)} out) | Tool Calls: {u.get('tool_calls_count', 0)}"
                )


    # Handle chat input (either from user typing or clicking sample query button)
    user_prompt = st.chat_input("Ask about products, legal contracts, or both...")
    active_prompt = sample_query or user_prompt

    if active_prompt:
        # Append and display user message
        msg_id = len(st.session_state.messages) + 1
        st.session_state.messages.append(
            {"role": "user", "content": active_prompt, "id": msg_id}
        )
        with st.chat_message("user"):
            st.markdown(active_prompt)

        # Run agent
        with st.chat_message("assistant"):
            status_container = st.status(
                "🤖 Agent thinking & evaluating data sources...", expanded=True
            )
            start_time = time.time()
            try:
                result = run_async(run_agent_with_details(active_prompt))
                elapsed = time.time() - start_time

                status_container.update(
                    label=f"✅ Response synthesized in {elapsed:.2f}s",
                    state="complete",
                    expanded=False,
                )

                output_text = result["output"]
                tool_calls = result.get("tool_calls", [])
                usage = result.get("usage", {})

                st.markdown(output_text)

                # Show tool execution details
                if tool_calls:
                    # Count total retrieved chunks across all tool calls
                    total_chunks = 0
                    for tc in tool_calls:
                        raw = tc.get("result") or ""
                        # Each chunk is separated by a double newline + "# " header
                        total_chunks += max(1, raw.count("\n# ")) if raw else 0

                    with st.expander(
                        f"🔍 Agent Tool Calls & Retrieved Sources — {total_chunks} chunk(s) via {len(tool_calls)} tool call(s)",
                        expanded=True,
                    ):
                        for idx, tc in enumerate(tool_calls, 1):
                            tool_name = tc.get("tool_name", "tool")
                            args = tc.get("args", {})
                            query_arg = args.get("search_query", str(args))
                            cat_arg = args.get("category")
                            raw_result = tc.get("result") or ""
                            # Count chunks in this tool's result
                            chunk_count = max(1, raw_result.count("\n# ")) if raw_result else 0

                            st.markdown(f"**Step {idx}: `{tool_name}`** — {chunk_count} chunk(s) retrieved")
                            if cat_arg:
                                st.caption(
                                    f'Search Query: *"{query_arg}"* • 🏷️ **Category Filter:** `{cat_arg}`'
                                )
                                st.info(
                                    f"⚡ **Metadata Filter Applied:** PostgreSQL GIN query `WHERE '{cat_arg}' = ANY(category)`"
                                )
                            else:
                                st.caption(f'Search Query: *"{query_arg}"*')
                            if raw_result:
                                st.text_area(
                                    f"Retrieved Context ({tool_name}) — {chunk_count} chunk(s)",
                                    raw_result,
                                    height=250,
                                    key=f"curr_tool_{msg_id}_{idx}",
                                )


                st.caption(
                    f"⏱️ Response time: {elapsed:.2f}s | Tokens: {usage.get('total_tokens', 0)} ({usage.get('input_tokens', 0)} in, {usage.get('output_tokens', 0)} out) | Tool Calls: {len(tool_calls)}"
                )

                # Save assistant response to session state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": output_text,
                        "tool_calls": tool_calls,
                        "usage": usage,
                        "id": msg_id + 1,
                    }
                )

            except Exception as e:
                status_container.update(label="❌ Error running agent", state="error")
                st.error(f"Agent execution failed: {e}")
                logging.getLogger("agentic_rag.app").error(
                    "Agent error in UI", exc_info=True
                )


# ==============================================================================
# TAB 2: Vector Search Explorer
# ==============================================================================
with tab_explorer:
    st.markdown("### 🔎 Direct Vector Similarity Search")
    st.caption(
        "Inspect and test similarity matching against PostgreSQL pgvector without LLM agent reasoning."
    )

    exp_tab_prod, exp_tab_cont = st.tabs(
        ["🛍️ Product Catalog Search", "📄 Legal Contracts Search"]
    )

    with exp_tab_prod:
        col_q, col_k = st.columns([4, 1])
        with col_q:
            prod_query = st.text_input(
                "Product Search Query",
                value="cotton casual shirt",
                key="prod_search_input",
            )
        with col_k:
            prod_limit = st.slider(
                "Max Results", min_value=1, max_value=12, value=4, key="prod_search_k"
            )

        if st.button("Search Products", type="primary", key="btn_prod_search"):
            if not is_online:
                st.error(
                    "Database is offline. Please initialize and ingest products first."
                )
            else:
                with st.spinner("Searching product embeddings..."):
                    try:
                        results = run_async(
                            search_products_direct(prod_query, limit=prod_limit)
                        )
                        if not results:
                            st.info("No products found.")
                        else:
                            st.success(f"Found {len(results)} matching products:")
                            for p in results:
                                with st.container():
                                    st.markdown(
                                        f"""
                                        <div class="product-card">
                                            <h4>{p["productname"]} <span style="font-size:0.8rem; color:#64748b;">(ID: {p["productid"]})</span></h4>
                                            <p><strong>Brand:</strong> {p["productbrand"]} | <strong>Gender:</strong> {p["gender"]} | <strong>Color:</strong> {p["primarycolor"]} | <strong>Price:</strong> ₹{p["priceinr"]:,.2f}</p>
                                            <p style="color:#334155;">{p["description"]}</p>
                                            <p style="font-size:0.75rem; color:#94a3b8; margin:0;">Vector Distance: {p["distance"]:.4f}</p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                    except Exception as e:
                        st.error(f"Search failed: {e}")

    with exp_tab_cont:
        col_cq, col_cat, col_ck = st.columns([3, 2, 1])
        with col_cq:
            cont_query = st.text_input(
                "Contract Search Query",
                value="licensing and royalty fees",
                key="cont_search_input",
            )
        with col_cat:
            categories = run_async(get_contract_categories()) if is_online else []
            cat_options = ["All Categories"] + categories
            selected_cat = st.selectbox(
                "Clause Category Filter", cat_options, key="cont_cat_filter"
            )
            filter_cat = None if selected_cat == "All Categories" else selected_cat
        with col_ck:
            cont_limit = st.slider(
                "Max Results", min_value=1, max_value=12, value=4, key="cont_search_k"
            )

        if st.button("Search Contracts", type="primary", key="btn_cont_search"):
            if not is_online:
                st.error(
                    "Database is offline. Please initialize and ingest contracts first."
                )
            else:
                filter_label = f" (Category: {filter_cat})" if filter_cat else ""
                with st.spinner(f"Searching contract embeddings{filter_label}..."):
                    try:
                        results = run_async(
                            search_contracts_direct(
                                cont_query, limit=cont_limit, category=filter_cat
                            )
                        )
                        if not results:
                            st.info("No contract chunks found matching criteria.")
                        else:
                            if filter_cat:
                                verified_count = sum(
                                    1
                                    for c in results
                                    if filter_cat in c.get("category", [])
                                )
                                st.info(
                                    f"🔍 **Metadata Filter Active:** `category = '{filter_cat}'`  \n"
                                    f"⚡ **PostgreSQL GIN Query:** `WHERE '{filter_cat}' = ANY(category)` using `idx_contracts_category`  \n"
                                    f"✅ **Verification Log:** {verified_count}/{len(results)} retrieved chunks confirmed to belong to `{filter_cat}`."
                                )
                            st.success(
                                f"Found {len(results)} matching contract chunks:"
                            )
                            for c in results:
                                cat_badges = " ".join(
                                    f'<span class="category-badge">{cat}</span>'
                                    for cat in c.get("category", [])
                                )
                                page_info = (
                                    f"Page {c['page']}" if c.get("page") else "Page N/A"
                                )
                                with st.container():
                                    st.markdown(
                                        f"""
                                        <div class="contract-card">
                                            <h4>📄 {c["document_name"]} <span style="font-size:0.8rem; color:#64748b;">(Chunk #{c["chunk_id"]} • {page_info})</span></h4>
                                            <div style="margin: 6px 0 8px 0;">{cat_badges}</div>
                                            <p style="font-size:0.75rem; color:#94a3b8; margin:0;">Vector Distance: {c["distance"]:.4f}</p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                    with st.expander(
                                        "View Full Chunk Content", expanded=False
                                    ):
                                        st.text(c["text"])
                    except Exception as e:
                        st.error(f"Search failed: {e}")


# ==============================================================================
# TAB 3: System Info & Architecture
# ==============================================================================
with tab_about:
    st.markdown("### 🏛️ Architecture Overview")
    st.markdown(
        """
        This application implements an **Agentic Retrieval-Augmented Generation (RAG)** architecture with **Stage 2 Metadata Filtering**:

        - **Orchestration Agent**: Built with `pydantic-ai`, running `openai:gpt-4o-mini`.
        - **Vector Database**: PostgreSQL with the `pgvector` extension and HNSW indexing (`vector_l2_ops` Euclidean distance `<->`).
        - **Metadata Filtering (Stage 2)**: GIN index on `category TEXT[]` array column enabling high-speed containment queries (`$cat = ANY(category)`).
        - **Embedding Model**: OpenAI `text-embedding-3-small` (1536 dimensions).
        - **Multi-Source Knowledge Bases**:
          1. **E-Commerce Catalog**: Products with brand, category, gender, color, pricing, and descriptions.
          2. **Legal Contracts**: SEC business contract filings chunked with paragraph/sentence boundaries, enriched with CUAD 41 clause categories and estimated page numbers.
        - **Observability**: Fully instrumented with `langfuse` for LLM traces, tool-call spans, and token-cost tracking, plus Python standard logging for infrastructure-level events.
        """
    )
