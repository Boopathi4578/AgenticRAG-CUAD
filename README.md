# Agentic RAG — Catalogue & Legal Contracts Tool

An Agentic RAG application using **Pydantic AI**, **Langfuse**, **OpenAI**, and **PostgreSQL (`pgvector`)** for intelligent retrieval and synthesis across e-commerce product catalogs and CUAD legal contracts.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | [Pydantic AI](https://ai.pydantic.dev/) |
| LLM / Embeddings | OpenAI `gpt-4o-mini` + `text-embedding-3-small` |
| Vector Database | PostgreSQL + `pgvector` (HNSW index) |
| Observability | [Langfuse v4](https://langfuse.com/) (`@observe`, traces, LLM spans) |
| Frontend | Streamlit |
| DB Migrations | Alembic |
| Package Manager | [uv](https://docs.astral.sh/uv/) |

---

## Project Structure

```
.
├── app.py                      # Streamlit UI (chat, vector search, DB management)
├── main.py                     # CLI entry point (build / build-contracts / search)
├── data_collect.py             # Downloads CUAD contracts + clause annotations
├── estimate_tokens.py          # Pre-ingestion token & cost estimator
├── sample_products.csv         # E-commerce product catalog (seed data)
├── schema.sql                  # PostgreSQL schema (pgvector, HNSW + GIN indexes)
├── Makefile                    # Convenience targets for dev, ingestion, evals, migrations
├── pyproject.toml              # Project dependencies (uv/hatchling)
├── alembic.ini                 # Alembic migration config
├── alembic/                    # DB migration versions
│   └── versions/
│       ├── 001_stage_1_initial_schema.py
│       └── 002_stage_2_metadata_filtering.py
├── src/
│   ├── agent.py                # Pydantic AI agent + retrieve_products / retrieve_contracts tools
│   ├── database.py             # asyncpg pool, schema init, DB health check, migration helpers
│   ├── ingestion.py            # CSV & contract ingestion, chunking, CUAD annotation assignment
│   ├── models.py               # Pydantic dataclasses: ProductData, Deps, ContractAnnotation
│   └── tracing.py              # Langfuse singleton config & @observe re-exports
├── evals/
│   ├── cuad_loader.py          # CUAD JSON loader + multi-signal relevance labelling
│   ├── retrieval_cases.py      # Retrieval Dataset builder (oracle benchmark)
│   ├── answer_cases.py         # Answer Dataset builder (41 NL question templates)
│   ├── metrics.py              # Recall@K, Precision@K, MRR, nDCG (pure Python)
│   ├── answer_evaluators.py    # LLM judge (QualityJudge via gpt-4o-mini)
│   ├── run_eval.py             # CLI eval runner (--mode retrieval|answer|all)
│   └── results/                # Timestamped JSON eval snapshots
└── data/
    ├── contracts/              # Raw CUAD contract .txt files (from data_collect.py)
    └── contracts_annotations.json  # CUAD-QA clause annotations (from data_collect.py)
```

---

## Quick Start

### 1. Clone & install dependencies

```bash
git clone <repo-url>
cd "Agentic RAG - Catalogue Tool"
uv sync
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```ini
# Required
OPENAI_API_KEY=sk-...

# Langfuse observability (optional but recommended)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# PostgreSQL (defaults to local)
POSTGRES_SERVER_DSN=postgresql://postgres:postgres@localhost:5432
POSTGRES_DATABASE=agentic_rag
```

### 3. Collect contract data (first time only)

Downloads ~510 CUAD contract text files and CUAD-QA clause annotations:

```bash
make collect-data
# or: uv run python data_collect.py
```

### 4. Estimate embedding costs (optional)

```bash
make estimate-tokens                        # both products & contracts
uv run python estimate_tokens.py --source products
uv run python estimate_tokens.py --source contracts
```

### 5. Ingest data into pgvector

```bash
make backend-build        # ingest product catalog (sample_products.csv)
make backend-contracts    # ingest CUAD legal contracts
```

### 6. Run the Streamlit UI

```bash
make dev
# or: uv run streamlit run app.py
```

---

## Streamlit UI Features

- 🟢 **Live DB Status Indicator** — real-time PostgreSQL & pgvector health badge with product/contract/category counts
- 💬 **Agentic Chat** — conversational multi-source RAG with expandable step-by-step tool traces and token usage
- 🔎 **Direct Vector Search** — test similarity search on products or contracts without LLM overhead; supports category filtering for contracts
- 🛠️ **DB Management** — initialize schema or re-ingest data with one click
- 📊 **Observability** — full Langfuse traces for all agent decisions, tool calls, and retrieval spans

---

## CLI Reference

```bash
# Ingest product catalog
uv run python main.py build

# Ingest legal contracts
uv run python main.py build-contracts

# Run a natural-language search via the agent
uv run python main.py search "Find comfortable black running shoes"
uv run python main.py search "Show termination clauses for franchise agreements"
```

---

## Agent & Tools

The Pydantic AI agent (`src/agent.py`) uses `gpt-4o-mini` and has two retrieval tools:

| Tool | Description |
|---|---|
| `retrieve_products` | Vector similarity search on the `products` table (top-8 by Euclidean distance) |
| `retrieve_contracts` | Vector similarity search on the `contracts` table with optional `category` metadata filter (GIN-indexed array containment) and fuzzy category auto-correction |

The `retrieve_contracts` tool auto-resolves category strings (case-insensitive + fuzzy difflib matching) to valid CUAD categories before querying, preventing zero-result misses from LLM spelling variation.

All tool calls are wrapped in **Langfuse observations** so every retrieval span is visible in the Langfuse dashboard.

---

## Database Schema

Two tables backed by `pgvector` HNSW indexes:

```sql
-- Products (e-commerce catalog)
CREATE TABLE products (
    ProductID integer PRIMARY KEY,
    ProductName text, ProductBrand text, Gender text,
    PriceInr decimal(10,2), NumImages integer,
    Description text, PrimaryColor text,
    embedding vector(1536)   -- HNSW index (L2)
);

-- Contracts (CUAD legal contracts, chunked with overlap)
CREATE TABLE contracts (
    contract_id SERIAL PRIMARY KEY,
    document_name TEXT,
    chunk_id INTEGER,
    category TEXT[],          -- GIN-indexed for metadata filtering
    page INTEGER,
    text TEXT,
    embedding vector(1536),   -- HNSW index (L2)
    UNIQUE (document_name, chunk_id)
);
```

Contract text is split into overlapping chunks (~1000 tokens, 200-token overlap) with paragraph-aware boundaries. Each chunk is tagged with CUAD clause categories derived from character-offset overlap with CUAD-QA annotations.

---

## Alembic Migrations

```bash
make migrate          # apply pending migrations (upgrade head)
make rollback         # revert latest migration
make migrate-status   # show current DB revision
make migrate-history  # show full migration history
```

Migrations live in `alembic/versions/`:
- `001` — initial schema (products + contracts Stage 1)
- `002` — Stage 2: adds `category TEXT[]`, `page`, GIN index, renames legacy columns

---

## Evaluation Suite (`evals/`)

Benchmarks retrieval quality and answer quality against CUAD ground-truth annotations.
See [`evals/README.md`](evals/README.md) for full details.

### Quick commands

```bash
make eval-smoke       # 5 retrieval cases, no LLM, fast sanity check
make eval-retrieval   # Recall@K / Precision@K / MRR / nDCG (limit=50/category)
make eval-answer      # Correctness + Faithfulness via LLM judge (10/category)
make eval-full        # Full benchmark: retrieval + answer
```

### Metrics targets

| Metric | Type | Target |
|---|---|---|
| Recall@K | Retrieval (deterministic) | > 0.70 |
| Precision@K | Retrieval (deterministic) | > 0.50 |
| MRR | Retrieval (deterministic) | > 0.60 |
| nDCG@K | Retrieval (deterministic) | > 0.65 |
| Correctness | LLM judge (gpt-4o-mini) | > 0.65 |
| Faithfulness | LLM judge (gpt-4o-mini) | > 0.80 |

Results are saved to `evals/results/YYYY-MM-DD_HH-MM_{mode}.json` and optionally uploaded to Logfire Experiments.

---

## Makefile Targets

```
make help               Show all available commands
make dev                Run Streamlit UI with watchdog auto-reload
make backend-build      Ingest product catalog into pgvector
make backend-contracts  Ingest CUAD legal contracts into pgvector
make backend-search     CLI agent search  (Q="your query")
make collect-data       Download CUAD contracts + clause annotations
make estimate-tokens    Token & cost estimate (SOURCE=products|contracts)
make migrate            Run Alembic migrations (upgrade head)
make rollback           Revert latest Alembic migration
make migrate-status     Show current DB revision
make migrate-history    Show full migration history
make eval-smoke         Smoke test: 5 retrieval cases, no LLM
make eval-retrieval     Full retrieval benchmark
make eval-answer        Answer quality eval (LLM judge)
make eval-full          Full benchmark: retrieval + answer
make lint               Compile-check all Python source files
make clean              Remove __pycache__ and .pyc files
```

---

## Observability with Langfuse

Every agent run, tool call, and direct retrieval is traced via **Langfuse v4**:

- `@observe(name="run_agent_with_details", as_type="agent")` — top-level agent span
- `@observe(name="search_products_direct", as_type="retriever")` — direct product search
- `@observe(name="search_contracts_direct", as_type="retriever")` — direct contract search
- Tool-level observations for `retrieve_products` and `retrieve_contracts`

Configure `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` in `.env` to enable cloud observability. Traces flush automatically on shutdown via `flush_langfuse()`.
