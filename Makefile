.PHONY: help dev backend-build backend-contracts backend-search \
       lint clean \
       eval-smoke eval-retrieval eval-answer eval-full

# ──────────────────────────────────────────────
# 🛍️  Agentic RAG — Catalogue & Contracts Tool
# ──────────────────────────────────────────────

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Frontend ──────────────────────────────────

dev: ## Run Streamlit frontend with watchdog auto-reload
	uv run streamlit run app.py \
		--server.runOnSave=true \
		--server.fileWatcherType=watchdog

frontend: dev ## Alias for `make dev`

# ── Backend / CLI ─────────────────────────────

backend-build: ## Ingest sample products into pgvector
	uv run python main.py build

backend-contracts: ## Ingest legal contracts from CUAD into pgvector
	uv run python main.py build-contracts

backend-search: ## Run a CLI search query (usage: make backend-search Q="your query")
	uv run python main.py search $(if $(Q),"$(Q)",)

estimate-tokens: ## Estimate embedding token usage & cost before ingestion (SOURCE=products|contracts)
	uv run python estimate_tokens.py $(if $(SOURCE),--source $(SOURCE),)

# ── Database Migrations (Alembic) ─────────────

migrate: ## Run pending database migrations (upgrade to Stage 2 head)
	uv run alembic upgrade head

rollback: ## Revert latest migration (e.g. Stage 2 -> Stage 1)
	uv run alembic downgrade -1

migrate-status: ## Show current database revision
	uv run alembic current

migrate-history: ## Show migration history
	uv run alembic history --verbose

# ── Collect Data ────────────────────────

collect-data: ## Collect contracts data from CUAD
	uv run python data_collect.py

# ── Evaluation (CUAD ground truth) ───────────

eval-smoke: ## Smoke test: 5 retrieval cases, no LLM calls, no Logfire
	uv run python -m evals.run_eval --mode retrieval --smoke --no-logfire

eval-retrieval: ## Retrieval eval: Recall@K / Precision@K / MRR / nDCG  (all categories, limit=50)
	uv run python -m evals.run_eval --mode retrieval --limit 50 --no-logfire

eval-answer: ## Answer eval: correctness + faithfulness via LLM judge  (10 cases/category, Logfire ON)
	uv run python -m evals.run_eval --mode answer --answer-limit 10

eval-full: ## Full benchmark: retrieval (50/cat) then answer eval (10/cat)
	uv run python -m evals.run_eval --mode all --limit 50 --answer-limit 10

# ── Utilities ─────────────────────────────────

lint: ## Run basic Python lint checks
	uv run python -m py_compile app.py
	uv run python -m py_compile main.py
	uv run python -m py_compile data_collect.py
	uv run python -m py_compile estimate_tokens.py
	uv run python -m py_compile src/database.py
	uv run python -m py_compile src/models.py
	uv run python -m py_compile src/ingestion.py
	uv run python -m py_compile src/agent.py
	uv run python -m py_compile alembic/env.py
	uv run python -m py_compile alembic/versions/001_stage_1_initial_schema.py
	uv run python -m py_compile alembic/versions/002_stage_2_metadata_filtering.py
	uv run python -m py_compile evals/cuad_loader.py
	uv run python -m py_compile evals/retrieval_cases.py
	uv run python -m py_compile evals/answer_cases.py
	uv run python -m py_compile evals/metrics.py
	uv run python -m py_compile evals/answer_evaluators.py
	uv run python -m py_compile evals/run_eval.py

clean: ## Remove Python caches and bytecode
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
