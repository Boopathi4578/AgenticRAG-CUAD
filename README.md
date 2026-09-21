# Agentic RAG - Catalogue & Legal Contracts Tool

An Agentic RAG application utilizing Pydantic AI, Logfire, OpenAI, and PostgreSQL (`pgvector`) for intelligent retrieval and synthesis across e-commerce product catalogs and legal contracts.

## Streamlit UI

Launch the interactive web UI:

```bash
uv run streamlit run app.py
```

### UI Features:
- 🟢 **Live Database Online/Offline Indicator**: Real-time status badge and health metrics for PostgreSQL and pgvector tables.
- 💬 **Interactive Agent Chat**: Conversational multi-source RAG assistant with expandable step-by-step tool execution traces and token usage statistics.
- 🔎 **Direct Vector Search Explorer**: Test similarity searches directly against products and contract embeddings without LLM overhead.
- 🛠️ **Database Management Controls**: Ingest sample products and contracts or initialize schemas with one click.
- 📊 **Observability**: Full backend logging and Logfire spans for agent decisions and retrieval calls.

## CLI Setup & Running

1. Configure `.env` (set `OPENAI_API_KEY`, optional `LOGFIRE_TOKEN`, `POSTGRES_SERVER_DSN`, and `POSTGRES_DATABASE`).
2. Sync dependencies:
   ```bash
   uv sync
   ```
3. Run CLI ingestion & search:
   ```bash
   # Ingest product catalog
   uv run main.py build

   # Ingest legal contracts
   uv run main.py build-contracts

   # Search via CLI
   uv run main.py search "Find comfortable black running shoes"
   ```

