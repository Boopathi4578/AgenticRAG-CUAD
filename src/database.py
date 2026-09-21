import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import asyncpg

from typing_extensions import AsyncGenerator

logger = logging.getLogger("agentic_rag.database")

# Path to SQL schema definition
SCHEMA_PATH = Path(__file__).parent.parent / "schema.sql"


def load_db_schema() -> str:
    """Reads DB schema SQL from file."""
    if SCHEMA_PATH.exists():
        return SCHEMA_PATH.read_text(encoding="utf-8")
    return ""


DB_SCHEMA = load_db_schema()


@asynccontextmanager
async def database_connect(
    create_db: bool = False,
) -> AsyncGenerator[asyncpg.Pool, None]:
    """Establishes and yields a PostgreSQL pool connection."""
    server_dsn = os.getenv(
        "POSTGRES_SERVER_DSN", "postgresql://postgres:postgres@localhost:5432"
    )
    database = os.getenv("POSTGRES_DATABASE", "agentic_rag")
    if create_db:
        logger.info(
            'Connecting to server DSN to ensure database "%s" exists', database
        )
        conn = await asyncpg.connect(server_dsn)
        try:
            db_exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", database
            )
            if not db_exists:
                logger.info('Creating database "%s"', database)
                await conn.execute(f'CREATE DATABASE "{database}"')
                logger.info('Created database "%s"', database)
        finally:
            await conn.close()

    pool = await asyncpg.create_pool(f"{server_dsn}/{database}")
    try:
        yield pool
    finally:
        await pool.close()


async def check_db_status() -> dict[str, Any]:
    """Checks PostgreSQL connectivity, database existence, and table stats."""
    server_dsn = os.getenv(
        "POSTGRES_SERVER_DSN", "postgresql://postgres:postgres@localhost:5432"
    )
    database = os.getenv("POSTGRES_DATABASE", "agentic_rag")
    logger.debug('database_status_check started (database="%s")', database)
    # 1. Test server connection
    try:
        conn = await asyncpg.connect(server_dsn, timeout=3.0)
        db_exists = bool(
            await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", database
            )
        )
        await conn.close()
    except Exception as e:
        logger.error("Database server unreachable at %s: %s", server_dsn, e)
        return {
            "status": "offline",
            "server_reachable": False,
            "database_exists": False,
            "tables_initialized": False,
            "product_count": 0,
            "contract_count": 0,
            "error": f"Server unreachable: {e}",
            "database_name": database,
        }

    if not db_exists:
        logger.warning(
            'PostgreSQL server online, but database "%s" does not exist', database
        )
        return {
            "status": "offline",
            "server_reachable": True,
            "database_exists": False,
            "tables_initialized": False,
            "product_count": 0,
            "contract_count": 0,
            "error": f'Database "{database}" does not exist yet',
            "database_name": database,
        }

    # 2. Check tables and counts in database
    try:
        async with database_connect(False) as pool:
            prod_table_exists = await pool.fetchval(
                "SELECT 1 FROM information_schema.tables WHERE table_name = 'products'"
            )
            contract_table_exists = await pool.fetchval(
                "SELECT 1 FROM information_schema.tables WHERE table_name = 'contracts'"
            )

            if not (prod_table_exists and contract_table_exists):
                logger.warning("Database connected, but schema tables are missing")
                return {
                    "status": "offline",
                    "server_reachable": True,
                    "database_exists": True,
                    "tables_initialized": False,
                    "product_count": 0,
                    "contract_count": 0,
                    "error": "Tables (products/contracts) not created yet",
                    "database_name": database,
                }

            p_count = await pool.fetchval("SELECT count(*) FROM products") or 0
            c_count = await pool.fetchval("SELECT count(*) FROM contracts") or 0

            # Stage 2: category statistics
            category_count = 0
            top_categories: list[dict[str, Any]] = []
            has_category_col = await pool.fetchval(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'contracts' AND column_name = 'category'"
            )
            if has_category_col and c_count > 0:
                category_count = (
                    await pool.fetchval(
                        "SELECT count(DISTINCT cat) FROM contracts, unnest(category) AS cat"
                    )
                    or 0
                )
                rows = await pool.fetch(
                    "SELECT cat, count(*) AS n "
                    "FROM contracts, unnest(category) AS cat "
                    "GROUP BY cat ORDER BY n DESC LIMIT 10"
                )
                top_categories = [
                    {"category": r["cat"], "count": int(r["n"])} for r in rows
                ]

            logger.info(
                'DB Online: database="%s", products=%d, contracts=%d, categories=%d',
                database,
                p_count,
                c_count,
                category_count,
            )

            return {
                "status": "online",
                "server_reachable": True,
                "database_exists": True,
                "tables_initialized": True,
                "product_count": int(p_count),
                "contract_count": int(c_count),
                "category_count": int(category_count),
                "top_categories": top_categories,
                "error": None,
                "database_name": database,
            }
    except Exception as e:
        logger.error('Error querying database "%s": %s', database, e)
        return {
            "status": "offline",
            "server_reachable": True,
            "database_exists": True,
            "tables_initialized": False,
            "product_count": 0,
            "contract_count": 0,
            "error": str(e),
            "database_name": database,
        }


async def migrate_contracts_schema(pool: asyncpg.Pool) -> None:
    """Migrates the contracts table from Stage 1 to Stage 2 schema.

    Detects old column names and applies ALTER TABLE renames + additions
    to preserve existing data.
    """
    has_old_col = await pool.fetchval(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'contracts' AND column_name = 'file_name'"
    )
    if not has_old_col:
        logger.info(
            "Contracts table already using Stage 2 schema (or does not exist yet)"
        )
        return

    logger.info("Migrating contracts table from Stage 1 → Stage 2 schema")
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Rename columns
            await conn.execute(
                "ALTER TABLE contracts RENAME COLUMN id TO contract_id"
            )
            await conn.execute(
                "ALTER TABLE contracts RENAME COLUMN file_name TO document_name"
            )
            await conn.execute(
                "ALTER TABLE contracts RENAME COLUMN chunk_index TO chunk_id"
            )
            await conn.execute(
                "ALTER TABLE contracts RENAME COLUMN content TO text"
            )

            # Add new columns
            await conn.execute(
                "ALTER TABLE contracts ADD COLUMN IF NOT EXISTS "
                "category TEXT[] NOT NULL DEFAULT '{\"Uncategorized\"}'"
            )
            await conn.execute(
                "ALTER TABLE contracts ADD COLUMN IF NOT EXISTS page INTEGER"
            )

            # Add GIN index on category
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_contracts_category "
                "ON contracts USING GIN (category)"
            )

    logger.info("Migration complete: contracts table now Stage 2")


async def initialize_database() -> None:
    """Initializes the database and executes schema.sql.

    If an existing Stage 1 contracts table is detected, migrates it
    to the Stage 2 schema via ALTER TABLE (preserving data).
    """
    logger.info("Initializing database and applying schema")
    async with database_connect(create_db=True) as pool:
        # Check for existing Stage 1 contracts table before applying schema
        has_contracts = await pool.fetchval(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'contracts'"
        )
        if has_contracts:
            await migrate_contracts_schema(pool)
        else:
            # Fresh install — apply full schema
            async with pool.acquire() as conn, conn.transaction():
                await conn.execute(DB_SCHEMA)

        # Ensure products table exists (idempotent CREATE IF NOT EXISTS)
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Only run the products + extension part if contracts already existed
                if has_contracts:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    await conn.execute(
                        DB_SCHEMA.split("CREATE TABLE IF NOT EXISTS contracts")[0]
                    )

    logger.info("Database initialized successfully with schema")
