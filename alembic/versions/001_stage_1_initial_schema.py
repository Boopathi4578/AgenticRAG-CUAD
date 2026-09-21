"""Stage 1: Initial schema for products and contracts

Revision ID: 001_stage_1
Revises:
Create Date: 2026-09-08 16:30:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_stage_1"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Creates the base Stage 1 schema (pgvector extension, products, contracts)."""
    # 1. Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 2. Products table
    op.execute("""
        CREATE TABLE IF NOT EXISTS products (
            ProductID integer PRIMARY KEY,
            ProductName text NOT NULL,
            ProductBrand text NOT NULL,
            Gender text NOT NULL,
            PriceInr decimal(10, 2) NOT NULL,
            NumImages integer NOT NULL,
            Description text NOT NULL,
            PrimaryColor text NOT NULL,
            embedding vector(1536)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_products_embedding
        ON products USING hnsw (embedding vector_l2_ops)
    """)

    # 3. Stage 1 Contracts table
    op.execute("""
        CREATE TABLE IF NOT EXISTS contracts (
            id SERIAL PRIMARY KEY,
            file_name TEXT NOT NULL,
            chunk_index INTEGER NOT NULL DEFAULT 0,
            content TEXT NOT NULL,
            embedding vector(1536),
            UNIQUE (file_name, chunk_index)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_contracts_embedding
        ON contracts USING hnsw (embedding vector_l2_ops)
    """)


def downgrade() -> None:
    """Tears down Stage 1 tables."""
    op.execute("DROP TABLE IF EXISTS contracts CASCADE")
    op.execute("DROP TABLE IF EXISTS products CASCADE")
