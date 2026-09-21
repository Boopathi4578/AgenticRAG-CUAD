"""Stage 2: Metadata Filtering (contract_id, document_name, category[], chunk_id, page, text)

Revision ID: 002_stage_2
Revises: 001_stage_1
Create Date: 2026-09-08 16:30:01.000000

Migration & Revert Path:
  - upgrade():   Stage 1 -> Stage 2 (adds metadata columns, GIN index, renames columns)
  - downgrade(): Stage 2 -> Stage 1 (reverts column names, removes category & page, preserves data)
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002_stage_2"
down_revision: str | None = "001_stage_1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrades contracts table from Stage 1 to Stage 2.

    Changes:
      - id -> contract_id
      - file_name -> document_name
      - chunk_index -> chunk_id
      - content -> text
      - Adds category TEXT[] NOT NULL DEFAULT '{"Uncategorized"}'
      - Adds page INTEGER
      - Adds GIN index on category for array containment queries ($cat = ANY(category))
    """
    # 1. Rename existing columns to Stage 2 naming
    op.execute("ALTER TABLE contracts RENAME COLUMN id TO contract_id")
    op.execute("ALTER TABLE contracts RENAME COLUMN file_name TO document_name")
    op.execute("ALTER TABLE contracts RENAME COLUMN chunk_index TO chunk_id")
    op.execute("ALTER TABLE contracts RENAME COLUMN content TO text")

    # 2. Add Stage 2 metadata columns
    op.execute(
        "ALTER TABLE contracts ADD COLUMN IF NOT EXISTS "
        "category TEXT[] NOT NULL DEFAULT '{\"Uncategorized\"}'"
    )
    op.execute("ALTER TABLE contracts ADD COLUMN IF NOT EXISTS page INTEGER")

    # 3. Create GIN index on category array
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_contracts_category "
        "ON contracts USING GIN (category)"
    )


def downgrade() -> None:
    """Reverts contracts table from Stage 2 back to Stage 1.

    Revert operations:
      - Drops GIN index idx_contracts_category
      - Drops page column
      - Drops category column
      - text -> content
      - chunk_id -> chunk_index
      - document_name -> file_name
      - contract_id -> id
    Existing chunk rows, embeddings, and content are fully preserved.
    """
    # 1. Drop Stage 2 index
    op.execute("DROP INDEX IF EXISTS idx_contracts_category")

    # 2. Drop Stage 2 columns
    op.execute("ALTER TABLE contracts DROP COLUMN IF EXISTS page")
    op.execute("ALTER TABLE contracts DROP COLUMN IF EXISTS category")

    # 3. Rename columns back to Stage 1 naming
    op.execute("ALTER TABLE contracts RENAME COLUMN text TO content")
    op.execute("ALTER TABLE contracts RENAME COLUMN chunk_id TO chunk_index")
    op.execute("ALTER TABLE contracts RENAME COLUMN document_name TO file_name")
    op.execute("ALTER TABLE contracts RENAME COLUMN contract_id TO id")
