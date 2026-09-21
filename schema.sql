CREATE EXTENSION IF NOT EXISTS vector;

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
);

CREATE INDEX IF NOT EXISTS idx_products_embedding ON products USING hnsw (embedding vector_l2_ops);

CREATE TABLE IF NOT EXISTS contracts (
    contract_id SERIAL PRIMARY KEY,
    document_name TEXT NOT NULL,
    category TEXT[] NOT NULL DEFAULT '{"Uncategorized"}',
    chunk_id INTEGER NOT NULL DEFAULT 0,
    page INTEGER,
    text TEXT NOT NULL,
    embedding vector(1536),
    UNIQUE (document_name, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_contracts_embedding ON contracts USING hnsw (embedding vector_l2_ops);
CREATE INDEX IF NOT EXISTS idx_contracts_category ON contracts USING GIN (category);
