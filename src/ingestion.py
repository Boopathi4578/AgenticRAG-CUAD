import asyncio
import csv
import json
import logging
from pathlib import Path

import asyncpg

import pydantic_core
from openai import AsyncOpenAI
from pydantic import TypeAdapter

from src.database import DB_SCHEMA, database_connect
from src.models import ContractAnnotation, ProductData

logger = logging.getLogger("agentic_rag.ingestion")

DEFAULT_CSV_PATH = Path(__file__).parent.parent / "sample_products.csv"
DEFAULT_CONTRACTS_DIR = Path(__file__).parent.parent / "data" / "contracts"
DEFAULT_ANNOTATIONS_PATH = (
    Path(__file__).parent.parent / "data" / "contracts_annotations.json"
)

# Chunking settings for contracts
CHUNK_SIZE = 1000  # approximate tokens (chars / 4)
CHUNK_OVERLAP = 200  # overlap in tokens

# Page estimation: ~3000 characters per page for plain text
CHARS_PER_PAGE = 3000


async def read_csv_file(file_path: str) -> list[ProductData]:
    """Reads a local CSV file and converts it into a list of ProductData objects."""
    product_list = []
    with open(file_path, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            product = ProductData(
                ProductID=int(row["ProductID"]),
                ProductName=row["ProductName"],
                ProductBrand=row["ProductBrand"],
                Gender=row["Gender"],
                PriceInr=float(row["Price (INR)"]),
                NumImages=int(row["NumImages"]),
                Description=row["Description"],
                PrimaryColor=row["PrimaryColor"],
            )
            product_list.append(product)
    product_list_ta = TypeAdapter(list[ProductData])
    return product_list_ta.validate_python(product_list)


async def insert_product_data(
    sem: asyncio.Semaphore,
    openai: AsyncOpenAI,
    pool: asyncpg.Pool,
    product_data: ProductData,
) -> None:
    """Creates embeddings for product and inserts row into PostgreSQL."""
    async with sem:
        # If the product is present in DB or not
        product_id = product_data.ProductID
        exists = await pool.fetchval(
            "SELECT 1 FROM products WHERE ProductID = $1", product_id
        )
        if exists:
            logger.debug("Skipping product_id=%d (already in DB)", product_id)
            return

        embedding = await openai.embeddings.create(
            input=product_data.embedding_content(),
            model="text-embedding-3-small",
        )
        embedding_json = pydantic_core.to_json(embedding.data[0].embedding).decode()
        await pool.execute(
            "INSERT INTO products (ProductID, ProductName, ProductBrand, Gender, PriceInr, NumImages, Description, PrimaryColor, embedding) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            product_data.ProductID,
            product_data.ProductName,
            product_data.ProductBrand,
            product_data.Gender,
            product_data.PriceInr,
            product_data.NumImages,
            product_data.Description,
            product_data.PrimaryColor,
            embedding_json,
        )


async def build_search_db(csv_file_path: str | None = None):
    """Builds the vector search database from a catalog CSV file."""
    path = csv_file_path or str(DEFAULT_CSV_PATH)
    product_list = await read_csv_file(path)
    openai = AsyncOpenAI()


    async with database_connect(True) as pool:
        logger.info('create schema: executing DB_SCHEMA')
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(DB_SCHEMA)

        sem = asyncio.Semaphore(10)
        async with asyncio.TaskGroup() as tg:
            for product in product_list:
                tg.create_task(insert_product_data(sem, openai, pool, product))


# ---------------------------------------------------------------------------
# Contract ingestion — Stage 2 (metadata filtering)
# ---------------------------------------------------------------------------


def load_annotations(
    annotations_path: Path | None = None,
) -> dict[str, list[ContractAnnotation]]:
    """Loads CUAD-QA clause annotations from JSON file.

    Returns a mapping: {filename.txt: [ContractAnnotation, ...]}
    """
    path = annotations_path or DEFAULT_ANNOTATIONS_PATH
    if not path.exists():
        logger.warning(
            'Annotations file not found at %s — chunks will be "Uncategorized"', path
        )
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    result: dict[str, list[ContractAnnotation]] = {}
    for filename, anns in raw.items():
        result[filename] = [
            ContractAnnotation(
                category=a["category"],
                start_char=a["start_char"],
                end_char=a["end_char"],
                text=a.get("text", ""),
            )
            for a in anns
        ]
    logger.info("Loaded annotations for %d contract files", len(result))
    return result


def assign_categories(
    chunk_start: int,
    chunk_end: int,
    annotations: list[ContractAnnotation],
) -> list[str]:
    """Find all CUAD categories whose annotation spans overlap this chunk."""
    categories = set()
    for ann in annotations:
        # Check for any overlap between [chunk_start, chunk_end) and [ann.start_char, ann.end_char)
        if ann.start_char < chunk_end and ann.end_char > chunk_start:
            categories.add(ann.category)
    return sorted(categories) if categories else ["Uncategorized"]


def estimate_page(char_offset: int) -> int:
    """Estimates page number from character offset (~3000 chars/page)."""
    return (char_offset // CHARS_PER_PAGE) + 1


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[tuple[str, int, int]]:
    """Splits text into overlapping chunks of approximately `chunk_size` tokens.

    Uses character-based splitting (≈4 chars per token) with paragraph-aware boundaries.

    Returns:
        List of (chunk_text, start_char, end_char) tuples.
    """
    chars_per_chunk = chunk_size * 4
    chars_overlap = overlap * 4

    if len(text) <= chars_per_chunk:
        return [(text, 0, len(text))]

    chunks: list[tuple[str, int, int]] = []
    start = 0
    while start < len(text):
        end = start + chars_per_chunk

        # Try to break at a paragraph or sentence boundary
        if end < len(text):
            # Look for paragraph break near the end
            newline_pos = text.rfind("\n\n", start + chars_per_chunk // 2, end + 200)
            if newline_pos != -1:
                end = newline_pos + 2
            else:
                # Fall back to sentence boundary
                period_pos = text.rfind(". ", start + chars_per_chunk // 2, end + 100)
                if period_pos != -1:
                    end = period_pos + 2

        chunk_content = text[start:end].strip()
        if chunk_content:
            chunks.append((chunk_content, start, end))
        start = end - chars_overlap

    return chunks


async def insert_contract_chunk(
    sem: asyncio.Semaphore,
    openai: AsyncOpenAI,
    pool: asyncpg.Pool,
    document_name: str,
    chunk_id: int,
    text: str,
    categories: list[str],
    page: int | None = None,
    max_retries: int = 5,
) -> None:
    """Creates an embedding for a contract chunk and inserts it into PostgreSQL.

    Stage 2: stores metadata (category, page) alongside the embedding.
    """
    async with sem:
        exists = await pool.fetchval(
            "SELECT 1 FROM contracts WHERE document_name = $1 AND chunk_id = $2",
            document_name,
            chunk_id,
        )
        if exists:
            return

        # Retry with exponential backoff for rate limits
        for attempt in range(max_retries):
            try:
                embedding = await openai.embeddings.create(
                    input=text,
                    model="text-embedding-3-small",
                )
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait_time = 2**attempt
                    logger.warning(
                        "Rate limited, retrying in %ds (attempt %d/%d)",
                        wait_time, attempt + 1, max_retries,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise
        else:
            raise RuntimeError(
                f"Failed to create embedding after {max_retries} retries "
                f"for {document_name} chunk {chunk_id}"
            )

        embedding_json = pydantic_core.to_json(embedding.data[0].embedding).decode()
        await pool.execute(
            "INSERT INTO contracts (document_name, category, chunk_id, page, text, embedding) "
            "VALUES ($1, $2, $3, $4, $5, $6)",
            document_name,
            categories,
            chunk_id,
            page,
            text,
            embedding_json,
        )


async def build_contracts_db(
    contracts_dir: str | None = None,
    annotations_path: str | None = None,
) -> None:
    """Builds the vector search database from contract text files.

    Stage 2: loads CUAD-QA annotations to assign clause categories per chunk
    and estimates page numbers from character offsets.
    """
    dir_path = Path(contracts_dir) if contracts_dir else DEFAULT_CONTRACTS_DIR
    if not dir_path.exists():
        raise FileNotFoundError(f"Contracts directory not found: {dir_path}")

    txt_files = sorted(dir_path.glob("*.txt"))
    logger.info("Found %d contract files", len(txt_files))

    # Load CUAD-QA annotations for category assignment
    ann_path = Path(annotations_path) if annotations_path else None
    all_annotations = load_annotations(ann_path)
    matched_files = sum(1 for f in txt_files if f.name in all_annotations)
    logger.info(
        "Annotation coverage: %d/%d files have CUAD-QA annotations",
        matched_files,
        len(txt_files),
    )

    openai = AsyncOpenAI()


    async with database_connect(True) as pool:
        logger.info('create schema: executing DB_SCHEMA')
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(DB_SCHEMA)

        sem = asyncio.Semaphore(10)
        total_chunks = 0
        categorized_chunks = 0

        async with asyncio.TaskGroup() as tg:
            for txt_file in txt_files:
                text = txt_file.read_text(encoding="utf-8")
                chunks = chunk_text(text)
                file_annotations = all_annotations.get(txt_file.name, [])

                for idx, (chunk_content, start_char, end_char) in enumerate(chunks):
                    # Assign categories from CUAD-QA annotations
                    categories = assign_categories(
                        start_char, end_char, file_annotations
                    )
                    page = estimate_page(start_char)

                    if categories != ["Uncategorized"]:
                        categorized_chunks += 1

                    tg.create_task(
                        insert_contract_chunk(
                            sem,
                            openai,
                            pool,
                            document_name=txt_file.name,
                            chunk_id=idx,
                            text=chunk_content,
                            categories=categories,
                            page=page,
                        )
                    )
                    total_chunks += 1

        logger.info(
            "Ingested %d chunks from %d files (%d with CUAD categories)",
            total_chunks, len(txt_files), categorized_chunks,
        )
        logger.info(
            "Ingestion complete: %d chunks, %d categorized, %d uncategorized",
            total_chunks,
            categorized_chunks,
            total_chunks - categorized_chunks,
        )
