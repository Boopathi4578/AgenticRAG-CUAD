from dataclasses import dataclass

import asyncpg
from openai import AsyncOpenAI


@dataclass
class ProductData:
    ProductID: int
    ProductName: str
    ProductBrand: str
    Gender: str
    PriceInr: float
    NumImages: int
    Description: str
    PrimaryColor: str

    def embedding_content(self) -> str:
        """Returns concatenated text used to produce product vector embeddings."""
        return "\n\n".join(
            (
                f"ProductID: {self.ProductID}",
                f"ProductName: {self.ProductName}",
                f"ProductBrand: {self.ProductBrand}",
                f"Gender: {self.Gender}",
                f"PrimaryColor: {self.PrimaryColor}",
                f"Description: {self.Description}",
            )
        )


@dataclass
class Deps:
    openai: AsyncOpenAI
    pool: asyncpg.Pool


@dataclass
class ContractAnnotation:
    """A single CUAD clause annotation span within a contract."""

    category: str
    start_char: int
    end_char: int
    text: str
