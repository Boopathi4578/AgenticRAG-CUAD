"""CUAD annotation loader and relevance matching utilities.

This module is the single source of truth for:
  - Loading data/contracts_annotations.json
  - Token-level text similarity (for soft relevance scoring)
  - Character-offset overlap (mirrors ingestion.py's assign_categories logic)
  - Multi-signal relevance decision
  - Stratified sampling of (doc_name, Annotation) pairs per category

Relevance signals — applied in priority order:
  1. Character-offset overlap  (exact same logic as ingestion.py assign_categories)
     → annotation span overlaps the chunk's [start_char, end_char) window
     → requires chunk-level offset data from the DB (chunk_start / chunk_end)
  2. Substring containment     (strong text signal when offsets are unavailable)
     → normalised annotation text is a substring of chunk text, or vice-versa
  3. Token overlap ratio ≥ threshold  (soft text similarity fallback)
     → |tokens(ann) ∩ tokens(chunk)| / |tokens(ann)| ≥ threshold
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ANNOTATIONS_PATH = Path(__file__).parent.parent / "data" / "contracts_annotations.json"


@dataclass(frozen=True)
class Annotation:
    """A single CUAD clause annotation span within a contract."""
    category: str
    start_char: int
    end_char: int
    text: str


# ---------------------------------------------------------------------------
# Annotation loader
# ---------------------------------------------------------------------------


def load_annotations(path: Path = ANNOTATIONS_PATH) -> Dict[str, List[Annotation]]:
    """Load CUAD annotations from JSON.

    Returns:
        Dict mapping document filename (e.g. "FOO_AGREEMENT.txt")
        → ordered list of Annotation objects.
    """
    with open(path, encoding="utf-8") as f:
        raw: Dict[str, list] = json.load(f)
    return {
        doc: [
            Annotation(
                category=a["category"],
                start_char=a["start_char"],
                end_char=a["end_char"],
                text=a.get("text", ""),
            )
            for a in anns
        ]
        for doc, anns in raw.items()
    }


# ---------------------------------------------------------------------------
# Text similarity helpers
# ---------------------------------------------------------------------------


def tokenize(text: str) -> Counter:
    """Lowercase word tokenizer → token frequency Counter."""
    return Counter(re.findall(r"\b\w+\b", text.lower()))


def token_overlap_ratio(annotation_text: str, chunk_text: str) -> float:
    """Fraction of annotation tokens that appear in chunk_text.

    Formula: |tokens(ann) ∩ tokens(chunk)| / |tokens(ann)|

    Returns a value in [0.0, 1.0].  Returns 0.0 for empty annotation.
    """
    ann_tokens = tokenize(annotation_text)
    if not ann_tokens:
        return 0.0
    chunk_tokens = tokenize(chunk_text)
    intersection = sum((ann_tokens & chunk_tokens).values())
    return intersection / sum(ann_tokens.values())


# ---------------------------------------------------------------------------
# Multi-signal relevance
# ---------------------------------------------------------------------------


def offsets_overlap(
    ann_start: int,
    ann_end: int,
    chunk_start: int,
    chunk_end: int,
) -> bool:
    """True if the annotation span overlaps the chunk's character window.

    Mirrors ingestion.py:assign_categories exactly:
      ann.start_char < chunk_end and ann.end_char > chunk_start
    """
    return ann_start < chunk_end and ann_end > chunk_start


def is_relevant_chunk(
    annotation: Annotation,
    chunk_text: str,
    chunk_start: Optional[int] = None,
    chunk_end: Optional[int] = None,
    token_threshold: float = 0.5,
) -> Tuple[bool, float]:
    """Multi-signal relevance check between a CUAD annotation and a DB chunk.

    Signal priority:
      1. Character-offset overlap (exact ingestion logic) — confidence = 1.0
      2. Normalised substring containment              — confidence = 0.95
      3. Token overlap ratio ≥ token_threshold         — confidence = overlap ratio

    Args:
        annotation:      The CUAD annotation being evaluated.
        chunk_text:      The stored text of the DB chunk.
        chunk_start:     Character offset where this chunk starts in the document
                         (from ingestion; pass None if unavailable).
        chunk_end:       Character offset where this chunk ends in the document.
        token_threshold: Minimum token overlap to count as relevant (default 0.5).

    Returns:
        (is_relevant: bool, confidence: float 0–1)
    """
    # Signal 1 — character-offset overlap (strongest, mirrors ingestion)
    if chunk_start is not None and chunk_end is not None:
        if offsets_overlap(annotation.start_char, annotation.end_char, chunk_start, chunk_end):
            return True, 1.0

    # Signal 2 — normalised substring containment
    ann_norm = " ".join(annotation.text.lower().split())
    chunk_norm = " ".join(chunk_text.lower().split())
    if ann_norm and (ann_norm in chunk_norm or chunk_norm in ann_norm):
        return True, 0.95

    # Signal 3 — token overlap ratio
    overlap = token_overlap_ratio(annotation.text, chunk_text)
    return overlap >= token_threshold, round(overlap, 4)


# ---------------------------------------------------------------------------
# Stratified sampler
# ---------------------------------------------------------------------------


def sample_annotations(
    data: Dict[str, List[Annotation]],
    categories: Optional[List[str]] = None,
    limit_per_category: int = 50,
    seed: int = 42,
) -> List[Tuple[str, Annotation]]:
    """Stratified sample of (doc_name, Annotation) pairs per CUAD category.

    Args:
        data:                Output of load_annotations().
        categories:          If given, restrict sampling to these categories.
        limit_per_category:  Maximum annotations to sample per category.
        seed:                RNG seed for reproducibility.

    Returns:
        Flat list of (document_name, annotation) pairs, shuffled within each
        category's sample to prevent systematic ordering bias.
    """
    rng = random.Random(seed)

    # Group by category, skipping empty annotation texts
    by_cat: Dict[str, List[Tuple[str, Annotation]]] = defaultdict(list)
    for doc, anns in data.items():
        for ann in anns:
            if ann.text.strip():
                by_cat[ann.category].append((doc, ann))

    target_cats = categories if categories else sorted(by_cat.keys())

    result: List[Tuple[str, Annotation]] = []
    for cat in target_cats:
        items = by_cat.get(cat, [])
        if not items:
            continue
        sampled = rng.sample(items, min(limit_per_category, len(items)))
        result.extend(sampled)

    return result
