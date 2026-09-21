"""Deterministic retrieval metrics as pydantic_evals Evaluator subclasses.

All four metrics compare the retrieved chunk list (output of the retrieval task)
against pre-computed RelevanceLabels (expected_output built by retrieval_cases.py).

Metric definitions
──────────────────
  Recall@K     = |relevant ∩ retrieved[:K]| / |relevant|
  Precision@K  = |relevant ∩ retrieved[:K]| / K
  MRR          = 1 / rank_of_first_relevant   (0.0 if none in top-K)
  nDCG@K       = DCG@K / IDCG@K
                 DCG@K  = Σ rel_i / log2(i+1)  for i=1..K  (binary relevance)
                 IDCG@K = Σ 1     / log2(i+1)  for i=1..min(K, |relevant|)

All evaluators are @dataclass subclasses as required by pydantic_evals.
All evaluate() methods are synchronous (no LLM calls, pure Python math).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Set

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from evals.retrieval_cases import RelevanceLabels, RetrievalInput


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _retrieved_ids(output: List[Dict[str, Any]], k: int) -> List[int]:
    """Extract the top-K chunk_ids from the retrieval task output."""
    return [r["chunk_id"] for r in (output or [])[:k]]


def _relevant_set(ctx: EvaluatorContext) -> Set[int]:
    """Extract the set of relevant chunk_ids from expected_output."""
    labels: RelevanceLabels | None = ctx.expected_output
    if labels is None:
        return set()
    return set(labels.chunk_ids)


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------


@dataclass
class RecallAtK(Evaluator[RetrievalInput, List[Dict[str, Any]]]):
    """Fraction of relevant chunks that appear in the top-K retrieved results.

    Returns 1.0 when there are no relevant chunks (trivially satisfied).
    """

    k: int = 8

    def evaluate(
        self,
        ctx: EvaluatorContext[RetrievalInput, List[Dict[str, Any]]],
    ) -> float:
        relevant = _relevant_set(ctx)
        if not relevant:
            return 1.0
        retrieved = set(_retrieved_ids(ctx.output, self.k))
        return len(relevant & retrieved) / len(relevant)


# ---------------------------------------------------------------------------
# Precision@K
# ---------------------------------------------------------------------------


@dataclass
class PrecisionAtK(Evaluator[RetrievalInput, List[Dict[str, Any]]]):
    """Fraction of the top-K retrieved chunks that are relevant."""

    k: int = 8

    def evaluate(
        self,
        ctx: EvaluatorContext[RetrievalInput, List[Dict[str, Any]]],
    ) -> float:
        relevant = _relevant_set(ctx)
        retrieved = _retrieved_ids(ctx.output, self.k)
        if not retrieved:
            return 0.0
        hits = sum(1 for cid in retrieved if cid in relevant)
        return hits / len(retrieved)


# ---------------------------------------------------------------------------
# MRR (Mean Reciprocal Rank)
# ---------------------------------------------------------------------------


@dataclass
class MRR(Evaluator[RetrievalInput, List[Dict[str, Any]]]):
    """Reciprocal rank of the first relevant result in the top-K list.

    Returns 0.0 if no relevant result appears in the top-K.
    """

    k: int = 8

    def evaluate(
        self,
        ctx: EvaluatorContext[RetrievalInput, List[Dict[str, Any]]],
    ) -> float:
        relevant = _relevant_set(ctx)
        for rank, cid in enumerate(_retrieved_ids(ctx.output, self.k), start=1):
            if cid in relevant:
                return 1.0 / rank
        return 0.0


# ---------------------------------------------------------------------------
# nDCG@K
# ---------------------------------------------------------------------------


@dataclass
class NDCGAtK(Evaluator[RetrievalInput, List[Dict[str, Any]]]):
    """Normalised Discounted Cumulative Gain @ K (binary relevance).

    Formula:
      DCG@K  = Σ_{i=1}^{K}  rel_i / log2(i+1)
      IDCG@K = Σ_{i=1}^{min(K,|R|)} 1 / log2(i+1)
      nDCG@K = DCG@K / IDCG@K

    Returns 1.0 when there are no relevant chunks (trivially perfect).
    Returns 0.0 when IDCG=0 (degenerate edge case).
    """

    k: int = 8

    def evaluate(
        self,
        ctx: EvaluatorContext[RetrievalInput, List[Dict[str, Any]]],
    ) -> float:
        relevant = _relevant_set(ctx)
        if not relevant:
            return 1.0

        retrieved = _retrieved_ids(ctx.output, self.k)

        # Actual DCG
        dcg = sum(
            (1.0 if cid in relevant else 0.0) / math.log2(i + 2)
            for i, cid in enumerate(retrieved)
        )

        # Ideal DCG: best possible ordering (all relevant first)
        ideal_hits = min(len(relevant), self.k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

        if idcg == 0.0:
            return 0.0
        return dcg / idcg
