"""Build natural-language answer evaluation cases.

Unlike the oracle retrieval benchmark (which uses gold clause text verbatim
as the query), these cases use realistic human questions to evaluate the
full RAG pipeline end-to-end:

  Question  = NL template rendered for a specific document + category
  Expected  = gold annotation text from CUAD (stored in AnswerInput.gold_text,
              not as expected_output, so Dataset type params stay clean)

The judge evaluator (QualityJudge) reads gold_text from ctx.inputs.gold_text.

Cost control
────────────
Default limit_per_category=10 keeps LLM calls manageable.
Use --answer-limit 50 for a final benchmark run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic_evals import Case, Dataset

from evals.cuad_loader import load_annotations, sample_annotations

# ---------------------------------------------------------------------------
# Natural-language question templates for all 41 CUAD categories
# {doc_short} = human-readable document abbreviation
# {category}  = raw CUAD category label (fallback template only)
# ---------------------------------------------------------------------------

NL_TEMPLATES: Dict[str, str] = {
    "Affiliate License-Licensee": (
        "What rights does the licensee have to sublicense to its affiliates under {doc_short}?"
    ),
    "Affiliate License-Licensor": (
        "Can the licensor grant licenses through its affiliates under {doc_short}?"
    ),
    "Agreement Date": "What is the effective signing date of {doc_short}?",
    "Anti-Assignment": (
        "Can either party in {doc_short} assign this agreement to a third party "
        "without the other party's prior written consent?"
    ),
    "Audit Rights": (
        "What audit or inspection rights do the parties have under {doc_short}?"
    ),
    "Cap On Liability": (
        "What is the maximum liability cap specified in {doc_short}?"
    ),
    "Change Of Control": (
        "What happens under {doc_short} if one of the parties undergoes a change of control "
        "or acquisition?"
    ),
    "Competitive Restriction Exception": (
        "Are there any carve-outs or exceptions to the competitive restrictions in {doc_short}?"
    ),
    "Covenant Not To Sue": (
        "Does {doc_short} include a covenant not to sue? If so, what does it cover?"
    ),
    "Document Name": "What is the official name or title of {doc_short}?",
    "Effective Date": "When does {doc_short} become legally effective?",
    "Exclusivity": (
        "What exclusivity obligations or exclusive dealing provisions are defined in {doc_short}?"
    ),
    "Expiration Date": (
        "When does {doc_short} expire, and under what conditions can it be renewed?"
    ),
    "Governing Law": (
        "Which jurisdiction's laws govern {doc_short}, and where must disputes be resolved?"
    ),
    "Insurance": (
        "What types and amounts of insurance are the parties required to maintain under {doc_short}?"
    ),
    "Ip Ownership Assignment": (
        "Who owns the intellectual property developed or created under {doc_short}, "
        "and are there any assignment provisions?"
    ),
    "Irrevocable Or Perpetual License": (
        "Does {doc_short} grant an irrevocable or perpetual license? "
        "Describe the scope and any conditions."
    ),
    "Joint Ip Ownership": (
        "Are there joint intellectual property ownership provisions in {doc_short}? "
        "How is ownership and control shared between the parties?"
    ),
    "License Grant": (
        "What license rights — including scope, territory, exclusivity, and sublicensing — "
        "are granted under {doc_short}?"
    ),
    "Liquidated Damages": (
        "What liquidated damages or penalties are specified in {doc_short} for breach or "
        "non-performance?"
    ),
    "Minimum Commitment": (
        "What minimum purchase, revenue, or volume commitments are required under {doc_short}?"
    ),
    "Most Favored Nation": (
        "Does {doc_short} include a most-favored-nation (MFN) or best-pricing clause?"
    ),
    "No-Solicit Of Customers": (
        "Does {doc_short} restrict either party from soliciting the other's customers "
        "during or after the agreement?"
    ),
    "No-Solicit Of Employees": (
        "Does {doc_short} restrict either party from recruiting or hiring the other's employees?"
    ),
    "Non-Compete": (
        "What non-compete or non-competition restrictions does {doc_short} impose, "
        "and for how long?"
    ),
    "Non-Disparagement": (
        "Does {doc_short} include a non-disparagement clause? "
        "What are the parties prohibited from saying?"
    ),
    "Non-Transferable License": (
        "Is the license granted in {doc_short} transferable or sublicensable?"
    ),
    "Notice Period To Terminate Renewal": (
        "What advance notice period is required to prevent automatic renewal of {doc_short}?"
    ),
    "Parties": "Who are the parties to {doc_short}, and how are they defined?",
    "Post-Termination Services": (
        "What services or obligations must be fulfilled after termination of {doc_short}?"
    ),
    "Price Restrictions": (
        "What price restrictions, pricing floors, or price-setting terms are specified in {doc_short}?"
    ),
    "Renewal Term": (
        "What are the renewal terms of {doc_short}? Does it auto-renew, and on what conditions?"
    ),
    "Revenue/Profit Sharing": (
        "How are revenues or profits allocated or shared between the parties under {doc_short}?"
    ),
    "Rofr/Rofo/Rofn": (
        "Does {doc_short} include a right of first refusal, right of first offer, "
        "or right of first negotiation?"
    ),
    "Source Code Escrow": (
        "Does {doc_short} require source code escrow? "
        "What are the conditions that trigger release?"
    ),
    "Termination For Convenience": (
        "Can either party terminate {doc_short} for convenience without cause? "
        "What notice period is required?"
    ),
    "Third Party Beneficiary": (
        "Does {doc_short} name any third-party beneficiaries who can enforce its terms?"
    ),
    "Uncapped Liability": (
        "Are there any categories of liability that are explicitly uncapped in {doc_short}?"
    ),
    "Unlimited/All-You-Can-Eat-License": (
        "Does {doc_short} grant an unlimited or all-you-can-eat license with no usage restrictions?"
    ),
    "Volume Restriction": (
        "What volume or quantity restrictions apply to the parties under {doc_short}?"
    ),
    "Warranty Duration": (
        "What is the duration of the warranty or guarantee specified in {doc_short}?"
    ),
}

_DEFAULT_TEMPLATE = (
    "Find and explain the {category} clause in {doc_short}."
)


# ---------------------------------------------------------------------------
# Input type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnswerInput:
    """Input to the answer task function, passed by pydantic_evals as ctx.inputs."""

    question: str        # rendered NL question for the RAG agent
    document_name: str   # full CUAD filename (for traceability)
    category: str        # CUAD clause category
    gold_text: str       # expected answer text; read by QualityJudge via ctx.inputs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc_short(doc_name: str, max_len: int = 45) -> str:
    """Return a human-readable abbreviation of a CUAD document filename.

    Example:
        "ACME_CORP_01_01_2020-EX-10-LICENSE AGREEMENT.txt"
        → "ACME CORP 01 01 2020 EX 10 LICENSE AGREEMENT"
    """
    stem = (
        doc_name.replace(".txt", "")
        .replace(".pdf", "")
        .replace("-", " ")
        .replace("_", " ")
    )
    return stem[:max_len].strip() if len(stem) > max_len else stem


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def build_answer_dataset(
    categories: Optional[List[str]] = None,
    limit_per_category: int = 10,
    seed: int = 42,
) -> Dataset:
    """Build a pydantic_evals Dataset for answer quality evaluation.

    Each case contains:
      - inputs:  AnswerInput (question, document, category, gold_text)
      - No expected_output — gold text lives in AnswerInput.gold_text
        so Dataset type params stay clean for the AnswerOutput output type.

    Args:
        categories:          Optional category filter.
        limit_per_category:  Max cases per category. Keep ≤ 10 to control LLM costs.
        seed:                RNG seed for reproducibility.

    Returns:
        pydantic_evals Dataset ready for evaluation with QualityJudge.
    """
    from evals.answer_evaluators import QualityJudge  # noqa: PLC0415

    data = load_annotations()
    sampled = sample_annotations(
        data,
        categories=categories,
        limit_per_category=limit_per_category,
        seed=seed,
    )

    cases: List[Case] = []
    for doc_name, ann in sampled:
        template = NL_TEMPLATES.get(ann.category, _DEFAULT_TEMPLATE)
        question = template.format(
            doc_short=_doc_short(doc_name),
            category=ann.category,
        )
        safe_doc = doc_name[:25].replace(" ", "_")
        safe_cat = ann.category.replace(" ", "_")
        case_name = f"{safe_doc}__{safe_cat}__{ann.start_char}"

        cases.append(
            Case(
                name=case_name,
                inputs=AnswerInput(
                    question=question,
                    document_name=doc_name,
                    category=ann.category,
                    gold_text=ann.text,
                ),
                metadata={"category": ann.category, "doc_name": doc_name},
            )
        )

    return Dataset(
        name="cuad-answer-eval",
        cases=cases,
        evaluators=[QualityJudge()],
    )
