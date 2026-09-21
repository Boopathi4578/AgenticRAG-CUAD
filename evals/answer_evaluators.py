"""Structured single-call LLM judge for answer correctness + faithfulness.

Design
──────
  • ONE OpenAI structured-output call per case  (not two separate calls)
  • Returns a dict{"correctness": bool, "faithfulness": bool} — pydantic_evals
    creates two separate metrics in the report from a dict return value
  • Model: gpt-4o-mini (cheap, reliable for binary boolean rubrics)
  • Logfire span wraps every judge call for full observability

Rubric philosophy
─────────────────
  Correctness  — LENIENT
    Paraphrases, summaries, and partial quotes all count.
    The agent only needs to capture the key substance of the gold clause.

  Faithfulness — STRICT
    Every factual claim in the answer must be verifiable from the retrieved
    context.  Even one hallucinated fact → False.
    If no context was retrieved, faithfulness is always False.

Output type
───────────
  AnswerOutput is defined here (not in answer_cases.py) because it is the
  return type of the task function, and the evaluator needs it for typing.
  run_eval.py imports it from here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import logfire
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from evals.answer_cases import AnswerInput

# ---------------------------------------------------------------------------
# Output type (returned by the answer task function in run_eval.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnswerOutput:
    """Output of the answer task function passed to the evaluator by pydantic_evals."""

    answer_text: str         # the agent's final synthesised answer
    retrieved_context: str   # concatenated text from all retrieve_contracts tool results


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert legal contract QA evaluator.

You receive:
  1. A QUESTION about a specific contract clause
  2. An EXPECTED ANSWER (gold annotation from the CUAD dataset — the ground truth)
  3. A SYSTEM ANSWER (the RAG system's response)
  4. RETRIEVED CONTEXT (the chunks returned by the retrieval tool, if any)

Evaluate two dimensions and respond with a JSON object only — no prose.

──────────────────────────────────────────────────────────────────
CORRECTNESS
──────────────────────────────────────────────────────────────────
Does the system answer accurately capture the key substance of the
expected answer?

  true   The answer contains, paraphrases, or summarises the expected
         answer's core clause information.
  false  The answer is wrong, says the clause is absent when it is
         present, or omits the key information entirely.

Leniency rules:
  • Paraphrases and summaries count as correct.
  • Partial quotes that include the key clause language count.
  • Short gold texts (≤ 20 chars, e.g. a date or party name) require
    a near-exact match.

──────────────────────────────────────────────────────────────────
FAITHFULNESS
──────────────────────────────────────────────────────────────────
Is every factual claim in the system answer grounded in the
retrieved context?

  true   Every fact stated in the answer can be directly verified from
         the retrieved chunks.
  false  The answer introduces at least one fact not present in the
         retrieved context (hallucination), OR no context was retrieved.

Strictness rules:
  • If retrieved context is empty or "(none retrieved)" → false.
  • A single ungrounded claim makes the whole answer unfaithful.
  • Inferences that are logically entailed by the retrieved text are
    acceptable; invented facts are not.
──────────────────────────────────────────────────────────────────
"""

_USER_TEMPLATE = """\
QUESTION:
{question}

EXPECTED ANSWER (CUAD ground truth):
{expected_answer}

SYSTEM ANSWER:
{system_answer}

RETRIEVED CONTEXT:
{retrieved_context}
"""


# ---------------------------------------------------------------------------
# Structured judge verdict
# ---------------------------------------------------------------------------


class JudgeVerdict(BaseModel):
    correctness: bool
    correctness_reason: str
    faithfulness: bool
    faithfulness_reason: str


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


@dataclass
class QualityJudge(Evaluator[AnswerInput, AnswerOutput]):
    """Single structured LLM call evaluating correctness + faithfulness.

    Returns a dict so pydantic_evals creates two separate metric columns:
      {"correctness": bool, "faithfulness": bool}
    """

    model: str = "gpt-4o-mini"

    async def evaluate(
        self,
        ctx: EvaluatorContext[AnswerInput, AnswerOutput],
    ) -> Dict[str, bool]:
        inp = ctx.inputs
        out = ctx.output

        # Guard: task function failed to produce output
        if out is None:
            return {"correctness": False, "faithfulness": False}

        with logfire.span(
            "quality_judge",
            model=self.model,
            category=inp.category,
            doc=inp.document_name[:50],
        ):
            client = AsyncOpenAI()
            user_msg = _USER_TEMPLATE.format(
                question=inp.question,
                expected_answer=inp.gold_text,
                system_answer=out.answer_text,
                retrieved_context=out.retrieved_context or "(none retrieved)",
            )

            response = await client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format=JudgeVerdict,
                temperature=0,
            )
            verdict: JudgeVerdict = response.choices[0].message.parsed

        logfire.info(
            "Judge: correctness={c}, faithfulness={f} | category={cat}",
            c=verdict.correctness,
            f=verdict.faithfulness,
            cat=inp.category,
            correctness_reason=verdict.correctness_reason,
            faithfulness_reason=verdict.faithfulness_reason,
        )

        return {
            "correctness": verdict.correctness,
            "faithfulness": verdict.faithfulness,
        }
