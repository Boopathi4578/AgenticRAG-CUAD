# CUAD RAG Evaluation Suite

Evaluates the `retrieve_contracts` RAG pipeline against CUAD ground-truth annotations,
measuring both retrieval quality and answer quality.

---

## Architecture

```
                        CUAD annotations
                     (data/contracts_annotations.json)
                               │
               ┌───────────────┴───────────────┐
               │                               │
          Gold spans                      Gold answers
      (Oracle benchmark)              (NL question templates)
               │                               │
               ▼                               ▼
        retrieval_cases.py              answer_cases.py
               │                               │
               ▼                               ▼
         Retrieval Eval                   Answer Eval
      search_contracts_direct()      run_agent_with_details()
               │                               │
     ┌─────────┼──────────┐             ┌──────┴──────┐
     │         │          │             │             │
  Recall@K   MRR       nDCG       Correctness  Faithfulness
     │         │          │             │             │
     └─────────┴──────────┘             └──────┬──────┘
                                               │
                                          pydantic_evals
                                               │
                                            Logfire
```

---

## File Map

| File | Purpose |
|------|---------|
| `cuad_loader.py` | Loads CUAD JSON, multi-signal relevance, stratified sampler |
| `retrieval_cases.py` | Builds oracle retrieval `Dataset` (offline DB label step) |
| `answer_cases.py` | Builds NL-question answer `Dataset` (all 41 category templates) |
| `metrics.py` | Recall@K, Precision@K, MRR, nDCG — pure Python, no LLM |
| `answer_evaluators.py` | `QualityJudge` — one structured gpt-4o-mini call per case |
| `run_eval.py` | CLI runner with `--mode retrieval\|answer\|all` |
| `results/` | Timestamped JSON snapshots of every completed run |

---

## Quick Start

```bash
# 1. Install eval dependencies (first time only)
uv sync

# 2. Smoke test — 5 retrieval cases, no LLM calls
make eval-smoke

# 3. Full retrieval benchmark (embedding calls only, cheap)
make eval-retrieval

# 4. Answer quality eval (LLM calls, moderate cost)
make eval-answer

# 5. Full benchmark (retrieval + answer)
make eval-full
```

---

## CLI Reference

```
uv run python -m evals.run_eval [options]

Flags:
  --mode          retrieval | answer | all   (default: retrieval)
  --categories    "Governing Law,Audit Rights"  (all if omitted)
  --limit         Max cases/cat for retrieval   (default: 50)
  --answer-limit  Max cases/cat for answer eval (default: 10)
  --k             Retrieval depth K             (default: 8)
  --smoke         Run only 5 cases/mode for a cheap sanity check
  --logfire / --no-logfire
                  Upload to Logfire Datasets & Experiments
                  (default: ON for answer mode, OFF for retrieval-only)
  --no-save       Skip writing evals/results/*.json snapshots
```

---

## Ground-Truth Relevance Labelling

A DB chunk is considered **relevant** to a CUAD annotation if any of these signals fire:

| Signal | How | Confidence |
|--------|-----|-----------|
| Character-offset overlap | `ann.start < chunk_end AND ann.end > chunk_start` — mirrors `ingestion.py:assign_categories` exactly | 1.0 |
| Substring containment | normalised `ann.text` ⊂ `chunk.text` or vice-versa | 0.95 |
| Token overlap ratio | `\|tokens(ann) ∩ tokens(chunk)\| / \|tokens(ann)\| ≥ 0.5` | 0.5–1.0 |

> **Oracle benchmark warning**: using the gold annotation text verbatim as the query is
> the hardest possible test for the retrieval system (best-case recall). Realistic NL
> queries will achieve equal or lower scores. Use `answer` mode with NL question templates
> for end-to-end realistic evaluation.

---

## Metrics

### Retrieval (deterministic Python)

| Metric | Formula | Target |
|--------|---------|--------|
| Recall@K | `\|relevant ∩ retrieved[:K]\| / \|relevant\|` | > 0.70 |
| Precision@K | `\|relevant ∩ retrieved[:K]\| / K` | > 0.50 |
| MRR | `1 / rank(first relevant)` | > 0.60 |
| nDCG@K | `DCG@K / IDCG@K` (binary relevance) | > 0.65 |

### Answer quality (LLM judge — gpt-4o-mini)

| Metric | Rubric |
|--------|--------|
| Correctness | Lenient: paraphrases + partial quotes count |
| Faithfulness | Strict: every claim must be in the retrieved context |

Target: Correctness > 0.65, Faithfulness > 0.80

---

## Results

JSON snapshots are saved to `evals/results/YYYY-MM-DD_HH-MM_{mode}.json` after each run.
Logfire experiment results appear under **AI Evaluations → Datasets & Experiments** in the
project configured by `logfire.configure()`.

---

## Cost Estimates

| Mode | Cases | API calls | Approx cost |
|------|-------|-----------|-------------|
| `eval-smoke` | 5 | 5 embeddings | < $0.001 |
| `eval-retrieval` | ~2000 | ~2000 embeddings | < $0.10 |
| `eval-answer` | ~410 | ~410 agent + 410 judge | ~$0.50–1.00 |
| `eval-full` | both | all of the above | ~$0.60–1.10 |

Costs are estimates based on `text-embedding-3-small` + `gpt-4o-mini` pricing.
