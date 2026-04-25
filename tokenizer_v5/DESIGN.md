# Tokenizer v5 — Pipeline Design

## Goal

Stage 0 (label audit) and Stage 1 (granular tokenization) of the research
plan in [`../docs/research_plan.md`](../docs/research_plan.md). The output is a
clean, audited, qualified-token CSV that feeds directly into the
within-question contrast experiments (Exp. 1–5).

## Inputs and outputs

**Input**: any CSV under `tokenizer/expanded_pool*.csv` with at least these
columns:
`question_id`, `sample_id`, `question`, `gold_answer`, `predicted_answer`,
`is_correct`, `reasoning_trace`. Optional but used when present:
`final_response_text`, `finish_reason`.

**Output (per run, under `tokenizer_v5/results/<run_id>/`):**
- `audited_traces.csv` — input columns + audit columns + new `is_correct`
- `audit_summary.json` — drop / relabel / flag counts; verdict distribution
- `tokenized_traces.csv` — `question_id, sample_id, gold_answer,
  predicted_answer, is_correct, tokenized_trace`
- `tokenize_summary.json` — per-question stats; `n_mixed_outcome_questions`
- `manifest.json` — config, input SHA-256, git SHA, model names, timings

## Pipeline

```
raw expanded_pool*.csv
    |
    v
+-------------+
|  audit.py   |  one judge LLM call per trace (cached)
|             |  decision rules -> re-label / drop / flag
+------+------+
       v
audited_traces.csv
       |
       v
+-------------+
| tokenize.py |  per-question metadata call (cached)
|             |  per-trace tokenize call (cached)
|             |  validate basetype:qualifier
+------+------+
       v
tokenized_traces.csv  ->  feeds Exp. 1-5
```

Both stages are **idempotent** and **resumable**: every LLM response is
cached on disk by content hash, so re-runs only call the API for new rows
or new prompts.

## Module layout

```
tokenizer_v5/
├── DESIGN.md           # this file
├── README.md           # quick start
├── __init__.py
├── _common.py          # run_id, sha256, csv/json IO, git SHA, paths
├── llm_client.py       # Together-compatible chat client + disk cache
├── audit.py            # Stage 0
├── tokenize.py         # Stage 1
└── pipeline.py         # one-shot: audit -> tokenize -> manifest
```

Cache and result directories are created at runtime:

```
tokenizer_v5/
├── cache/
│   ├── audit/<judge_model>/<sha256>.json
│   └── tokenize/<tokenizer_model>/{metadata,per_trace}/<sha256>.json
└── results/
    └── <run_id>/...
```

## Stage 0 — Audit (`audit.py`)

### Prompt schema

System: pinned to a strict-but-fair grader role; emphasizes mathematical
equivalence (fractions vs decimals, latex vs plain, equivalent algebraic
forms, set/tuple ordering).

User: a structured JSON payload per trace:

```json
{
  "question_id": "...",
  "question": "...",
  "gold_answer": "...",
  "predicted_answer": "...",
  "trace_excerpt": "<final_response_text or last 500 chars of reasoning_trace>"
}
```

Required JSON output (validated by the script):

```json
{
  "verdict": "correct|incorrect|ambiguous|non_attempt",
  "confidence": 0.0,
  "reason": "one sentence",
  "trace_concludes_predicted": true
}
```

### Decision rules (pre-registered)

Applied uniformly after all rows are scored:

1. `verdict == "non_attempt"` → `keep = False`,
   `drop_reason = "non_attempt"`.
2. `trace_concludes_predicted == False` AND `confidence ≥ τ` (default 0.8)
   → `keep = False`, `drop_reason = "extraction_failure"` (the predicted
   answer was never the trace's actual conclusion).
3. `verdict ∈ {correct, incorrect}` AND `confidence ≥ τ` AND
   `verdict_bool != original is_correct` → re-label, set
   `is_relabeled = True`.
4. Otherwise: keep original label, set `is_low_confidence_label = True`.

The audit CSV preserves the original `is_correct` as `is_correct_original`
and writes the working label into `is_correct`. Downstream stages may
filter on `keep == True` (default) or include flagged rows for sensitivity
analysis (`--include-flagged`).

### Concurrency / cost

LLM calls run in a `ThreadPoolExecutor` (default 4 workers; configurable).
At ~1 call per trace, a 1 200-trace corpus is ~5 minutes wall time at 4
concurrent workers and a fast judge.

## Stage 1 — Tokenize (`tokenize.py`)

### Two-phase per question

1. **Metadata** (one call per `question_id`): builds a per-question
   tokenization guide that constrains qualifier choices for that
   question's traces. Cached by question id + tokenizer model + prompt
   hash.
2. **Per-trace tokenize** (one call per trace): emits a
   `basetype:qualifier` token sequence using the metadata. Cached by
   trace.

### Token format (enforced)

Every emitted token is `basetype:qualifier`:

- `basetype` ∈ fixed 13-token vocabulary (analyze, instantiate, compute,
  apply-formula, rewrite, check-constraint, case-split, backtrack,
  conclude, guess, simplify, compare, derive-intermediate).
- `qualifier` is a short hyphenated noun phrase (1–3 words, lowercase)
  describing the mathematical content of that span.

### Validation

After per-trace tokenize, each token must:
- be non-empty,
- contain exactly one `:`,
- have its prefix in the `BASETYPE_VOCAB` constant,
- have a non-empty qualifier.

Traces with `tokenized_trace == "MISSING"` or any invalid token are
dropped, and the `(question_id, sample_id, drop_reason, invalid_tokens)`
tuple is recorded in `tokenize_summary.json["dropped_traces"]` (capped to
200 examples in the summary; full list reproducible from cache).

## Pipeline runner — `pipeline.py`

```bash
python -m tokenizer_v5.pipeline \
  --input tokenizer/expanded_pool_s100_seed73_qwen25_7b_hot30.csv \
  --judge-model openai/gpt-oss-120b \
  --tokenizer-model openai/gpt-oss-120b \
  --max-workers 4
```

Key flags:
- `--skip-audit`: tokenize directly from the input.
- `--skip-tokenize`: audit only.
- `--confidence-threshold` (default 0.8): re-label / drop threshold.
- `--run-id`: pin the run id (otherwise auto-generated as
  `YYYYMMDD-HHMMSS-<hex>`).

Each invocation writes `results/<run_id>/manifest.json` recording inputs,
config, git SHA, stage timings, and the two stage summaries.

## Reproducibility

- Seed `73` (matches the corpus suffix).
- Cache hit / miss counts reported in every summary JSON.
- Manifest stores input file SHA-256 so re-analysis is auditable.
- Outputs are stable across re-runs as long as model, prompts, and
  decision rules are unchanged.

## Failure modes & invariants

- Judge LLM returns invalid JSON → fall back to
  `verdict="ambiguous", confidence=0.0` (row will be flagged
  low-confidence, original label preserved).
- Tokenizer returns invalid tokens or `"MISSING"` → drop the trace; never
  silently accept malformed output.
- Question with zero usable traces post-tokenize → omitted from the
  output; logged in summary.
- All API errors are caught per-row; the pipeline never partially
  overwrites an output CSV.

## Non-goals

- Not implementing equivalence in Python (sympy / regex). The judge LLM
  owns that.
- Not modifying the legacy `tokenizer/` directory; v5 is a parallel stack.
- No streaming; no token-level alignment outputs.

## Concrete next steps

1. Run pipeline on a small (≤ 50 trace) slice of `expanded_pool*.csv` to
   sanity-check prompts and surface any malformed-token rates.
2. Manually spot-check 30 rows of audit disagreements (judge vs original)
   — proceed only if judge wins ≥ 80%.
3. Run on the full corpus.
4. Stand up `experiments/exp1_single_token_diagnostics/` against the
   resulting `tokenized_traces.csv`.
