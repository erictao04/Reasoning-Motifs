# Tokenizer v5

Audit + granular tokenization pipeline for the within-question contrast
research direction. See [DESIGN.md](DESIGN.md) for the full design and
[../docs/research_plan.md](../docs/research_plan.md) for the paper plan.

## Quick start

Set `TOGETHER_API_KEY` in any of `tokenizer_v5/.env`, `tokenizer/.env`, or
the repo-root `.env`.

End-to-end (audit -> tokenize):

```bash
python -m tokenizer_v5.pipeline \
  --input tokenizer/expanded_pool_s100_seed73_qwen25_7b_hot30.csv \
  --judge-model openai/gpt-oss-120b \
  --tokenizer-model openai/gpt-oss-120b
```

Audit only:

```bash
python -m tokenizer_v5.audit \
  --input tokenizer/expanded_pool_s100_seed73_qwen25_7b_hot30.csv \
  --output tokenizer_v5/results/scratch/audited.csv \
  --summary tokenizer_v5/results/scratch/audit_summary.json
```

Tokenize only (assumes input already audited):

```bash
python -m tokenizer_v5.tokenize \
  --input tokenizer_v5/results/scratch/audited.csv \
  --output tokenizer_v5/results/scratch/tokenized.csv \
  --summary tokenizer_v5/results/scratch/tokenize_summary.json
```

## Outputs

Each pipeline invocation writes to `tokenizer_v5/results/<run_id>/`:

- `audited_traces.csv` — original CSV plus audit columns and a revised
  `is_correct`.
- `audit_summary.json`
- `tokenized_traces.csv` — `question_id, sample_id, gold_answer,
  predicted_answer, is_correct, tokenized_trace`.
- `tokenize_summary.json` — includes `n_mixed_outcome_questions` (the
  `|M|` Exp. 1 needs).
- `manifest.json` — config, hashes, timings.

## Caching

All LLM responses are cached on disk by content hash:

- `tokenizer_v5/cache/audit/<judge_model>/<sha>.json`
- `tokenizer_v5/cache/tokenize/<tokenizer_model>/{metadata,per_trace}/<sha>.json`

Re-runs with the same input + same prompts are free (no API calls). Bump
the prompt to invalidate.

## Stdlib only

No `pip install` required. Uses `urllib` for HTTP, `csv` for IO, and
`concurrent.futures.ThreadPoolExecutor` for parallel API calls.

## Smoke test (no API key needed)

Cached responses live in `cache/` so most logic can be exercised offline
once one full run has been done. To check prompts and decision rules
without spending tokens, re-run the pipeline pointed at a tiny CSV
(2-3 questions) — first run hits the API once per row, subsequent runs
read from disk.
