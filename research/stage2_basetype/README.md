# Stage 2 Basetype Diagnostics

This is a basetype-only variant of Stage 2.

Before analysis, each token is collapsed as:

- `basetype:qualifier` -> `basetype`
- `basetype` -> `basetype` (unchanged)

## Run

```bash
python research/stage2_basetype/run_stage2_basetype.py \
  --input tokenizer_v5/results/gpt-oss-tokenized-traces.csv \
  --outdir research/stage2_basetype/results_gpt-oss-tokenizer-v5
```

## Outputs

- `token_deltas_all.csv`
- `leaderboard_lifesavers.csv`
- `leaderboard_killers.csv`
- `per_question_token_deltas.csv`
- `question_baselines.csv`
- `count_bucket_effects.csv`
- `permutation_null.json`
- `forest_plot.png` (if matplotlib is available)
- `summary.json`
