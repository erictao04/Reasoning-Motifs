# Stage 2: Experiment 1 (Single-Token Delta)

Implements the Stage 2 protocol from `docs/research_plan.md`:

- restrict to mixed-outcome questions
- compute within-question token delta
- sign test + bootstrap CI + BH q-values
- count-bucket analysis (`0`, `1`, `>=2`)
- permutation null sanity check

## Run

```bash
python research/stage2_exp1_single_token_delta/run_stage2_exp1.py \
  --input tokenizer/clean_expanded_pool_s100_seed73_qwen25_7b_hot30_tokenized_gpt-oss-120b_0.csv \
  --outdir research/stage2_exp1_single_token_delta/results_gpt-oss-120b_0
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
