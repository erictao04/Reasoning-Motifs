# Stage 8: Experiment 7 (First-Action BOW)

Bag-of-words analysis focused on the first `action:*` token in each trace.

Models:

- `M0`: oracle per-question baseline
- `M1`: has-first-action only
- `M2`: first-action token BOW + question fixed effects
- `M3`: first-action qualifier-word BOW + question fixed effects

## Run

```bash
python research/stage8_exp7_first_action_bow/run_stage8_exp7.py \
  --input research/gpt_oss_subset_15q_all_traces_tokenized_gpt-oss-chunked_live_1.csv \
  --outdir experiments/exp7_first_action_bow_chunked
```

## Outputs

- `cv_metrics.json`
- `first_action_summary.csv`
- `coefficients.csv`
- `first_action_success_rates.png`
