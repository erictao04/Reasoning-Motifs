# Stage 7: Experiment 6 (Prefix-Based Early Prediction)

Runs grouped-CV correctness prediction from partial reasoning prefixes.

Models at each prefix fraction:

- `M0`: oracle per-question baseline
- `M1`: prefix length only
- `M2`: L1 logistic on token counts + question fixed effects
- `M3`: `M2` + bigrams

## Run

```bash
python research/stage7_exp6_prefix_prediction/run_stage7_exp6.py \
  --input tokenizer/clean_expanded_pool_s100_seed73_qwen25_7b_hot30_tokenized_gpt-oss-120b_0.csv \
  --outdir experiments/exp6_prefix_prediction
```

## Outputs

- `prefix_metrics.json`
- `auc_vs_prefix.png`
- `early_token_leaderboard.csv` (optional)
- `time_to_detection.json` (optional)
