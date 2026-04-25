# Stage 3: Experiment 2 (Trace Fingerprint Classifier)

Runs grouped cross-validation classifiers for correctness prediction:

- `M0`: oracle per-question baseline
- `M1`: trace length only
- `M2`: L1 logistic on token counts + question fixed effects
- `M3`: `M2` + bigrams

## Run

```bash
python research/stage3_exp2_classifier/run_stage3_exp2.py \
  --input tokenizer/clean_expanded_pool_s100_seed73_qwen25_7b_hot30_tokenized_gpt-oss-120b_0.csv \
  --stage2-token-deltas research/stage2_exp1_single_token_delta/results_gpt-oss-120b_0/token_deltas_all.csv \
  --outdir research/stage3_exp2_classifier/results_gpt-oss-120b_0
```

## Outputs

- `cv_metrics.json`
- `feature_overlap.json`
- `coefficients.csv`
