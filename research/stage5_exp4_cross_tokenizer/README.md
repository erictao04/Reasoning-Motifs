# Stage 5: Experiment 4 (Cross-Tokenizer Robustness)

Implements the Stage 5 protocol from `docs/research_plan.md`:

- compare Stage 2 token rankings across tokenizer LLMs
- compute pairwise Spearman rho at:
  - basetype level (`basetype:*` collapsed)
  - full-token level after qualifier canonicalization
- compute the stable diagnostic set:
  - tokens in top-K diagnostics for at least `ceil(n/2)` tokenizers

## Run

```bash
python research/stage5_exp4_cross_tokenizer/run_stage5_exp4.py \
  --stage2-root research/stage2_exp1_single_token_delta \
  --outdir experiments/exp4_cross_tokenizer
```

Optionally provide explicit Stage 2 result directories:

```bash
python research/stage5_exp4_cross_tokenizer/run_stage5_exp4.py \
  --result-dirs \
    research/stage2_exp1_single_token_delta/results_gpt-oss-120b_0 \
    research/stage2_exp1_single_token_delta/results_deepseek-v3.1_0 \
    research/stage2_exp1_single_token_delta/results_gemma-4-31b-it_0 \
  --outdir experiments/exp4_cross_tokenizer
```

## Outputs

- `pairwise_spearman.csv`
- `stable_diagnostic_set.csv`

