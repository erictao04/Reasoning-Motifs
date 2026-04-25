# Stage 6: Experiment 5 (Failure-Mode Clustering)

Implements Stage 6 from `docs/research_plan.md`:

- build per-question token-delta vectors from Stage 2 output
- project with UMAP
- cluster with HDBSCAN
- compute bootstrap-resample cluster stability (ARI)
- export cluster signatures and UMAP figure

## Run

```bash
python research/stage6_exp5_failure_mode_clustering/run_stage6_exp5.py \
  --input research/stage2_exp1_single_token_delta/results_gpt-oss-120b_0/per_question_token_deltas.csv \
  --outdir research/stage6_exp5_failure_mode_clustering/results_gpt-oss-120b_0
```

## Outputs

- `cluster_signatures.csv`
- `question_clusters.csv`
- `umap.png`
- `summary.json`
