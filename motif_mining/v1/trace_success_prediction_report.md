# Trace Success Prediction Report

- Input traces: `C:\Users\erict\Programming\School\CPSC 440\Reasoning-Motifs\motif_mining\v1\trace_critical_steps.csv`
- Input metrics: `C:\Users\erict\Programming\School\CPSC 440\Reasoning-Motifs\motif_mining\v1\signal_correlation_metrics.csv`
- Output predictions: `C:\Users\erict\Programming\School\CPSC 440\Reasoning-Motifs\motif_mining\v1\trace_success_predictions.csv`
- Rows scored: **262**
- Accuracy at threshold 0.5: **0.7519**
- Mean predicted p_success: **0.8734**

Model details:
- Uses signal weights from |Pearson correlation with correctness|.
- Uses threshold-centered normalization based on |mean_correct - mean_incorrect|.
- Final `raw_score` is converted with sigmoid to `p_success`.