# Signal-Correctness Correlation Report

- Input: `C:\Users\erict\Programming\School\CPSC 440\Reasoning-Motifs\motif_mining\v1\trace_critical_steps.csv`
- Rows: **262**
- Correct rate: **0.714**
- Metrics CSV: `C:\Users\erict\Programming\School\CPSC 440\Reasoning-Motifs\motif_mining\v1\signal_correlation_metrics.csv`

## Ranked Signals (by |Pearson|)

| Signal | mean(correct) | mean(incorrect) | Pearson | Spearman | AUC (higher=>correct) | Cohen d | Best threshold acc | Direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| failure_signal | 0.938931 | 7.057235 | -0.510872 | -0.561648 | 0.159002 | -1.309705 | 0.832061 | le @ 2.294245 |
| log_success_to_failure_ratio | 16.682658 | 5.352928 | 0.469163 | 0.543937 | 0.847380 | 1.170829 | 0.835878 | ge @ 3.383376 |
| success_to_failure_ratio | 69684113675.393494 | 10079891984.449743 | 0.385168 | 0.543937 | 0.847380 | 0.919828 | 0.835878 | ge @ 29.613868 |
| net_signal | 115.206935 | 76.770907 | 0.293587 | 0.278890 | 0.678111 | 0.676855 | 0.763359 | ge @ 28.842394 |
| success_signal | 116.145866 | 83.828141 | 0.255658 | 0.251537 | 0.660642 | 0.582804 | 0.751908 | ge @ 29.092525 |
| trace_length | 17.171123 | 16.480000 | 0.045168 | 0.047118 | 0.530053 | 0.099646 | 0.713740 | ge @ 1.000000 |

## Notes

- Positive Pearson/Spearman means higher signal is associated with correct answers.
- AUC near 1.0 means strong separation where larger signal implies correctness.
- AUC near 0.0 means inverse separation (larger implies incorrectness).
- Best threshold accuracy is single-signal classification quality on this dataset.