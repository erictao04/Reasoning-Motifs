# Motif Card Report

## Dataset snapshot

- Traces: 450
- Questions: 15
- Success / failure: 297 / 153
- Avg token count: 1.60 (median 1.0)
- Rows with noise: 0 (0.0%)
- Unique tokens: 40
- Per-question bundles emitted: 15

## Why motif cards look useful

- Top global cards cover 291 / 450 traces (64.7%) and 13 / 15 questions.
- Full motif model AUC: 0.4618; question-local AUC: 0.5119.
- Length baseline AUC: 0.7498; question-local AUC: 0.7100.
- In length-matched within-question pairs (gap <= 1), motif scores rank the correct trace above the incorrect one 10.5% of the time vs 58.1% for raw length.

## Compression experiment

| Top-K cards | Motif AUC | Question-local AUC | Avg retained features |
|---|---:|---:|---:|
| 5 | 0.5419 | 0.5240 | 5.0 |
| 10 | 0.5093 | 0.5093 | 10.0 |
| 20 | 0.5059 | 0.5097 | 20.0 |
| 40 | 0.4838 | 0.5383 | 40.0 |

## Top Global Cards

| Motif | Support | Questions | Success when present | Success when absent | Stage | q-value |
|---|---:|---:|---:|---:|---|---:|
| `strategy:choose-representation` | 84 | 7 | 0.714 | 0.648 | early | 0.43 |
| `strategy:set-subgoal` | 80 | 7 | 0.787 | 0.632 | middle | 0.02536 |
| `strategy:common-denominator` | 43 | 2 | 1.000 | 0.624 | middle | 2.735e-07 |
| `strategy:set-subgoal strategy:set-subgoal` | 41 | 4 | 0.805 | 0.645 | middle | 0.1085 |
| `strategy:set-subgoal strategy:set-subgoal strategy:set-subgoal` | 35 | 3 | 0.857 | 0.643 | late | 0.02536 |
| `strategy:choose-representation strategy:set-subgoal` | 28 | 4 | 0.821 | 0.649 | early | 0.1258 |
| `strategy:use-formula` | 28 | 1 | 0.929 | 0.642 | early | 0.006261 |
| `strategy:plan-stepwise` | 27 | 1 | 0.926 | 0.643 | early | 0.006261 |
| `strategy:choose-representation strategy:set-subgoal strategy:set-subgoal` | 25 | 2 | 0.840 | 0.649 | early | 0.1057 |
| `strategy:coefficient-matching` | 25 | 1 | 1.000 | 0.640 | middle | 0.0004063 |
| `strategy:setup` | 24 | 1 | 0.875 | 0.648 | early | 0.05886 |
| `strategy:setup-coordinates` | 22 | 1 | 0.000 | 0.694 | early | 1.224e-09 |
| `strategy:equate-exponents` | 22 | 1 | 1.000 | 0.643 | middle | 0.0008623 |
| `strategy:coefficient-matching strategy:solve-system` | 22 | 1 | 1.000 | 0.643 | late | 0.0008623 |
| `strategy:solve-system` | 22 | 1 | 1.000 | 0.643 | late | 0.0008623 |

## Success-Enriched Cards

| Motif | Support | Questions | Success when present | Success when absent | Stage | q-value |
|---|---:|---:|---:|---:|---|---:|
| `strategy:common-denominator` | 43 | 2 | 1.000 | 0.624 | middle | 2.735e-07 |
| `strategy:set-subgoal strategy:set-subgoal strategy:set-subgoal` | 35 | 3 | 0.857 | 0.643 | late | 0.02536 |
| `strategy:choose-representation strategy:set-subgoal strategy:set-subgoal` | 25 | 2 | 0.840 | 0.649 | early | 0.1057 |
| `strategy:choose-representation strategy:set-subgoal` | 28 | 4 | 0.821 | 0.649 | early | 0.1258 |
| `strategy:set-subgoal strategy:set-subgoal` | 41 | 4 | 0.805 | 0.645 | middle | 0.1085 |
| `strategy:set-subgoal` | 80 | 7 | 0.787 | 0.632 | middle | 0.02536 |
| `strategy:apply-pythagorean` | 8 | 2 | 0.750 | 0.658 | late | 0.7643 |
| `strategy:choose-representation` | 84 | 7 | 0.714 | 0.648 | early | 0.43 |

## Failure-Enriched Cards

| Motif | Support | Questions | Success when present | Success when absent | Stage | q-value |
|---|---:|---:|---:|---:|---|---:|
| `strategy:reduce-to-known-form` | 20 | 2 | 0.600 | 0.663 | middle | 0.6987 |

## Frontend Notes

- Motifs are contiguous subsequences of token length 1 to 3.
- The JSON bundle includes both global cards and per-question local cards.
- Per-question bundles expose local top cards plus the subset of global cards visible in that question.
