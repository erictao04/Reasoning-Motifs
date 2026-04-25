# Motif Card Report

## Dataset snapshot

- Traces: 450
- Questions: 15
- Success / failure: 297 / 153
- Avg token count: 11.96 (median 12.0)
- Rows with noise: 97 (21.6%)
- Unique tokens: 248
- Per-question bundles emitted: 15

## Why motif cards look useful

- Top global cards cover 283 / 450 traces (62.9%) and 14 / 15 questions.
- Full motif model AUC: 0.4802; question-local AUC: 0.5158.
- Length baseline AUC: 0.7020; question-local AUC: 0.7836.
- In length-matched within-question pairs (gap <= 1), motif scores rank the correct trace above the incorrect one 44.7% of the time vs 52.0% for raw length.

## Compression experiment

| Top-K cards | Motif AUC | Question-local AUC | Avg retained features |
|---|---:|---:|---:|
| 5 | 0.4707 | 0.5138 | 5.0 |
| 10 | 0.4476 | 0.4907 | 10.0 |
| 20 | 0.5336 | 0.5825 | 20.0 |
| 40 | 0.6667 | 0.7156 | 40.0 |

## Top Global Cards

| Motif | Support | Questions | Success when present | Success when absent | Stage | q-value |
|---|---:|---:|---:|---:|---|---:|
| `action:apply-formula action:compute` | 90 | 9 | 0.744 | 0.639 | middle | 0.08015 |
| `action:rewrite action:compute` | 77 | 8 | 0.792 | 0.633 | middle | 0.01525 |
| `action:instantiate action:apply-formula` | 74 | 7 | 0.689 | 0.654 | early | 0.66 |
| `action:analyze action:apply-formula` | 52 | 8 | 0.769 | 0.646 | early | 0.1093 |
| `action:analyze action:instantiate` | 51 | 10 | 0.686 | 0.657 | early | 0.8095 |
| `action:instantiate action:compute` | 50 | 7 | 0.380 | 0.695 | middle | 0.0002632 |
| `action:compute milestone:derived:closed-form` | 48 | 6 | 0.542 | 0.674 | late | 0.09624 |
| `action:apply-formula action:rewrite` | 47 | 6 | 0.681 | 0.658 | early | 0.9185 |
| `milestone:derived:closed-form action:compute` | 47 | 5 | 0.660 | 0.660 | late | 1 |
| `action:substitute action:compute` | 36 | 2 | 1.000 | 0.630 | late | 6.273e-06 |
| `action:instantiate action:apply-formula action:compute` | 35 | 5 | 0.829 | 0.646 | early | 0.05732 |
| `action:compute action:conclude` | 33 | 9 | 0.667 | 0.659 | late | 1 |
| `strategy:choose-representation action:analyze` | 31 | 4 | 0.710 | 0.656 | early | 0.7692 |
| `strategy:set-subgoal action:apply-formula` | 30 | 4 | 0.967 | 0.638 | early | 0.0004362 |
| `strategy:set-subgoal action:rewrite` | 27 | 3 | 0.963 | 0.641 | middle | 0.0009941 |

## Success-Enriched Cards

| Motif | Support | Questions | Success when present | Success when absent | Stage | q-value |
|---|---:|---:|---:|---:|---|---:|
| `action:substitute action:compute` | 36 | 2 | 1.000 | 0.630 | late | 6.273e-06 |
| `action:rewrite milestone:reduced-to:one-variable` | 27 | 2 | 1.000 | 0.638 | middle | 0.0001843 |
| `strategy:choose-representation action:instantiate` | 24 | 3 | 1.000 | 0.641 | early | 0.0003756 |
| `strategy:set-subgoal action:rewrite action:compute` | 20 | 2 | 1.000 | 0.644 | late | 0.001354 |
| `action:simplify action:rewrite` | 16 | 3 | 1.000 | 0.647 | early | 0.004961 |
| `strategy:common-denominator action:combine` | 16 | 2 | 1.000 | 0.647 | early | 0.004961 |
| `action:analyze action:apply-formula action:rewrite` | 14 | 2 | 1.000 | 0.649 | early | 0.007685 |
| `action:compute milestone:derived:closed-form action:conclude` | 10 | 3 | 1.000 | 0.652 | late | 0.02963 |
| `action:conclude milestone:derived:closed-form` | 10 | 2 | 1.000 | 0.652 | late | 0.02963 |
| `action:analyze strategy:choose-representation action:instantiate` | 8 | 3 | 1.000 | 0.654 | early | 0.07177 |
| `strategy:set-subgoal action:apply-formula` | 30 | 4 | 0.967 | 0.638 | early | 0.0004362 |
| `strategy:set-subgoal action:rewrite` | 27 | 3 | 0.963 | 0.641 | middle | 0.0009941 |
| `strategy:choose-representation action:analyze action:instantiate` | 18 | 2 | 0.944 | 0.648 | early | 0.01725 |
| `strategy:set-subgoal action:analyze` | 23 | 5 | 0.913 | 0.646 | early | 0.01251 |
| `milestone:derived:closed-form action:conclude` | 12 | 3 | 0.917 | 0.653 | late | 0.08455 |

## Failure-Enriched Cards

| Motif | Support | Questions | Success when present | Success when absent | Stage | q-value |
|---|---:|---:|---:|---:|---|---:|
| `action:compute action:instantiate action:compute` | 11 | 3 | 0.000 | 0.677 | early | 9.959e-05 |
| `action:compute action:simplify` | 10 | 4 | 0.000 | 0.675 | late | 0.0002132 |
| `action:derive-intermediate milestone:derived:closed-form` | 10 | 3 | 0.000 | 0.675 | late | 0.0002132 |
| `action:compute action:instantiate` | 13 | 4 | 0.077 | 0.677 | early | 0.0002132 |
| `action:instantiate action:compute action:instantiate` | 11 | 3 | 0.091 | 0.674 | early | 0.0006666 |
| `action:compute action:derive-intermediate` | 19 | 5 | 0.105 | 0.684 | middle | 1.595e-05 |
| `action:compute action:apply-formula` | 9 | 4 | 0.222 | 0.669 | middle | 0.01669 |
| `action:analyze noise:corrupted-span` | 18 | 9 | 0.278 | 0.676 | middle | 0.003764 |
| `strategy:choose-representation action:apply-formula` | 16 | 2 | 0.312 | 0.673 | early | 0.01126 |
| `action:compute milestone:derived:closed-form action:compute` | 9 | 3 | 0.333 | 0.667 | late | 0.08546 |
| `action:instantiate action:compute` | 50 | 7 | 0.380 | 0.695 | middle | 0.0002632 |
| `noise:corrupted-span action:analyze` | 11 | 7 | 0.364 | 0.667 | early | 0.07177 |
| `action:compute action:check-constraint` | 8 | 4 | 0.375 | 0.665 | middle | 0.1545 |
| `milestone:derived:closed-form strategy:set-subgoal` | 8 | 3 | 0.375 | 0.665 | late | 0.1545 |
| `action:analyze action:instantiate action:compute` | 12 | 5 | 0.417 | 0.667 | early | 0.1433 |

## Frontend Notes

- Motifs are contiguous subsequences of token length 1 to 3.
- The JSON bundle includes both global cards and per-question local cards.
- Per-question bundles expose local top cards plus the subset of global cards visible in that question.
