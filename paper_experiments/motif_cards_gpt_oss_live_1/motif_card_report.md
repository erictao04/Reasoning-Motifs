# Motif Card Report

## Dataset snapshot

- Traces: 450
- Questions: 15
- Success / failure: 297 / 153
- Avg token count: 11.96 (median 12.0)
- Rows with noise: 97 (21.6%)
- Unique tokens: 248

## Why motif cards look useful

- Top global cards cover 450 / 450 traces (100.0%) and 15 / 15 questions.
- Full motif model AUC: 0.7619; question-local AUC: 0.7970.
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
| `action:compute` | 339 | 15 | 0.776 | 0.306 | middle | 1.968e-10 |
| `action:analyze` | 249 | 14 | 0.618 | 0.711 | early | 0.06197 |
| `action:apply-formula` | 249 | 12 | 0.747 | 0.552 | early | 0.0001433 |
| `action:conclude` | 237 | 13 | 0.819 | 0.484 | late | 1.968e-10 |
| `action:instantiate` | 196 | 12 | 0.653 | 0.665 | early | 0.8795 |
| `action:rewrite` | 171 | 9 | 0.807 | 0.570 | middle | 3.82e-06 |
| `milestone:derived:closed-form` | 120 | 7 | 0.683 | 0.652 | late | 0.6349 |
| `action:simplify` | 112 | 11 | 0.679 | 0.654 | middle | 0.7095 |
| `noise:corrupted-span` | 97 | 15 | 0.165 | 0.796 | late | 1.968e-10 |
| `action:apply-formula action:compute` | 90 | 9 | 0.744 | 0.639 | middle | 0.07772 |
| `action:derive-intermediate` | 85 | 7 | 0.553 | 0.685 | middle | 0.03297 |
| `strategy:choose-representation` | 84 | 7 | 0.714 | 0.648 | early | 0.2967 |
| `strategy:set-subgoal` | 80 | 7 | 0.787 | 0.632 | early | 0.0154 |
| `action:rewrite action:compute` | 77 | 8 | 0.792 | 0.633 | middle | 0.01385 |
| `action:instantiate action:apply-formula` | 74 | 7 | 0.689 | 0.654 | early | 0.6546 |

## Success-Enriched Cards

| Motif | Support | Questions | Success when present | Success when absent | Stage | q-value |
|---|---:|---:|---:|---:|---|---:|
| `action:compute` | 339 | 15 | 0.776 | 0.306 | middle | 1.968e-10 |
| `action:substitute` | 50 | 2 | 1.000 | 0.618 | middle | 2.536e-08 |
| `milestone:derived:solution` | 49 | 2 | 1.000 | 0.618 | late | 2.536e-08 |
| `action:combine` | 47 | 2 | 1.000 | 0.620 | early | 3.487e-08 |
| `action:compute milestone:derived:solution` | 43 | 2 | 1.000 | 0.624 | late | 1.642e-07 |
| `strategy:common-denominator` | 43 | 2 | 1.000 | 0.624 | early | 1.642e-07 |
| `action:substitute action:compute` | 36 | 2 | 1.000 | 0.630 | late | 2.927e-06 |
| `action:rewrite milestone:reduced-to:one-variable` | 27 | 2 | 1.000 | 0.638 | middle | 0.0001236 |
| `strategy:choose-representation action:instantiate` | 24 | 3 | 1.000 | 0.641 | early | 0.0002655 |
| `action:compare` | 23 | 2 | 1.000 | 0.642 | middle | 0.0002655 |
| `strategy:set-subgoal action:rewrite action:compute` | 20 | 2 | 1.000 | 0.644 | late | 0.001043 |
| `milestone:derived:solution action:conclude` | 19 | 2 | 1.000 | 0.645 | late | 0.001043 |
| `action:compute milestone:derived:solution action:conclude` | 19 | 2 | 1.000 | 0.645 | late | 0.001043 |
| `action:simplify action:rewrite` | 16 | 3 | 1.000 | 0.647 | early | 0.004187 |
| `strategy:common-denominator action:combine` | 16 | 2 | 1.000 | 0.647 | early | 0.004187 |

## Failure-Enriched Cards

| Motif | Support | Questions | Success when present | Success when absent | Stage | q-value |
|---|---:|---:|---:|---:|---|---:|
| `action:compute action:instantiate action:compute` | 11 | 3 | 0.000 | 0.677 | early | 6.477e-05 |
| `action:compute action:simplify` | 10 | 4 | 0.000 | 0.675 | late | 0.0001433 |
| `action:derive-intermediate milestone:derived:closed-form` | 10 | 3 | 0.000 | 0.675 | late | 0.0001433 |
| `noise:corrupted-span` | 97 | 15 | 0.165 | 0.796 | late | 1.968e-10 |
| `action:compute action:instantiate` | 13 | 4 | 0.077 | 0.677 | early | 0.0001433 |
| `action:case-split` | 12 | 2 | 0.083 | 0.676 | early | 0.0002472 |
| `action:instantiate action:compute action:instantiate` | 11 | 3 | 0.091 | 0.674 | early | 0.0004901 |
| `action:compute action:derive-intermediate` | 19 | 5 | 0.105 | 0.684 | middle | 7.404e-06 |
| `action:compute action:apply-formula` | 9 | 4 | 0.222 | 0.669 | middle | 0.01512 |
| `action:analyze noise:corrupted-span` | 18 | 9 | 0.278 | 0.676 | middle | 0.003148 |
| `strategy:choose-representation action:apply-formula` | 16 | 2 | 0.312 | 0.673 | early | 0.01002 |
| `action:compute milestone:derived:closed-form action:compute` | 9 | 3 | 0.333 | 0.667 | late | 0.08318 |
| `action:instantiate action:compute` | 50 | 7 | 0.380 | 0.695 | middle | 0.0001765 |
| `noise:corrupted-span action:analyze` | 11 | 7 | 0.364 | 0.667 | early | 0.06923 |
| `action:compute action:check-constraint` | 8 | 4 | 0.375 | 0.665 | middle | 0.1521 |

