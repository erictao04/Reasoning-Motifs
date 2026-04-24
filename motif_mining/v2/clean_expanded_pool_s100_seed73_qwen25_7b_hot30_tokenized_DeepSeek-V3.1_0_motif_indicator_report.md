# Motif Indicator Report

- Input: `tokenizer\clean_expanded_pool_s100_seed73_qwen25_7b_hot30_tokenized_DeepSeek-V3.1_0.csv`
- Total traces: 2339
- Correct traces: 1192 (50.96%)
- Incorrect traces: 1147 (49.04%)
- Motif settings: contiguous n-grams, length 1-5, min support 100

## Strongest Success Indicators

| Motif | Support | P(correct given motif) | P(incorrect given motif) | Delta vs baseline | Lift(correct) | Lift(incorrect) |
|---|---:|---:|---:|---:|---:|---:|
| `conclude compute conclude` | 110 | 68.18% | 31.82% | 17.22% | 1.3379 | 0.6488 |
| `simplify conclude` | 277 | 67.87% | 32.13% | 16.91% | 1.3318 | 0.6552 |
| `rewrite compute conclude` | 124 | 66.94% | 33.06% | 15.97% | 1.3134 | 0.6743 |
| `apply-formula rewrite compute` | 174 | 66.67% | 33.33% | 15.70% | 1.3082 | 0.6797 |
| `rewrite derive-intermediate` | 169 | 65.68% | 34.32% | 14.72% | 1.2888 | 0.6999 |
| `conclude compute` | 203 | 65.02% | 34.98% | 14.06% | 1.2759 | 0.7132 |
| `compute simplify conclude` | 164 | 64.63% | 35.37% | 13.67% | 1.2683 | 0.7212 |
| `derive-intermediate instantiate` | 129 | 64.34% | 35.66% | 13.38% | 1.2625 | 0.7272 |
| `rewrite apply-formula` | 276 | 62.68% | 37.32% | 11.72% | 1.2300 | 0.7610 |
| `derive-intermediate apply-formula` | 120 | 62.50% | 37.50% | 11.54% | 1.2264 | 0.7647 |
| `rewrite analyze` | 105 | 61.90% | 38.10% | 10.94% | 1.2147 | 0.7769 |
| `rewrite apply-formula compute` | 112 | 60.71% | 39.29% | 9.75% | 1.1914 | 0.8011 |
| `apply-formula rewrite` | 557 | 59.61% | 40.39% | 8.64% | 1.1696 | 0.8237 |
| `rewrite compute simplify` | 104 | 58.65% | 41.35% | 7.69% | 1.1509 | 0.8431 |
| `instantiate rewrite compute` | 126 | 57.94% | 42.06% | 6.97% | 1.1369 | 0.8578 |
| `apply-formula rewrite simplify` | 144 | 56.94% | 43.06% | 5.98% | 1.1174 | 0.8780 |
| `apply-formula instantiate compute` | 140 | 56.43% | 43.57% | 5.47% | 1.1073 | 0.8885 |
| `analyze instantiate rewrite` | 255 | 56.08% | 43.92% | 5.12% | 1.1004 | 0.8957 |
| `apply-formula compute compute` | 289 | 56.06% | 43.94% | 5.09% | 1.0999 | 0.8961 |
| `compute rewrite` | 418 | 55.98% | 44.02% | 5.02% | 1.0985 | 0.8977 |
| `rewrite instantiate` | 127 | 55.91% | 44.09% | 4.94% | 1.0970 | 0.8992 |
| `conclude` | 1996 | 55.56% | 44.44% | 4.60% | 1.0902 | 0.9062 |
| `rewrite` | 1467 | 55.35% | 44.65% | 4.39% | 1.0861 | 0.9105 |
| `analyze apply-formula rewrite` | 125 | 55.20% | 44.80% | 4.24% | 1.0832 | 0.9136 |
| `compute compute compute compute conclude` | 165 | 55.15% | 44.85% | 4.19% | 1.0822 | 0.9146 |

## Strongest Failure Indicators

| Motif | Support | P(correct given motif) | P(incorrect given motif) | Delta vs baseline | Lift(correct) | Lift(incorrect) |
|---|---:|---:|---:|---:|---:|---:|
| `case-split` | 214 | 28.50% | 71.50% | -22.46% | 0.5593 | 1.4580 |
| `compute analyze` | 105 | 28.57% | 71.43% | -22.39% | 0.5606 | 1.4566 |
| `derive-intermediate compute compute` | 109 | 32.11% | 67.89% | -18.85% | 0.6301 | 1.3844 |
| `compute check-constraint compute` | 104 | 33.65% | 66.35% | -17.31% | 0.6604 | 1.3530 |
| `apply-formula derive-intermediate` | 155 | 34.19% | 65.81% | -16.77% | 0.6710 | 1.3419 |
| `rewrite simplify compute` | 109 | 34.86% | 65.14% | -16.10% | 0.6841 | 1.3283 |
| `instantiate analyze` | 120 | 35.00% | 65.00% | -15.96% | 0.6868 | 1.3255 |
| `derive-intermediate compute` | 301 | 35.55% | 64.45% | -15.41% | 0.6975 | 1.3143 |
| `compute check-constraint` | 324 | 36.42% | 63.58% | -14.54% | 0.7146 | 1.2965 |
| `compare` | 192 | 36.46% | 63.54% | -14.50% | 0.7154 | 1.2958 |
| `case-split compute` | 113 | 37.17% | 62.83% | -13.79% | 0.7293 | 1.2813 |
| `analyze derive-intermediate` | 109 | 38.53% | 61.47% | -12.43% | 0.7561 | 1.2535 |
| `compute compute compute compute compute` | 279 | 38.71% | 61.29% | -12.25% | 0.7596 | 1.2499 |
| `analyze analyze` | 145 | 40.69% | 59.31% | -10.27% | 0.7984 | 1.2095 |
| `analyze instantiate compute` | 358 | 40.78% | 59.22% | -10.18% | 0.8002 | 1.2076 |
| `analyze apply-formula instantiate` | 114 | 42.11% | 57.89% | -8.86% | 0.8262 | 1.1806 |
| `derive-intermediate conclude` | 104 | 42.31% | 57.69% | -8.65% | 0.8302 | 1.1765 |
| `compute compare` | 106 | 42.45% | 57.55% | -8.51% | 0.8330 | 1.1735 |
| `instantiate compute compute compute compute` | 122 | 42.62% | 57.38% | -8.34% | 0.8364 | 1.1701 |
| `derive-intermediate derive-intermediate` | 147 | 42.86% | 57.14% | -8.10% | 0.8410 | 1.1653 |
| `compute compute compute compute` | 423 | 43.03% | 56.97% | -7.94% | 0.8443 | 1.1618 |
| `check-constraint compute` | 204 | 43.14% | 56.86% | -7.82% | 0.8465 | 1.1596 |
| `analyze compute` | 146 | 43.15% | 56.85% | -7.81% | 0.8467 | 1.1593 |
| `rewrite rewrite rewrite` | 134 | 43.28% | 56.72% | -7.68% | 0.8493 | 1.1566 |
| `rewrite compute compute compute` | 119 | 43.70% | 56.30% | -7.26% | 0.8575 | 1.1481 |

## Strongest Single-Token Success Indicators

| Motif | Support | P(correct given motif) | P(incorrect given motif) | Delta vs baseline | Lift(correct) | Lift(incorrect) |
|---|---:|---:|---:|---:|---:|---:|
| `conclude` | 1996 | 55.56% | 44.44% | 4.60% | 1.0902 | 0.9062 |
| `rewrite` | 1467 | 55.35% | 44.65% | 4.39% | 1.0861 | 0.9105 |
| `simplify` | 806 | 52.23% | 47.77% | 1.27% | 1.0249 | 0.9741 |
| `apply-formula` | 1807 | 51.96% | 48.04% | 1.00% | 1.0197 | 0.9796 |
| `compute` | 2035 | 51.15% | 48.85% | 0.19% | 1.0038 | 0.9961 |

## Strongest Single-Token Failure Indicators

| Motif | Support | P(correct given motif) | P(incorrect given motif) | Delta vs baseline | Lift(correct) | Lift(incorrect) |
|---|---:|---:|---:|---:|---:|---:|
| `case-split` | 214 | 28.50% | 71.50% | -22.46% | 0.5593 | 1.4580 |
| `compare` | 192 | 36.46% | 63.54% | -14.50% | 0.7154 | 1.2958 |
| `check-constraint` | 643 | 45.10% | 54.90% | -5.86% | 0.8850 | 1.1195 |
| `analyze` | 1950 | 47.74% | 52.26% | -3.22% | 0.9368 | 1.0656 |
| `derive-intermediate` | 899 | 48.61% | 51.39% | -2.35% | 0.9538 | 1.0480 |
| `instantiate` | 1824 | 49.23% | 50.77% | -1.73% | 0.9661 | 1.0353 |

## Notes

- `Delta vs baseline` is `P(correct given motif) - baseline_correct_rate`.
- Positive delta implies success-associated motif; negative delta implies failure-associated motif.
- Rare motifs can look extreme; use support counts to judge reliability.
