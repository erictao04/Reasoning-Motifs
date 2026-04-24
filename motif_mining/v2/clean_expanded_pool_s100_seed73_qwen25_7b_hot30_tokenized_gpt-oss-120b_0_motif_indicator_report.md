# Motif Indicator Report

- Input: `tokenizer\clean_expanded_pool_s100_seed73_qwen25_7b_hot30_tokenized_gpt-oss-120b_0.csv`
- Total traces: 2683
- Correct traces: 1233 (45.96%)
- Incorrect traces: 1450 (54.04%)
- Motif settings: contiguous n-grams, length 1-5, min support 100

## Strongest Success Indicators

| Motif | Support | P(correct given motif) | P(incorrect given motif) | Delta vs baseline | Lift(correct) | Lift(incorrect) |
|---|---:|---:|---:|---:|---:|---:|
| `instantiate compute instantiate` | 140 | 67.14% | 32.86% | 21.19% | 1.4610 | 0.6080 |
| `instantiate compute instantiate compute` | 113 | 63.72% | 36.28% | 17.76% | 1.3865 | 0.6714 |
| `analyze instantiate instantiate` | 157 | 63.69% | 36.31% | 17.74% | 1.3860 | 0.6718 |
| `analyze apply-formula compute` | 103 | 62.14% | 37.86% | 16.18% | 1.3521 | 0.7006 |
| `instantiate simplify compute` | 124 | 62.10% | 37.90% | 16.14% | 1.3512 | 0.7013 |
| `apply-formula simplify` | 174 | 62.07% | 37.93% | 16.11% | 1.3506 | 0.7019 |
| `compute apply-formula` | 268 | 61.57% | 38.43% | 15.61% | 1.3397 | 0.7111 |
| `apply-formula check-constraint` | 100 | 61.00% | 39.00% | 15.04% | 1.3274 | 0.7216 |
| `derive_intermediate compute` | 119 | 60.50% | 39.50% | 14.55% | 1.3166 | 0.7308 |
| `simplify conclude` | 134 | 60.45% | 39.55% | 14.49% | 1.3153 | 0.7319 |
| `compute derive_intermediate` | 103 | 60.19% | 39.81% | 14.24% | 1.3098 | 0.7365 |
| `instantiate instantiate` | 309 | 59.87% | 40.13% | 13.91% | 1.3028 | 0.7425 |
| `analyze instantiate compute` | 138 | 59.42% | 40.58% | 13.46% | 1.2930 | 0.7509 |
| `apply-formula instantiate compute` | 131 | 58.78% | 41.22% | 12.82% | 1.2790 | 0.7627 |
| `compute instantiate compute` | 174 | 58.62% | 41.38% | 12.66% | 1.2756 | 0.7657 |
| `analyze apply-formula instantiate` | 156 | 58.33% | 41.67% | 12.38% | 1.2693 | 0.7710 |
| `apply-formula rewrite` | 158 | 58.23% | 41.77% | 12.27% | 1.2670 | 0.7729 |
| `instantiate compute` | 617 | 57.37% | 42.63% | 11.42% | 1.2485 | 0.7887 |
| `derive_intermediate` | 239 | 56.90% | 43.10% | 10.95% | 1.2382 | 0.7974 |
| `instantiate compute compute` | 190 | 56.84% | 43.16% | 10.89% | 1.2369 | 0.7986 |
| `simplify derive-intermediate` | 157 | 56.69% | 43.31% | 10.73% | 1.2335 | 0.8014 |
| `compute instantiate` | 348 | 56.61% | 43.39% | 10.65% | 1.2318 | 0.8029 |
| `derive-intermediate compute conclude` | 112 | 56.25% | 43.75% | 10.29% | 1.2240 | 0.8095 |
| `analyze rewrite` | 224 | 54.02% | 45.98% | 8.06% | 1.1754 | 0.8508 |
| `check-constraint compute` | 163 | 53.99% | 46.01% | 8.03% | 1.1748 | 0.8514 |

## Strongest Failure Indicators

| Motif | Support | P(correct given motif) | P(incorrect given motif) | Delta vs baseline | Lift(correct) | Lift(incorrect) |
|---|---:|---:|---:|---:|---:|---:|
| `instantiate derive-intermediate` | 134 | 17.91% | 82.09% | -28.05% | 0.3897 | 1.5189 |
| `case-split` | 275 | 29.45% | 70.55% | -16.50% | 0.6409 | 1.3053 |
| `apply_formula compute` | 208 | 29.81% | 70.19% | -16.15% | 0.6486 | 1.2988 |
| `check_constraint` | 182 | 30.22% | 69.78% | -15.74% | 0.6576 | 1.2912 |
| `apply-formula derive-intermediate` | 198 | 31.31% | 68.69% | -14.64% | 0.6814 | 1.2709 |
| `analyze instantiate apply-formula` | 155 | 31.61% | 68.39% | -14.34% | 0.6879 | 1.2654 |
| `check-constraint instantiate` | 110 | 31.82% | 68.18% | -14.14% | 0.6924 | 1.2616 |
| `compute check_constraint` | 107 | 32.71% | 67.29% | -13.25% | 0.7118 | 1.2451 |
| `instantiate check-constraint` | 103 | 34.95% | 65.05% | -11.00% | 0.7605 | 1.2036 |
| `analyze analyze analyze` | 109 | 35.78% | 64.22% | -10.18% | 0.7786 | 1.1883 |
| `instantiate apply_formula` | 119 | 36.13% | 63.87% | -9.82% | 0.7863 | 1.1817 |
| `simplify rewrite` | 109 | 37.61% | 62.39% | -8.34% | 0.8185 | 1.1543 |
| `compute compare` | 111 | 38.74% | 61.26% | -7.22% | 0.8430 | 1.1335 |
| `check-constraint derive-intermediate` | 129 | 38.76% | 61.24% | -7.20% | 0.8434 | 1.1332 |
| `apply_formula` | 437 | 39.13% | 60.87% | -6.83% | 0.8515 | 1.1263 |
| `analyze derive-intermediate` | 125 | 39.20% | 60.80% | -6.76% | 0.8530 | 1.1250 |
| `compute compute compute conclude` | 142 | 39.44% | 60.56% | -6.52% | 0.8581 | 1.1206 |
| `compute analyze` | 181 | 39.78% | 60.22% | -6.18% | 0.8656 | 1.1143 |
| `derive-intermediate apply-formula` | 230 | 40.43% | 59.57% | -5.52% | 0.8799 | 1.1022 |
| `analyze analyze` | 329 | 41.03% | 58.97% | -4.92% | 0.8929 | 1.0911 |
| `compute compute conclude` | 341 | 41.94% | 58.06% | -4.02% | 0.9125 | 1.0744 |
| `compute check-constraint` | 251 | 42.63% | 57.37% | -3.33% | 0.9276 | 1.0616 |
| `simplify instantiate` | 133 | 42.86% | 57.14% | -3.10% | 0.9326 | 1.0573 |
| `compute apply-formula compute` | 105 | 42.86% | 57.14% | -3.10% | 0.9326 | 1.0573 |
| `define` | 107 | 42.99% | 57.01% | -2.97% | 0.9355 | 1.0549 |

## Strongest Single-Token Success Indicators

| Motif | Support | P(correct given motif) | P(incorrect given motif) | Delta vs baseline | Lift(correct) | Lift(incorrect) |
|---|---:|---:|---:|---:|---:|---:|
| `derive_intermediate` | 239 | 56.90% | 43.10% | 10.95% | 1.2382 | 0.7974 |
| `simplify` | 1158 | 53.02% | 46.98% | 7.07% | 1.1538 | 0.8692 |
| `substitute` | 196 | 52.55% | 47.45% | 6.60% | 1.1435 | 0.8780 |
| `conclude` | 1721 | 51.66% | 48.34% | 5.70% | 1.1240 | 0.8945 |
| `apply-formula` | 1283 | 50.82% | 49.18% | 4.86% | 1.1058 | 0.9100 |
| `rewrite` | 779 | 50.19% | 49.81% | 4.24% | 1.0922 | 0.9216 |
| `instantiate` | 1723 | 49.22% | 50.78% | 3.26% | 1.0709 | 0.9397 |
| `compute` | 2193 | 49.02% | 50.98% | 3.06% | 1.0667 | 0.9433 |
| `check-constraint` | 722 | 46.95% | 53.05% | 1.00% | 1.0217 | 0.9816 |
| `compare` | 336 | 46.73% | 53.27% | 0.77% | 1.0168 | 0.9857 |

## Strongest Single-Token Failure Indicators

| Motif | Support | P(correct given motif) | P(incorrect given motif) | Delta vs baseline | Lift(correct) | Lift(incorrect) |
|---|---:|---:|---:|---:|---:|---:|
| `case-split` | 275 | 29.45% | 70.55% | -16.50% | 0.6409 | 1.3053 |
| `check_constraint` | 182 | 30.22% | 69.78% | -15.74% | 0.6576 | 1.2912 |
| `apply_formula` | 437 | 39.13% | 60.87% | -6.83% | 0.8515 | 1.1263 |
| `define` | 107 | 42.99% | 57.01% | -2.97% | 0.9355 | 1.0549 |
| `derive-intermediate` | 833 | 43.58% | 56.42% | -2.38% | 0.9482 | 1.0440 |
| `case_split` | 113 | 44.25% | 55.75% | -1.71% | 0.9628 | 1.0316 |
| `analyze` | 2469 | 44.59% | 55.41% | -1.36% | 0.9703 | 1.0252 |
| `factor` | 101 | 45.54% | 54.46% | -0.41% | 0.9910 | 1.0076 |

## Notes

- `Delta vs baseline` is `P(correct given motif) - baseline_correct_rate`.
- Positive delta implies success-associated motif; negative delta implies failure-associated motif.
- Rare motifs can look extreme; use support counts to judge reliability.
