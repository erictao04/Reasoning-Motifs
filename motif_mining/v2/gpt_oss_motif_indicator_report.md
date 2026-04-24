# Motif Indicator Report

- Input: `tokenizer\clean_expanded_pool_s100_seed73_qwen25_7b_hot30_tokenized_gpt-oss-120b_0.csv`
- Total traces: 2683
- Correct traces: 1233 (45.96%)
- Incorrect traces: 1450 (54.04%)
- Motif settings: contiguous n-grams, length 1-3, min support 3

## Strongest Success Indicators

| Motif | Support | P(correct given motif) | P(incorrect given motif) | Delta vs baseline | Lift(correct) | Lift(incorrect) |
|---|---:|---:|---:|---:|---:|---:|
| `solve` | 92 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `solve conclude` | 33 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `solve compute` | 32 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `simplify derive-intermediate compare` | 30 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `compute solve` | 28 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `simplify simplify isolate` | 27 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `compute_sum` | 26 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `identify_parameters compute_sum` | 26 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `find_common_denominator add_fractions` | 26 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `identify_parameters` | 26 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `find_common_denominator` | 26 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `add_fractions` | 26 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `transform` | 25 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `instantiate transform` | 25 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `find_common_denominator add_fractions identify_parameters` | 24 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |

## Strongest Failure Indicators

| Motif | Support | P(correct given motif) | P(incorrect given motif) | Delta vs baseline | Lift(correct) | Lift(incorrect) |
|---|---:|---:|---:|---:|---:|---:|
| `instantiate derive-intermediate instantiate` | 28 | 0.00% | 100.00% | -45.96% | 0.0000 | 1.8503 |
| `count_favorable` | 28 | 0.00% | 100.00% | -45.96% | 0.0000 | 1.8503 |
| `identify-symmetry` | 27 | 0.00% | 100.00% | -45.96% | 0.0000 | 1.8503 |
| `count_total` | 26 | 0.00% | 100.00% | -45.96% | 0.0000 | 1.8503 |
| `analyze count_total` | 26 | 0.00% | 100.00% | -45.96% | 0.0000 | 1.8503 |
| `analyze apply-identity` | 25 | 0.00% | 100.00% | -45.96% | 0.0000 | 1.8503 |
| `apply-identity` | 25 | 0.00% | 100.00% | -45.96% | 0.0000 | 1.8503 |
| `rearrange compute` | 25 | 0.00% | 100.00% | -45.96% | 0.0000 | 1.8503 |
| `compute check-constraint rearrange` | 24 | 0.00% | 100.00% | -45.96% | 0.0000 | 1.8503 |
| `case-split rearrange` | 24 | 0.00% | 100.00% | -45.96% | 0.0000 | 1.8503 |
| `count_favorable apply_formula` | 23 | 0.00% | 100.00% | -45.96% | 0.0000 | 1.8503 |
| `rearrange compute check-constraint` | 23 | 0.00% | 100.00% | -45.96% | 0.0000 | 1.8503 |
| `case-split rearrange compute` | 23 | 0.00% | 100.00% | -45.96% | 0.0000 | 1.8503 |
| `check-constraint rearrange compute` | 23 | 0.00% | 100.00% | -45.96% | 0.0000 | 1.8503 |
| `apply-identity case-split` | 20 | 0.00% | 100.00% | -45.96% | 0.0000 | 1.8503 |

## Strongest Single-Token Success Indicators

| Motif | Support | P(correct given motif) | P(incorrect given motif) | Delta vs baseline | Lift(correct) | Lift(incorrect) |
|---|---:|---:|---:|---:|---:|---:|
| `solve` | 92 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `compute_sum` | 26 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `identify_parameters` | 26 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `find_common_denominator` | 26 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `add_fractions` | 26 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `transform` | 25 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `multiply` | 24 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `sqrt` | 24 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `verify` | 8 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `differentiate` | 3 | 100.00% | 0.00% | 54.04% | 2.1760 | 0.0000 |
| `rationalize` | 27 | 96.30% | 3.70% | 50.34% | 2.0954 | 0.0685 |
| `infer` | 26 | 92.31% | 7.69% | 46.35% | 2.0086 | 0.1423 |
| `apply_theorem` | 16 | 87.50% | 12.50% | 41.54% | 1.9040 | 0.2313 |
| `divide` | 45 | 86.67% | 13.33% | 40.71% | 1.8859 | 0.2467 |
| `combine` | 21 | 85.71% | 14.29% | 39.76% | 1.8651 | 0.2643 |

## Strongest Single-Token Failure Indicators

| Motif | Support | P(correct given motif) | P(incorrect given motif) | Delta vs baseline | Lift(correct) | Lift(incorrect) |
|---|---:|---:|---:|---:|---:|---:|
| `count_favorable` | 28 | 0.00% | 100.00% | -45.96% | 0.0000 | 1.8503 |
| `identify-symmetry` | 27 | 0.00% | 100.00% | -45.96% | 0.0000 | 1.8503 |
| `count_total` | 26 | 0.00% | 100.00% | -45.96% | 0.0000 | 1.8503 |
| `apply-identity` | 25 | 0.00% | 100.00% | -45.96% | 0.0000 | 1.8503 |
| `normalize` | 20 | 0.00% | 100.00% | -45.96% | 0.0000 | 1.8503 |
| `identify-similarity` | 26 | 3.85% | 96.15% | -42.11% | 0.0837 | 1.7792 |
| `determine-ratio` | 25 | 4.00% | 96.00% | -41.96% | 0.0870 | 1.7763 |
| `apply-area-ratio` | 24 | 4.17% | 95.83% | -41.79% | 0.0907 | 1.7732 |
| `count` | 38 | 5.26% | 94.74% | -40.69% | 0.1145 | 1.7530 |
| `apply_algorithm` | 25 | 8.00% | 92.00% | -37.96% | 0.1741 | 1.7023 |
| `solve-equation` | 20 | 10.00% | 90.00% | -35.96% | 0.2176 | 1.6653 |
| `approximate` | 20 | 15.00% | 85.00% | -30.96% | 0.3264 | 1.5728 |
| `derive_constraint` | 24 | 16.67% | 83.33% | -29.29% | 0.3627 | 1.5420 |
| `backtrack` | 86 | 20.93% | 79.07% | -25.03% | 0.4554 | 1.4631 |
| `sum` | 21 | 23.81% | 76.19% | -22.15% | 0.5181 | 1.4098 |

## Notes

- `Delta vs baseline` is `P(correct given motif) - baseline_correct_rate`.
- Positive delta implies success-associated motif; negative delta implies failure-associated motif.
- Rare motifs can look extreme; use support counts to judge reliability.
