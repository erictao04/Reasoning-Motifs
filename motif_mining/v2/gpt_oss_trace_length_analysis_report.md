# Trace Length vs Quality Analysis

- Input: `tokenizer\clean_expanded_pool_s100_seed73_qwen25_7b_hot30_tokenized_gpt-oss-120b_0.csv`
- Parsed rows: 2683
- Skipped rows: 0
- Unique questions seen: 100
- Questions with analyzable class balance: 77

## Aggregate Conclusion

- Weighted mean length delta (correct - incorrect): 0.7132 tokens
- Unweighted mean length delta: 0.4884 tokens
- Median per-question length delta: 3.4600 tokens
- Questions where correct traces are longer: 62
- Questions where incorrect traces are longer: 15
- Questions tied: 0
- Fraction (non-tied) with correct longer: 80.52%
- Sign test (two-sided) p-value: 0.000000
- Stratified permutation test p-value (within-question label shuffle, weighted mean delta): 0.567087

## Questions With Largest Positive Delta (Correct Longer)

| question_id | delta_mean (tokens) | n_correct | n_incorrect | p-value |
|---|---:|---:|---:|---:|
| 485 | 39.5143 | 7 | 20 | 0.001999 |
| 270 | 15.2500 | 24 | 3 | 0.000500 |
| 172 | 13.1111 | 27 | 2 | 0.004998 |
| 522 | 10.5667 | 18 | 10 | 0.020990 |
| 530 | 10.5256 | 26 | 3 | 0.000500 |
| 169 | 9.8718 | 26 | 3 | 0.002499 |
| 851 | 9.5942 | 3 | 23 | 0.008496 |
| 52 | 9.3974 | 26 | 3 | 0.000500 |
| 1025 | 8.9778 | 9 | 15 | 0.193903 |
| 492 | 8.6533 | 25 | 3 | 0.001000 |

## Questions With Largest Negative Delta (Incorrect Longer)

| question_id | delta_mean (tokens) | n_correct | n_incorrect | p-value |
|---|---:|---:|---:|---:|
| 605 | -236.6923 | 13 | 13 | 0.285857 |
| 1201 | -13.5833 | 2 | 24 | 0.066467 |
| 715 | -9.2917 | 2 | 24 | 0.034983 |
| 1312 | -8.4167 | 2 | 24 | 0.436282 |
| 956 | -6.5000 | 2 | 24 | 0.372314 |
| 150 | -6.0870 | 2 | 23 | 0.563718 |
| 1060 | -4.2609 | 2 | 23 | 0.060970 |
| 923 | -2.6800 | 2 | 25 | 0.353323 |
| 348 | -2.2400 | 25 | 2 | 0.236382 |
| 880 | -2.1250 | 8 | 20 | 0.363818 |

## Notes

- Each question is analyzed independently to control for difficulty differences.
- Primary effect is `mean_length(correct) - mean_length(incorrect)` per question.
- Aggregate permutation p-value tests whether the overall length effect is stronger than expected by chance when correctness labels are shuffled only within each question.
