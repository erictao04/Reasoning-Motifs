# Trace Length vs Quality Analysis

- Input: `tokenizer\expanded_pool_s100_seed73_qwen25_7b_hot30_tokenized_DeepSeek-V3.1_0.csv`
- Parsed rows: 2339
- Skipped rows: 541
- Unique questions seen: 96
- Questions with analyzable class balance: 57

## Aggregate Conclusion

- Weighted mean length delta (correct - incorrect): -79.6109 tokens
- Unweighted mean length delta: -82.2440 tokens
- Median per-question length delta: -0.0500 tokens
- Questions where correct traces are longer: 28
- Questions where incorrect traces are longer: 29
- Questions tied: 0
- Fraction (non-tied) with correct longer: 49.12%
- Sign test (two-sided) p-value: 1.000000
- Stratified permutation test p-value (within-question label shuffle, weighted mean delta): 0.176165

## Questions With Largest Positive Delta (Correct Longer)

| question_id | delta_mean (tokens) | n_correct | n_incorrect | p-value |
|---|---:|---:|---:|---:|
| 956 | 2887.7609 | 2 | 23 | 0.046977 |
| 1164 | 588.0673 | 13 | 8 | 0.241879 |
| 485 | 496.0706 | 5 | 17 | 0.093453 |
| 1121 | 384.2222 | 5 | 18 | 0.868566 |
| 887 | 293.3782 | 12 | 13 | 0.502249 |
| 542 | 273.4000 | 12 | 10 | 0.908046 |
| 522 | 244.2721 | 17 | 8 | 0.441279 |
| 415 | 190.0000 | 18 | 2 | 0.247876 |
| 1025 | 179.5500 | 10 | 12 | 0.004998 |
| 252 | 59.5038 | 19 | 7 | 0.659170 |

## Questions With Largest Negative Delta (Incorrect Longer)

| question_id | delta_mean (tokens) | n_correct | n_incorrect | p-value |
|---|---:|---:|---:|---:|
| 844 | -1461.7667 | 4 | 15 | 0.233383 |
| 165 | -1415.1136 | 22 | 4 | 0.133933 |
| 840 | -1333.9470 | 11 | 12 | 0.051974 |
| 1155 | -645.7083 | 8 | 15 | 0.480760 |
| 1142 | -610.1000 | 5 | 20 | 0.286357 |
| 1000 | -568.7632 | 4 | 19 | 0.798601 |
| 864 | -539.6182 | 11 | 15 | 0.470765 |
| 535 | -497.1571 | 14 | 10 | 0.389805 |
| 1139 | -496.7619 | 3 | 21 | 0.969515 |
| 224 | -478.1879 | 11 | 15 | 0.400300 |

## Notes

- Each question is analyzed independently to control for difficulty differences.
- Primary effect is `mean_length(correct) - mean_length(incorrect)` per question.
- Aggregate permutation p-value tests whether the overall length effect is stronger than expected by chance when correctness labels are shuffled only within each question.
