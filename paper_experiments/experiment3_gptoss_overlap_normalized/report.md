# Experiment 3 Report (Normalized GPT-OSS Labels)

## Setup

- GPT-OSS input: `gpt-oss-tokenized-traces.csv`
- DeepSeek input: `clean_expanded_pool_s100_seed73_qwen25_7b_hot30_tokenized_DeepSeek-V3.1_0.csv`
- Normalization: `NFKC`, dash normalization, strip GPT-OSS subtype suffix after `:`
- Shared traces used: 674
- Shared questions used: 30
- Mixed-outcome shared questions: 17

## Cross-Tokenization Transfer

| Transfer | AUC | Q-Local AUC | Length Q-Local AUC | Balanced Accuracy | Selected Features |
| --- | ---: | ---: | ---: | ---: | ---: |
| train_gpt_oss_normalized_test_deepseek | 0.5932 | 0.5529 | 0.5897 | 0.5704 | 200 |
| train_deepseek_test_gpt_oss_normalized | 0.6554 | 0.5702 | 0.5822 | 0.6453 | 211 |

## Shared Motif Stability

- Shared features: 109
- Sign agreement rate: 0.6055
- Shared success motifs: 31
- Shared failure motifs: 35

| Motif | Direction | Left Weight | Right Weight | Weight Sum |
| --- | --- | ---: | ---: | ---: |
| compute check-constraint | failure | -3.2976 | -1.1959 | 4.4935 |
| analyze instantiate instantiate | failure | -0.4036 | -3.5881 | 3.9917 |
| derive-intermediate apply-formula rewrite | success | 1.3610 | 2.3633 | 3.7242 |
| simplify compute simplify | success | 1.7393 | 1.9697 | 3.7090 |
| instantiate compute apply-formula | failure | -1.9990 | -1.3108 | 3.3098 |
| apply-formula compute rewrite | failure | -1.6284 | -1.5155 | 3.1439 |
| compute apply-formula conclude | failure | -2.5447 | -0.3116 | 2.8563 |
| analyze apply-formula rewrite | success | 2.0752 | 0.6500 | 2.7251 |
| instantiate compute conclude | success | 1.5499 | 1.1682 | 2.7181 |
| apply-formula rewrite rewrite | success | 1.2561 | 1.4473 | 2.7035 |
| apply-formula derive-intermediate compute | failure | -1.5016 | -1.1045 | 2.6061 |
| compute simplify conclude | success | 0.6287 | 1.9486 | 2.5773 |
| apply-formula rewrite conclude | success | 1.0178 | 1.5499 | 2.5678 |
| apply-formula rewrite | success | 1.3935 | 1.1204 | 2.5139 |
| rewrite conclude | success | 1.1073 | 1.3610 | 2.4682 |
