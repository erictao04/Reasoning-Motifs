# Experiment 3 Report

## Setup

- GPT-OSS input: `gpt-oss-tokenized-traces.csv`
- DeepSeek input: `clean_expanded_pool_s100_seed73_qwen25_7b_hot30_tokenized_DeepSeek-V3.1_0.csv`
- Shared traces used: 674
- Shared questions used: 30
- Mixed-outcome shared questions: 17
- Unmatched GPT-OSS traces excluded: 58
- Unmatched DeepSeek traces excluded: 1665

## Cross-Tokenization Transfer

| Transfer | AUC | Q-Local AUC | Length Q-Local AUC | Balanced Accuracy | Selected Features |
| --- | ---: | ---: | ---: | ---: | ---: |
| train_gpt_oss_subset_test_deepseek_subset | 0.5000 | 0.5000 | 0.5897 | 0.5000 | 340 |
| train_deepseek_subset_test_gpt_oss_subset | 0.5000 | 0.5000 | 0.5822 | 0.5000 | 211 |

## Shared Motif Stability

- Shared features: 0
- Sign agreement rate: 0.0000
- Shared success motifs: 0
- Shared failure motifs: 0

| Motif | Direction | Left Weight | Right Weight | Weight Sum |
| --- | --- | ---: | ---: | ---: |
