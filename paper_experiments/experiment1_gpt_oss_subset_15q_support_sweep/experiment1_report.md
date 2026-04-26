# Experiment 1 Report

Input: `tokenizer/gpt_oss_subset_15q_all_traces_tokenized_gpt-oss-chunked_live_1.csv`

## Dataset

- Total traces: 450
- Total questions: 15
- Mixed-outcome traces: 450
- Mixed-outcome questions: 15

## Held-out Question Prediction Support Sweep

| Support Rule | Slice | Accuracy | Balanced Accuracy | AUC | Q-Local AUC | Length Q-Local AUC | Avg Selected Features |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |

| support >= 3 | full | 0.4267 | 0.5696 | 0.7613 | 0.7692 | 0.7836 | 698.0 |
| support >= 3 | mixed_questions | 0.4267 | 0.5696 | 0.7613 | 0.7692 | 0.7836 | 698.0 |
| support >= 4 | full | 0.4222 | 0.5676 | 0.7800 | 0.7920 | 0.7836 | 537.8 |
| support >= 4 | mixed_questions | 0.4222 | 0.5676 | 0.7800 | 0.7920 | 0.7836 | 537.8 |
| support >= 12 | full | 0.4000 | 0.5533 | 0.7646 | 0.8013 | 0.7836 | 215.4 |
| support >= 12 | mixed_questions | 0.4000 | 0.5533 | 0.7646 | 0.8013 | 0.7836 | 215.4 |