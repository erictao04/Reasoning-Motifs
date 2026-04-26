# Experiment 1 Report

Input: `tokenizer/gpt_oss_subset_15q_all_traces_tokenized_gpt-oss-chunked_live_1.csv`
Baseline source: `raw_words`

## Dataset

- Total traces: 450
- Total questions: 15
- Mixed-outcome traces: 450
- Mixed-outcome questions: 15

## Held-out Question Prediction Support Sweep

| Support Rule | Slice | Accuracy | Balanced Accuracy | AUC | Q-Local AUC | Length Q-Local AUC | Avg Selected Features |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |

| support >= 3 | full | 0.7444 | 0.7537 | 0.7708 | 0.8149 | 0.8329 | 698.0 |
| support >= 3 | mixed_questions | 0.7444 | 0.7537 | 0.7708 | 0.8149 | 0.8329 | 698.0 |
| support >= 4 | full | 0.7489 | 0.7585 | 0.7618 | 0.8094 | 0.8329 | 537.8 |
| support >= 4 | mixed_questions | 0.7489 | 0.7585 | 0.7618 | 0.8094 | 0.8329 | 537.8 |
| support >= 12 | full | 0.7244 | 0.7387 | 0.7661 | 0.8130 | 0.8329 | 215.4 |
| support >= 12 | mixed_questions | 0.7244 | 0.7387 | 0.7661 | 0.8130 | 0.8329 | 215.4 |