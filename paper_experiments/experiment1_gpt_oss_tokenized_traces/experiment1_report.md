# Experiment 1 Report

Input: `gpt-oss-tokenized-traces.csv`

## Dataset

- Total traces: 732
- Total questions: 30
- Mixed-outcome traces: 597
- Mixed-outcome questions: 24

## Held-out Question Prediction

| Slice | Accuracy | Balanced Accuracy | AUC | Q-Local AUC | Length Q-Local AUC | Avg Selected Features |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.5369 | 0.6178 | 0.5749 | 0.6278 | 0.6836 | 276.2 |
| mixed_questions | 0.4387 | 0.5746 | 0.6832 | 0.7137 | 0.6976 | 233.0 |
