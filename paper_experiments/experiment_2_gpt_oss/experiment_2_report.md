# Experiment 2 Report

## Setup

- Input: `gpt-oss-tokenized-traces.csv`
- Traces: 732
- Questions: 30
- Mixed-outcome questions: 24
- Mixed-outcome traces: 597
- Folds: 5 question-holdout
- Seed: 73
- Min motif support: 12
- Motif lengths: 1 to 3

## Early-Prefix Prediction

| Prefix Fraction | Motif AUC | Motif Q-Local AUC | Length Q-Local AUC | Motif Balanced Accuracy | Avg Selected Features |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.5506 | 0.4858 | 0.6728 | 0.5576 | 57.2 |
| 0.50 | 0.6277 | 0.5485 | 0.6793 | 0.5935 | 142.2 |
| 0.75 | 0.6702 | 0.5543 | 0.6871 | 0.6015 | 215.4 |
| 1.00 | 0.5748 | 0.6275 | 0.6836 | 0.6178 | 276.2 |
