# Tokenization Ablation Note

This note compares lightweight token transformations on
[`gpt-oss-tokenized-traces.csv`](/Users/czqz/440-project/Reasoning-Motifs/gpt-oss-tokenized-traces.csv)
under the same 5-fold question-holdout setup used for Experiment 2.

## Main Finding

The current tokenization appears too specific for this dataset size.
The baseline file contains:

- 732 traces
- 30 questions
- 845 unique full tokens
- 13 unique action heads
- 488 singleton full tokens
- 565 full tokens with support <= 2

Most of the sparsity is coming from suffixes such as
`analyze:setup`, `analyze:goal`, `compute:product`, and similar variants.
Collapsing tokens to their action head improves both global AUC and
question-local AUC substantially.

## Baseline From Experiment 2

| Representation | Prefix | Features | AUC | Q-Local AUC |
| --- | ---: | ---: | ---: | ---: |
| original 1-3 gram motifs | 0.75 | 215.4 | 0.6702 | 0.5543 |
| original 1-3 gram motifs | 1.00 | 276.2 | 0.5748 | 0.6275 |

## Best Ablations

| Representation | Prefix | Motif Order | Avg Selected Features | AUC | Q-Local AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| action only | 1.00 | 1 | 10.8 | 0.7355 | 0.7025 |
| action only | 1.00 | 2 | 70.4 | 0.7124 | 0.6895 |
| head + coarse tail | 1.00 | 1 | 108.2 | 0.6981 | 0.7068 |
| action only | 0.75 | 1 | 9.8 | 0.7101 | 0.6332 |
| head + coarse tail | 0.75 | 1 | 95.2 | 0.7079 | 0.6536 |

Definitions:

- `action only`: keep only the token head before `:`
- `head + coarse tail`: emit both the action head and a coarse first tail segment

Example:

- `analyze:problem-setup` -> `analyze`
- `rewrite:parity-rewrite` -> `rewrite`
- `compute:product` -> `compute`

For `head + coarse tail`:

- `analyze:problem-setup` -> `analyze`, `analyze:problem`

## Interpretation

The strongest improvement came from collapsing semantic micro-labels into a
small action vocabulary. This suggests the current labels are overfitting to
surface phrasing or problem-specific details instead of capturing reusable
reasoning moves.

The best coarse-grained success-leaning actions were:

- `rewrite`
- `compare`
- `simplify`
- `compute`
- `conclude`

The strongest failure-leaning actions were:

- `check-constraint`
- `case-split`
- `analyze`

These signals are much easier to estimate reliably than the original
suffix-heavy inventory.

## Recommended Next Step

If the goal is to improve Experiment 2 on this dataset, the first change to try
is:

1. Replace full `head:tail` tokens with `action only` tokens.
2. Use unigram motifs first.
3. Re-run the same 5-fold question-holdout evaluation.

After that, the next best variant to test is `head + coarse tail` unigrams if
you want slightly more interpretability without bringing back full sparsity.
