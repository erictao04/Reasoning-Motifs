# GPT-OSS Chunked Live 1 Assessment

Source file: `tokenizer/gpt_oss_subset_15q_all_traces_tokenized_gpt-oss-chunked_live_1.csv`

Snapshot assessed on 2026-04-25:

- 330 written traces
- 11 complete questions
- 251 correct / 79 incorrect
- 0 `MISSING` rows

## Tokenization quality

- 210 unique tokens across 330 traces
- Mean trace length: 11.01 tokens
- Median trace length: 12 tokens
- Max trace length: 22 tokens
- Token type counts:
  - `action`: 1900
  - `milestone`: 1109
  - `strategy`: 548
  - `noise`: 75
- Rows containing `noise:corrupted-span`: 58 / 330 = 17.6%
- Rows containing digit-bearing tokens: 74 / 330 = 22.4%
- Milestone tokens containing digits: 215 / 1109 = 19.4%
- Rows containing `final*` tokens: 57 / 330 = 17.3%
- Rows containing `solution` tokens: 49 / 330 = 14.8%

Strengths:

- No parse failures in the written subset.
- Token schema is normalized and consistent; no malformed tokens were detected.
- Vocabulary is much less sparse than the earlier free-form run.

Risks:

- Incorrect traces are substantially shorter and much noisier than correct traces.
- Some milestones are still too literal or answer-adjacent.
- Numeric equation/value milestones remain fairly common.

Observed class differences:

- Correct traces: mean length 12.30, noise rate 4.0%
- Incorrect traces: mean length 6.90, noise rate 60.8%

## Experiment 2 style results

5-fold question holdout, motif lengths 1 to 3, min support 6.

### New tokenizer, full tokens

| Prefix | Motif AUC | Motif q-local AUC | Length AUC |
|---|---:|---:|---:|
| 0.25 | 0.6409 | 0.6930 | 0.8365 |
| 0.50 | 0.7539 | 0.7814 | 0.8436 |
| 0.75 | 0.8113 | 0.8408 | 0.8432 |
| 1.00 | 0.8381 | 0.8715 | 0.8448 |

### Older tokenizer on overlapping question set

Old comparison set: 291 traces across the same 11 question IDs from `gpt-oss-tokenized-traces.csv`.

| Prefix | Old Motif AUC | Old Motif q-local AUC | Old Length AUC |
|---|---:|---:|---:|
| 0.25 | 0.5082 | 0.4942 | 0.8735 |
| 0.50 | 0.5674 | 0.5906 | 0.8825 |
| 0.75 | 0.4688 | 0.5464 | 0.8885 |
| 1.00 | 0.5597 | 0.6391 | 0.8897 |

Interpretation:

- The new tokenizer is dramatically stronger than the older one on the overlapping question set.
- Unlike the old run, motif signal now stays strong all the way to the full trace.
- The new motif model almost catches the trace-length baseline globally, and beats the old motif representation by a large margin.

## Ablations

Full-trace, 5-fold question holdout, min support 6.

| Variant | Motif AUC | Motif q-local AUC | Length AUC |
|---|---:|---:|---:|
| Full tokens | 0.8381 | 0.8715 | 0.8448 |
| Action only | 0.8378 | 0.8818 | 0.8448 |
| No noise token | 0.8216 | 0.8439 | 0.8587 |
| Drop noise + end markers | 0.8025 | 0.8401 | 0.8358 |
| Actions + strategies only | 0.8240 | 0.8354 | 0.8484 |
| Milestones only | 0.4505 | 0.5094 | 0.8545 |
| Drop digit-bearing tokens | 0.8381 | 0.8697 | 0.8344 |

Interpretation:

- `noise:corrupted-span` helps, especially at early prefixes, but it is not the whole story.
- End-of-trace markers (`action:conclude`, `final*`, `solution`) add signal, but performance remains strong even after removing them.
- Milestones alone perform poorly; most of the predictive lift is coming from actions and strategies.
- Dropping digit-bearing tokens barely changes performance, so exact numeric values are not the main driver of the current lift.

## Overall read

This tokenizer is a real improvement over the older free-form version:

- cleaner outputs
- no `MISSING` rows in the written subset
- much stronger held-out predictive signal
- substantially lower token sparsity

The main remaining quality issue is not parse failure anymore. It is representation design:

- milestones are still weaker than actions/strategies
- some milestones are too literal
- `noise` and completion cues are predictive enough that they can partially shortcut reasoning quality

## Recommended next changes

1. Keep `noise:corrupted-span`, but report experiments both with and without it.
2. Remove answer-adjacent milestone families such as `final*` and `solution` from the canonical output.
3. Tighten milestone templates toward structural states and away from literal equations.
4. Consider compressing repeated action runs to reduce verbosity bias.
5. Re-run this assessment once all 15 questions finish, since this snapshot currently covers only 11 completed questions.
