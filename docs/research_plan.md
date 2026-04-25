# Reasoning-Motifs — Research Plan

## Research question

**Which reasoning patterns in LLM math traces are predictive of answer
correctness, controlling for question difficulty?**

The core methodological commitment of this project is **within-question
contrast of mixed-outcome traces**: we only trust signal that survives
restricting to questions where the model produces both correct and incorrect
traces. This eliminates the question-difficulty confound that contaminates
naive global motif mining.

## Thesis (what we will claim in the paper)

1. Individual qualified reasoning-action tokens (`basetype:qualifier`,
   e.g. `analyze:parity`, `guess:final-answer`) carry significant
   per-question signal about answer correctness.
2. A simple sparse linear classifier on those token features beats the
   per-question majority-class baseline on held-out traces.
3. The diagnostic tokens are stable across multiple labeling LLMs, arguing
   the result reflects properties of the reasoning, not of any one
   tokenizer's quirks.
4. Failure modes cluster into interpretable types tied to question
   structure (geometry / number-theory / combinatorics / etc.).

## Pipeline overview

```
raw trace CSV
    -> [Stage 0]  LLM label audit            -> audited_traces.csv
    -> [Stage 1]  granular tokenizer          -> tokenized.csv
    -> [Stage 2]  Exp. 1 — single-token Δ
    -> [Stage 3]  Exp. 2 — fingerprint classifier
    -> [Stage 4]  Exp. 3 — position conditioning
    -> [Stage 5]  Exp. 4 — cross-tokenizer robustness
    -> [Stage 6]  Exp. 5 — failure-mode clustering
```

Each stage produces a CSV / JSON artifact and a short report. Nothing is
overwritten — every stage is idempotent and resumable.

---

## Stage 0 — LLM label audit

### Why

`is_correct` labels can be wrong via lenient grading (false positives) or
strict-form grading (false negatives, e.g. `1/2` vs `0.5`). Within-question
contrast is exquisitely sensitive to label noise — a single mislabeled trace
in a 4-trace mixed question can flip the dominant signal. Auditing first
keeps Stage 2's leaderboard interpretable.

### Approach

A single judge LLM scores each `(question, gold_answer, predicted_answer,
reasoning_trace)` tuple and returns a structured verdict. Math equivalence
is left to the LLM rather than implemented in code (sympy / regex are
brittle on real benchmark answers).

### Judge prompt (pinned)

System: "You are a strict but fair math grader. Decide whether the
predicted answer is mathematically equivalent to the gold answer. Be
robust to formatting (fractions vs decimals, latex vs plain, equivalent
algebraic forms)."

User payload (one trace at a time):

```json
{
  "question": "...",
  "gold_answer": "...",
  "predicted_answer": "...",
  "trace_tail": "<last ~500 chars of reasoning_trace>"
}
```

Required JSON output:

```json
{
  "verdict": "correct" | "incorrect" | "ambiguous" | "non_attempt",
  "confidence": 0.0 to 1.0,
  "reason": "one sentence",
  "trace_concludes_predicted": true | false
}
```

### Configuration

- Model: a strong instruction-tuned model (separate from the trace-generating
  model and from the tokenizer model, to avoid model-self-bias).
- Temperature 0; deterministic.
- One request per trace. Cache by `(question_id, sample_id, judge_model)`
  so re-runs are free.
- Audit CSV columns appended to the source: `judge_verdict`,
  `judge_confidence`, `judge_reason`, `judge_trace_concludes_predicted`,
  `judge_model`, `judge_disagreement` (`True` iff
  `judge_verdict != original is_correct`).

### Decision rules (pre-registered)

For the analysis corpus:

- Drop traces with `judge_verdict == "non_attempt"`.
- Drop traces where `judge_trace_concludes_predicted == False` AND
  `judge_confidence >= 0.8` (these are answer-extraction failures, not
  reasoning failures).
- Adopt `judge_verdict` as the working `is_correct` label when
  `judge_confidence >= 0.8`.
- Keep original label, flag `is_low_confidence_label = True` when judge
  is below 0.8.
- Run the headline analyses on the high-confidence subset; report a
  sensitivity check using the full set as supplementary.

### Audit metrics to report

- `% disagreement` between judge and original label.
- `% disagreement among confident judgements` (the meaningful one).
- `% non_attempt`, `% ambiguous`.
- Spot-check: 30 random disagreements reviewed by hand and labeled
  agree-with-judge / agree-with-original / unsure. If judge wins ≥ 80%,
  proceed; otherwise iterate the prompt.

---

## Stage 1 — Tokenization (granular)

### Why

Bare basetype tokens (`analyze`, `compute`) are too coarse: two traces with
the same step type but different mathematical content look identical. The
new tokenizer emits `basetype:qualifier` (e.g., `analyze:parity`,
`compute:modular-inverse`), making single tokens diagnostic.

### Implementation

`tokenizer/tokenize_traces.py` (already updated). Two-phase per question:
1. Metadata phase: build a question-specific tokenization guide that
   constrains qualifiers.
2. Per-trace phase: emit `basetype:qualifier` token sequence.

The basetype vocabulary is **fixed** (13 tokens). Qualifiers are drawn
from a suggested pool but extensible per question.

### Inputs / outputs

- Input: audited CSV from Stage 0.
- Output: tokenized CSV per tokenizer model. For Exp. 4 (robustness),
  re-tokenize the same audited CSV under ≥ 3 different tokenizer LLMs.

### Quality gate

After tokenization, drop rows where `tokenized_trace == "MISSING"` and rows
whose tokens have no colon (the model failed to follow the qualifier
schema). Report drop rates per tokenizer.

---

## Stage 2 — Experiment 1: within-question single-token Δ

### Hypothesis

H1: there exist tokens T whose presence in a trace shifts the
within-question success probability away from the question's baseline,
and this effect generalizes across mixed-outcome questions.

### Procedure

1. Restrict to mixed-outcome questions M (≥ 1 success and ≥ 1 failure
   trace post-audit). Report `|M|`.
2. For each token T appearing in ≥ 10 questions in M, compute per question:
   - `succ_with_T(q) = P(correct | T ∈ trace, q)`
   - `succ_without_T(q) = P(correct | T ∉ trace, q)`
   - `Δ_q(T) = succ_with_T(q) − succ_without_T(q)`
   - Skip if either side has 0 traces.
3. Aggregate: paired sign test on `Δ_q(T)` series; report `mean_Δ`,
   `median_Δ`, 95% bootstrap CI (10k resamples), sign-test p, BH q-value
   at FDR=0.10.
4. Repeat with token count buckets `{0, 1, ≥2}` (the "thrashing"
   hypothesis: 2× backtrack is worse than 1× backtrack).

### Sanity checks

- Permutation: shuffle correctness labels within each question, recompute
  ranking. Top tokens should collapse to noise.
- Length control: regress per-token `mean_Δ` on average trace length;
  large residual is the real reasoning signal.

### Deliverables

- `experiments/exp1_single_token_diagnostics/leaderboard_lifesavers.csv`
- `experiments/exp1_single_token_diagnostics/leaderboard_killers.csv`
- `experiments/exp1_single_token_diagnostics/forest_plot.png` (top-15 each)
- `experiments/exp1_single_token_diagnostics/permutation_null.json`

### Headline metrics

- `n_lifesaver` and `n_killer` tokens at q ≤ 0.10.
- Top-3 named tokens with `mean_Δ` and 95% CI for the abstract.

---

## Stage 3 — Experiment 2: trace fingerprint classifier

### Hypothesis

H2: a sparse linear model on bag-of-tokens features predicts correctness
above the per-question majority-class baseline; non-zero features overlap
with Stage 2's leaderboards (cross-validation that the within-question
signal is real).

### Procedure

1. Features per trace:
   - presence vector over qualified-token vocabulary
   - count vector
   - count vector + trace length
2. Models (in increasing complexity):
   - `M0`: predict per-question majority class.
   - `M1`: logistic regression on trace length only.
   - `M2`: L1-regularized logistic regression on tokens, with
     `question_id` one-hot fixed effects.
   - `M3`: `M2` + bigrams.
3. Cross-validation: 5-fold **group-k-fold by `question_id`** so no
   question leaks across train/test. Tune L1 strength on inner CV.

### Metrics

- Primary: AUC and balanced accuracy on held-out traces.
- AUC lift over `M0` (the interpretable headline number — "predictability
  above question difficulty").
- Calibration: reliability diagram + ECE.
- Feature overlap: Jaccard between `M2` non-zero features and Stage 2's
  top-50 leaderboard. Target ≥ 0.5 (independent confirmation).

### Sanity checks

- Train on shuffled labels: AUC ≈ 0.5.
- Drop the question fixed effect: AUC inflates. This is the *positive*
  control showing the difficulty confound exists.

### Deliverables

- `experiments/exp2_classifier/cv_metrics.json`
- `experiments/exp2_classifier/feature_overlap.json`
- `experiments/exp2_classifier/coefficients.csv`

---

## Stage 4 — Experiment 3: position-conditioned analysis

### Hypothesis

H3: where a token appears within the trace matters. Late `guess:*` and
`conclude:*` after `backtrack:*` are more diagnostic than the same tokens
appearing early.

### Procedure

1. Bin each token occurrence by trace-relative position:
   `first_third`, `middle_third`, `last_third`.
2. For each (token, position-bin), repeat Stage 2's within-question Δ.
3. Compare position-conditioned `|Δ|` distributions to position-collapsed
   `|Δ|`.

### Metrics

- Per (token, position): mean Δ, CI, q-value (same schema as Stage 2).
- Aggregate: number of tokens with a position-bin whose `|Δ|` is ≥ 1.5×
  the position-collapsed `|Δ|`. Test with paired bootstrap.

### Sanity check

- Should recover the obvious: `conclude:*` in `last_third` is uninformative
  (universal); `conclude:*` in `first_third` is rare and likely
  failure-correlated.

### Deliverables

- `experiments/exp3_position/position_conditioned_deltas.csv`
- `experiments/exp3_position/position_lift_plot.png`

---

## Stage 5 — Experiment 4: cross-tokenizer robustness (reviewer-killer)

### Hypothesis

H4: the top diagnostic tokens are consistent across different tokenizer
LLMs, i.e. the result is a property of the reasoning, not of one
labeling model's quirks.

### Procedure

1. Run Stage 2 separately under each of ≥ 3 tokenizer LLMs (re-run
   Stage 1 with different `--model`).
2. Compare rankings at two levels:
   - Basetype level (qualifiers collapsed): direct comparison.
   - Full-token level: fuzzy-match qualifier synonyms (canonicalize
     `mod` ≡ `modular`, `eq` ≡ `equation`) before comparison.
3. Compute pairwise Spearman ρ on token rankings (by `mean_Δ`).
4. Identify the **stable diagnostic set**: tokens in top-30 for ≥ ⌈n/2⌉
   tokenizers.

### Metrics

- Pairwise Spearman ρ (target ρ ≥ 0.4).
- Size of stable diagnostic set; per-token tokenizer-vote count.

### Deliverables

- `experiments/exp4_cross_tokenizer/pairwise_spearman.csv`
- `experiments/exp4_cross_tokenizer/stable_diagnostic_set.csv`

---

## Stage 6 — Experiment 5: per-question failure-mode clustering

### Hypothesis

H5: questions cluster by the qualitative shape of their failure-mode
signature (e.g., NT-style failures vs geometry-style failures).

### Procedure

1. For each q in M, compute the per-question token-Δ vector from
   Stage 2.
2. Cluster questions on those vectors (UMAP for projection,
   HDBSCAN for clustering, fixed seed).
3. Manually label clusters by inspecting member questions and dominant
   killer tokens.

### Metrics

- Cluster stability across resamples (ARI ≥ 0.4 over 10 bootstrap resamples).
- Qualitative narrative of ≥ 3 distinct clusters with interpretable token
  signatures.

### Deliverables

- `experiments/exp5_failure_modes/cluster_signatures.csv`
- `experiments/exp5_failure_modes/umap.png`

---

## Pre-registered headline metrics (for paper abstract)

1. `n_lifesaver_tokens` and `n_killer_tokens` at q ≤ 0.10 (Stage 2).
2. Stage 3 classifier AUC and AUC-lift over `M0`.
3. Stage 5 mean pairwise Spearman ρ across tokenizer LLMs.
4. Three named killer motifs with `mean_Δ` and 95% CI.

## Reproducibility

- Single config: `experiments/config.yaml` records corpus path, judge
  model, tokenizer models, seeds, FDR threshold, fold count.
- Each stage's output JSON records: git commit SHA, config hash, input
  file SHA-256, library versions.
- Random seeds frozen at `73` (matches corpus suffix).

## Risks and mitigations

- **Label-judge bias toward original label.** Mitigation: judge prompt
  emphasizes mathematical equivalence; spot-check disagreements; use a
  judge model from a different lab than both the trace generator and the
  tokenizer.
- **Qualifier vocabulary drift across tokenizers.** Mitigation: Stage 5
  has explicit basetype-level fallback; canonicalize qualifier synonyms
  before full-token comparison.
- **Insufficient mixed-outcome questions** (`|M|` too small). Mitigation:
  if `|M| < 50`, expand corpus by sampling more traces per question
  with higher temperature on questions currently lacking mixed outcomes,
  rather than weakening the within-question design.
- **Trace truncation masquerading as failure.** Mitigation: Stage 0 drops
  via `judge_trace_concludes_predicted == False`; report drop rate.
- **Multiple-comparisons after exploration.** Mitigation: top-K leaderboard
  is exploratory; primary inferential claim is Stage 3's held-out AUC,
  which is one number with one CI.

## Non-goals

- We are not building a new motif-mining algorithm. `motif_mining/v3`
  remains as a baseline and is not extended.
- We are not training reasoning models. The traces are an input dataset.
- We are not claiming causal mechanism — only diagnostic / predictive
  patterns conditional on question.
