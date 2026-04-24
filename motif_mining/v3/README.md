# Motif Mining Toolkit (v3)

Production-style Python toolkit for mining discriminative reasoning motifs from tokenized math reasoning traces.

This toolkit is designed for **trace-level document support** (a motif counts at most once per trace) and explicitly compares **successful vs unsuccessful** traces.

## What Is Implemented

- Shared preprocessing and corpus summaries
- Bounded-gap skip-gram mining (beyond contiguous n-grams)
- Sequential pattern mining with gaps
  - `python` backend (always available)
  - `prefixspan` backend (optional; auto-fallback)
- Redundancy reduction for sequential patterns
- Sequential rule mining (`antecedent => consequent`)
- Discriminative scoring and optional significance tests
- CLI + script wrappers + unit tests
- CSV/JSON outputs and optional matplotlib plots

## Expected Input CSV

Required columns:

- `question_id`
- `tokenized_trace`
- `is_correct`

Trace identity columns:

- either `trace_id`, or
- composite key: `question_id` + `sample_id`
- `quesiton_id` (typo) is also accepted and normalized to `question_id`

Assumptions:

- `tokenized_trace` is token-delimited by a space by default (override with `--delimiter`, e.g. `|`)
- `is_correct` accepts booleans or 0/1-like strings (`True`, `False`, `1`, `0`, etc.)

Example trace:

```text
analyze|compute|apply-formula|instantiate|compute|check-constraint|conclude
```

## Project Layout

```text
analysis/
  __init__.py
  io_utils.py
  scoring.py
  skipgrams.py
  sequential_patterns.py
  rules.py
  stats.py
  cli.py
scripts/
  run_skipgram_analysis.py
  run_prefixspan_analysis.py
  run_rule_mining.py
  summarize_results.py
tests/
  test_skipgrams.py
  test_scoring.py
  test_io.py
requirements.txt
README.md
```

## Installation

From `motif_mining/v3`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional backend:

```bash
pip install prefixspan
```

## CLI Usage

From `motif_mining/v3`:

```bash
python -m analysis.cli summarize --input encoded_traces_0.csv --outdir results/run0 --plots
python -m analysis.cli skipgrams --input encoded_traces_0.csv --outdir results/run0 --min-len 2 --max-len 4 --max-gap 2
python -m analysis.cli patterns --input encoded_traces_0.csv --outdir results/run0 --backend auto --min-support 5 --max-len 4 --reduce-redundancy
python -m analysis.cli rules --input encoded_traces_0.csv --outdir results/run0 --min-support 5 --max-len 4
python -m analysis.cli all --input encoded_traces_0.csv --outdir results/run0 --backend auto --plots
```

Useful preprocessing flags (available in subcommands):

- `--deduplicate`
- `--dedupe-per-question`
- `--min-trace-len`
- `--max-trace-len`
- `--max-questions` (first N unique `question_id`, default no cap)

## Outputs

Each run directory contains files such as:

- `summary.json`
- `cleaned_traces.csv`
- `skipgrams_all.csv`
- `skipgrams_success.csv`
- `skipgrams_failure.csv`
- `skipgrams_report.md`
- `sequential_patterns_all.csv`
- `sequential_patterns_success.csv`
- `sequential_patterns_failure.csv`
- `sequential_patterns_report.md`
- `rules_all.csv`
- `rules_success.csv`
- `rules_failure.csv`
- `rules_report.md`
- `report.md` (combined run summary + links)
- `plots/trace_length_histogram.png`
- `plots/skipgrams_top_success.png`, etc.

## Method Notes

### Skip-grams vs contiguous n-grams

- Contiguous n-grams require adjacent tokens.
- Skip-grams here allow bounded gaps between neighboring motif tokens.
- Example with max gap `g=1`: `a _ b` is allowed, `a _ _ b` is not.

### Sequential patterns vs skip-grams

- Skip-grams are bounded-gap local motifs.
- Sequential patterns are more general subsequences with gaps (possibly long-range).
- This toolkit mines frequent subsequences and then scores their discriminativeness.

### Discriminative metrics

For each motif/rule:

- `success_count`, `failure_count`
- `success_support`, `failure_support`
- `support_difference = success_support - failure_support`
- `lift = success_support / (failure_support + epsilon)`
- `log_odds_ratio` (smoothed)
- `p_value` via Fisher exact test when `scipy` is available
- `q_value` via Benjamini-Hochberg correction

### Redundancy reduction rule

A pattern `P` is dropped if a longer supersequence `Q` exists with:

- `P` subsequence of `Q`
- nearly identical support in both classes (within tolerance)
- equal or better discriminative score (`abs_log_odds_ratio`)

This is a pragmatic closed-like filter, not exact closed pattern mining.

## Complexity Caveats

- Pure Python sequential mining enumerates subsequences up to `max_len`.
- Approximate complexity: `O(N * sum_k C(L, k))` where `N` is number of traces and `L` average trace length.
- Rule mining can be expensive due to candidate pair expansion; use lower `--max-candidates`, smaller `--max-len`, or higher `--min-support` for large datasets.

## Script Wrappers

Convenience wrappers (equivalent to CLI subcommands):

```bash
python scripts/run_skipgram_analysis.py --input encoded_traces_0.csv --outdir results/run0
python scripts/run_prefixspan_analysis.py --input encoded_traces_0.csv --outdir results/run0 --backend auto
python scripts/run_rule_mining.py --input encoded_traces_0.csv --outdir results/run0
python scripts/summarize_results.py --outdir results/run0
```

## Tests

From `motif_mining/v3`:

```bash
pytest -q
```

Covers:

- token parsing
- skip-gram generation correctness
- trace-level motif uniqueness behavior
- scoring/log-odds sanity and q-value creation

## Assumptions

- Token order is meaningful and motifs are ordered sequences.
- Support is document-level (trace-level), never raw in-trace frequency.
- Correctness labels are only used for scoring/comparison, not parsing.
- For very large datasets, prefer optional `prefixspan` backend or tighter search constraints.
