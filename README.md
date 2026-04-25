# Reasoning-Motifs

Reasoning-Motifs is a small research toolkit for collecting reasoning traces from
LLMs on math benchmarks, then analyzing which trace patterns correlate with
correct and incorrect answers.

The current primary workflow is built around [reasoner.py](./reasoner.py) and
the [`reasoning_trace_sampling`](./reasoning_trace_sampling) package.

For the researcher-facing webapp path, the canonical downstream analysis stack
is:

1. curated pilot/tokenized trace export
2. corpus-global motif mining through `motif_mining/v3`
3. read-only web artifacts served by a thin local API
4. SPA-based exploration and storytelling

## Main Workflow

1. Load benchmark questions from `data/benchmarks`.
2. Sample one or more reasoning traces per question through the Together API.
3. Save accepted trajectories to CSV in `data/traces`.
4. Summarize question-level behavior or run downstream motif analysis.

The most important runtime paths are:

- [`reasoner.py`](./reasoner.py): CLI entrypoint
- [`reasoning_trace_sampling/benchmarking.py`](./reasoning_trace_sampling/benchmarking.py): benchmark registry and loaders
- [`reasoning_trace_sampling/sampling.py`](./reasoning_trace_sampling/sampling.py): single-question prompting, answer parsing, optional LLM verification
- [`reasoning_trace_sampling/trajectory_collection.py`](./reasoning_trace_sampling/trajectory_collection.py): fixed-budget and adaptive trajectory collection
- [`reasoning_trace_sampling/question_stats.py`](./reasoning_trace_sampling/question_stats.py): per-question rollups from trajectory CSVs

For a deeper walkthrough of the trajectory-sampling module, see
[`docs/trajectory_sampling.md`](./docs/trajectory_sampling.md).

## Repository Layout

- `data/benchmarks`: local benchmark datasets
- `data/traces`: collected trajectory CSVs and event logs
- `reasoning_trace_sampling`: modular trace collection package
- `reasoning_motifs_web`: shared export and API payload models for the webapp
- `motif_mining/v3`: canonical global motif mining toolkit for the webapp path
- `scripts`: dataset prep and analysis helpers
- `experiments`: lightweight experiment notes

## CLI Commands

List known benchmark presets:

```bash
python3 reasoner.py benchmarks
```

Collect a single trace for one question:

```bash
python3 reasoner.py one \
  --benchmark-name sample_math \
  --question-index 0
```

Collect a fixed number of accepted traces per question:

```bash
python3 reasoner.py many \
  --benchmark-name sample_math \
  --samples-per-question 2 \
  --max-attempts-per-question 4 \
  --show-progress
```

Run adaptive collection, where mixed-outcome questions get more budget:

```bash
python3 reasoner.py adaptive-many \
  --benchmark-name sample_math \
  --scout-samples 2 \
  --target-samples 4 \
  --scout-max-attempts-per-question 4 \
  --max-attempts-per-question 8 \
  --show-progress
```

Summarize an existing trajectory CSV by question:

```bash
python3 reasoner.py stats \
  --input data/traces/sample_math.csv \
  --output data/traces/question_stats.csv
```

## Environment

The collection commands expect a Together API key in `TOGETHER_API_KEY`. The
API shim also loads a repo-local `.env` file if present.

Most of the core collection code is stdlib-only. The only obvious non-stdlib
helper is `pandas`, used by `scripts/normalize_aimo_parquet.py`.

## Smoke Test

A small local smoke test is available without API credentials:

```bash
python3 -m unittest tests/test_trajectory_sampling_smoke.py
```
