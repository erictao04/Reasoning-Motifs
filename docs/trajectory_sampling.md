# Trajectory Sampling

This document is the practical guide to the `reasoning_trace_sampling` module.
It focuses on the active collection flow rather than the older ad hoc motif
scripts.

## Purpose

The module collects reasoning trajectories from a model, extracts a final
answer, checks whether that answer matches the benchmark gold answer, and saves
accepted trajectories for downstream analysis.

An accepted trajectory is a model response that produced a clear final answer.
Accepted trajectories may still be wrong; "accepted" only means the run yielded
something parseable enough to score.

## Core Components

### `BenchmarkRegistry`

File: `reasoning_trace_sampling/benchmarking.py`

Responsibilities:

- define benchmark presets
- resolve benchmark file paths
- infer answer formats when loading ad hoc files
- load CSV or JSON question sets into `BenchmarkItem` objects
- filter question slices by level or random sample

Each benchmark item tracks:

- `question_id`
- `question`
- `gold_answer`
- `answer_format`
- optional metadata like `level`, `subject`, and `source_id`

### `APIShim`

File: `reasoning_trace_sampling/api_shim.py`

Responsibilities:

- load `TOGETHER_API_KEY` from the environment or `.env`
- build Together chat-completions requests
- throttle request starts
- retry rate limits with backoff
- surface fatal model-configuration errors cleanly

This layer is intentionally thin so the sampling logic can stay mostly provider
agnostic.

### `ReasoningTraceSampling`

File: `reasoning_trace_sampling/sampling.py`

Responsibilities:

- build prompt suffixes that enforce benchmark-specific answer markers
- make one API request for one benchmark item
- parse the model response into:
  - visible final response text
  - reasoning trace
  - predicted answer
  - correctness fields
  - usage metadata
- optionally fall back to an LLM verifier when deterministic parsing fails

Answer formats supported today:

- `final`: expects a trailing `FINAL: <answer>` line
- `hash`: expects a trailing `#### <answer>` line
- `boxed`: expects a trailing `\boxed{answer}` line

### `TraceCollector`

File: `reasoning_trace_sampling/trajectory_collection.py`

Responsibilities:

- collect a fixed number of accepted samples per question
- retry each question up to a maximum attempt budget
- run attempts concurrently across questions
- stream accepted rows to CSV while collection is still running
- emit progress updates and JSONL event logs

Important behavior:

- sample IDs count accepted trajectories, not raw attempts
- attempt indices count every model call, including failed parse attempts
- if a response has no clear answer marker, it does not become a trajectory row

### `AdaptiveTraceCollector`

File: `reasoning_trace_sampling/trajectory_collection.py`

Responsibilities:

- scout each question with a small number of accepted samples
- densify only the questions that show both right and wrong outcomes
- write a question-level adaptive summary

This is the current scaling strategy for motif collection because it spends more
budget where trace diversity is most informative.

### `QuestionTraceAnalyzer`

File: `reasoning_trace_sampling/question_stats.py`

Responsibilities:

- read a trajectory CSV
- aggregate rows by question
- compute right/wrong counts and mixed-outcome status
- filter and write question-level summary CSVs

## Data Flow

### Fixed-budget collection

1. `reasoner.py many` resolves a benchmark and loads `BenchmarkItem` rows.
2. `ReasoningTraceSampling.ask_one()` prompts the model for each attempt.
3. `TraceCollector` keeps retrying each question until it has enough accepted
   samples or runs out of attempts.
4. Accepted trajectories are written as `TrajectoryRecord` rows.
5. The final CSV is saved under `data/traces`.

### Adaptive collection

1. `reasoner.py adaptive-many` runs a scout phase per question.
2. Questions with both right and wrong accepted trajectories are marked mixed.
3. Only mixed questions get additional sampling budget.
4. Final rows are written to the trajectory CSV, and question decisions are
   written to the adaptive summary CSV.

## Output Files

Trajectory CSV columns come from `TRAJECTORY_FIELDNAMES` in
`trajectory_collection.py`. Important fields include:

- `question_id`
- `sample_id`
- `attempt_index`
- `predicted_answer`
- `is_correct`
- `answer_source`
- `answer_validation`
- `reasoning_trace`
- token usage metadata
- request metadata and error text

Adaptive summary CSV columns come from `ADAPTIVE_SUMMARY_FIELDNAMES` and track:

- scout targets vs final targets
- accepted counts
- right/wrong splits
- whether the question was densified
- the final decision label

## Suggested Entry Points

For code reading:

1. `reasoner.py`
2. `reasoning_trace_sampling/trajectory_collection.py`
3. `reasoning_trace_sampling/sampling.py`
4. `reasoning_trace_sampling/benchmarking.py`

For local smoke checks:

1. `python3 reasoner.py benchmarks`
2. `python3 -m unittest tests/test_trajectory_sampling_smoke.py`

For real collection:

1. ensure `TOGETHER_API_KEY` is set
2. start with `sample_math`
3. then move to `many` or `adaptive-many` on a real benchmark preset

## Notes on Older Files

`encoder.py` and `parser.py` represent an older motif-encoding path. They are
still useful for historical context, but the active collection pipeline is the
modular `reasoning_trace_sampling` package plus `reasoner.py`.
