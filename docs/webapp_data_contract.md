# Webapp Data Contract

This document defines the canonical payloads for the researcher-facing webapp.
It is the source of truth for the export script, local API, and SPA.

## Curated Pilot Inputs

The app path uses these repo-local pilot fixtures:

- tokenized traces:
  - `tokenizer_local/v2/tokenized_pilot_traces.csv`
- raw reasoning traces:
  - `tokenizer_local/question_independent_incorrect_see_correctness/pilot_traces.csv`

These fixtures provide:

- 22 questions
- shared token vocabulary across questions
- raw question text, gold answers, predicted answers, and reasoning traces

For corpus-global motifs, the webapp path treats `motif_mining/v3` as the
canonical mining implementation.

## Payloads

### `overview.json`

Top-level corpus data for the landing page.

Fields:

- `corpus_label`
- `num_questions`
- `num_traces`
- `num_success`
- `num_failure`
- `success_rate`
- `avg_token_count`
- `median_token_count`
- `featured_question_ids`
- `story_sections`
- `top_success_motifs`
- `top_failure_motifs`

### `questions.json`

Flat list of explorer rows. Each row includes:

- `question_id`
- `question_text`
- `gold_answer`
- `benchmark_name`
- `total_traces`
- `success_count`
- `failure_count`
- `success_rate`
- `avg_token_count`
- `median_token_count`
- `distinct_predicted_answers`
- `local_motif_separation`
- `top_success_motif`
- `top_failure_motif`
- `tags`

### `question/<id>.json`

Full question detail payload used by the question page.

Fields:

- question metadata and corpus tags
- `insights`
- `answer_distribution`
- `local_motifs`
- `global_motifs`
- `representative_traces`
- `all_traces`

## Core Entities

### Motif row

- `motif`
- `tokens`
- `length`
- `scope`
- `direction`
- `success_count`
- `failure_count`
- `success_support`
- `failure_support`
- `support_difference`
- `lift`
- `log_odds_ratio`
- optional `q_value`

### Trace summary

- `trace_id`
- `question_id`
- `sample_id`
- `attempt_index`
- `predicted_answer`
- `is_correct`
- `tokenized_trace`
- `tokens`
- `token_count`
- `reasoning_trace`
- `matched_success_motifs`
- `matched_failure_motifs`

## Export Rules

- Question-local motifs are contiguous motifs of length 1-3 with document-level
  support.
- Question-local motifs are hidden when the question lacks both success and
  failure traces or has weak support.
- Corpus-global motifs are mined through `motif_mining/v3` and then filtered to
  those present in each question’s traces.
- Representative traces are selected deterministically from the question’s
  traces and stored directly in the question detail payload.
