#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TRACE_DIR="data/traces/target_motif_v1"
mkdir -p "$TRACE_DIR"

SAMPLE_SIZE="${SAMPLE_SIZE:-700}"
SAMPLE_SEED="${SAMPLE_SEED:-41}"
MAX_TOKENS="${MAX_TOKENS:-12288}"
WORKERS="${WORKERS:-30}"
RUN_NAME="expanded_pool_s${SAMPLE_SIZE}_seed${SAMPLE_SEED}_w${WORKERS}_tok${MAX_TOKENS}"

COMMON_ARGS=(
  --temperature-schedule 0.2,0.5,0.8
  --scout-samples 8
  --target-samples 24
  --scout-max-attempts-per-question 48
  --max-attempts-per-question 120
  --timeout-seconds 120
  --workers "$WORKERS"
  --show-progress
)

python3 reasoner.py adaptive-many \
  --benchmark-name expanded_motif_pool \
  --sample-size "$SAMPLE_SIZE" \
  --sample-seed "$SAMPLE_SEED" \
  --max-tokens "$MAX_TOKENS" \
  "${COMMON_ARGS[@]}" \
  --output "$TRACE_DIR/${RUN_NAME}_mixedtemp_adaptive.csv" \
  --summary-output "$TRACE_DIR/${RUN_NAME}_mixedtemp_adaptive_question_summary.csv" \
  --log-jsonl "$TRACE_DIR/${RUN_NAME}_mixedtemp_adaptive.log.jsonl"

python3 scripts/analyze_sampled_dataset.py \
  --input "$TRACE_DIR/${RUN_NAME}_mixedtemp_adaptive.csv" \
  --output-dir "$TRACE_DIR/${RUN_NAME}_analysis"
