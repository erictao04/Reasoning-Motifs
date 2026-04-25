from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Canonical curated pilot inputs for the webapp path.
DEFAULT_TOKENIZED_TRACE_CSV = (
    REPO_ROOT / "tokenizer_local" / "v2" / "tokenized_pilot_traces.csv"
)
DEFAULT_RAW_TRACE_CSV = (
    REPO_ROOT
    / "tokenizer_local"
    / "question_independent_incorrect_see_correctness"
    / "pilot_traces.csv"
)

# Generated demo artifacts live here by default.
DEFAULT_WEBAPP_ARTIFACT_DIR = REPO_ROOT / "webapp_artifacts" / "pilot_v1"
