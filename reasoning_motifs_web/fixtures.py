from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Curated pilot inputs kept for tests and smaller demos.
PILOT_TOKENIZED_TRACE_CSV = (
    REPO_ROOT / "tokenizer_local" / "v2" / "tokenized_pilot_traces.csv"
)
PILOT_RAW_TRACE_CSV = (
    REPO_ROOT
    / "tokenizer_local"
    / "question_independent_incorrect_see_correctness"
    / "pilot_traces.csv"
)

# Expanded corpus inputs are the current default for the webapp path.
DEFAULT_TOKENIZED_TRACE_CSV = (
    REPO_ROOT
    / "tokenizer"
    / "clean_expanded_pool_s100_seed73_qwen25_7b_hot30_tokenized_gpt-oss-120b_0.csv"
)
DEFAULT_RAW_TRACE_CSV = (
    REPO_ROOT / "tokenizer" / "expanded_pool_s100_seed73_qwen25_7b_hot30.csv"
)
DEFAULT_GLOBAL_MOTIFS_CSV = (
    REPO_ROOT
    / "motif_mining"
    / "v3"
    / "results"
    / "clean_expanded_pool_s100_seed73_qwen25_7b_hot30_tokenized_gpt-oss-120b_0"
    / "skipgrams_all.csv"
)

# Generated demo artifacts live here by default.
DEFAULT_WEBAPP_ARTIFACT_DIR = REPO_ROOT / "webapp_artifacts" / "expanded_v1"
