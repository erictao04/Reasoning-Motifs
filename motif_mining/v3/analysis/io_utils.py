from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ["question_id", "trace_id", "tokenized_trace", "is_correct"]


TRUE_VALUES = {"1", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "false", "f", "no", "n"}


def parse_bool(value: Any) -> bool:
    """Parse flexible booleans used in CSV labels."""
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError("Boolean value cannot be None")

    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def parse_tokenized_trace(raw_trace: str, delimiter: str = "|") -> list[str]:
    """Split a delimited token trace and drop empty tokens."""
    if raw_trace is None:
        return []
    text = str(raw_trace).strip()
    if not text or text.upper() == "MISSING":
        return []
    return [token.strip() for token in text.split(delimiter) if token.strip()]


def load_trace_csv(path: str | Path) -> pd.DataFrame:
    """Load encoded traces CSV and validate required columns."""
    df = pd.read_csv(path)
    df = normalize_id_columns(df)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ID schema to required columns.

    Supported input variants:
    - question_id + trace_id
    - question_id + sample_id (composite key)
    - quesiton_id + sample_id (common typo in question_id)
    """
    work = df.copy()

    if "question_id" not in work.columns and "quesiton_id" in work.columns:
        work = work.rename(columns={"quesiton_id": "question_id"})

    if "question_id" not in work.columns:
        raise ValueError("Missing required question id column: expected 'question_id' or 'quesiton_id'.")

    if "trace_id" not in work.columns:
        if "sample_id" not in work.columns:
            raise ValueError(
                "Missing trace identifier columns: provide 'trace_id' or 'sample_id' "
                "(used with question_id as a composite key)."
            )
        work["trace_id"] = (
            work["question_id"].astype(str).str.strip()
            + ":"
            + work["sample_id"].astype(str).str.strip()
        )

    return work


def preprocess_traces(
    df: pd.DataFrame,
    *,
    delimiter: str = "|",
    deduplicate: bool = False,
    dedupe_per_question: bool = False,
    min_trace_len: int | None = None,
    max_trace_len: int | None = None,
) -> pd.DataFrame:
    """
    Parse traces, normalize labels, and apply optional filtering.

    When dedupe_per_question is True, duplicate (question_id, tokenized_trace) pairs are dropped.
    Otherwise, deduplicate=True drops duplicate tokenized_trace rows globally.
    """
    work = df.copy()
    work = normalize_id_columns(work)
    work["is_correct"] = work["is_correct"].map(parse_bool)
    work["tokens"] = work["tokenized_trace"].map(lambda x: parse_tokenized_trace(str(x), delimiter))
    work["trace_len"] = work["tokens"].map(len)

    if min_trace_len is not None:
        work = work[work["trace_len"] >= min_trace_len]
    if max_trace_len is not None:
        work = work[work["trace_len"] <= max_trace_len]

    if dedupe_per_question:
        work = work.drop_duplicates(subset=["question_id", "tokenized_trace"])
    elif deduplicate:
        work = work.drop_duplicates(subset=["tokenized_trace"])

    work = work.reset_index(drop=True)
    return work


def split_success_failure(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into successful and unsuccessful trace DataFrames."""
    success = df[df["is_correct"]].copy().reset_index(drop=True)
    failure = df[~df["is_correct"]].copy().reset_index(drop=True)
    return success, failure


def traces_by_question(df: pd.DataFrame) -> dict[Any, pd.DataFrame]:
    """Group preprocessed traces by question_id."""
    return {qid: g.copy().reset_index(drop=True) for qid, g in df.groupby("question_id", sort=False)}


def token_frequency(df: pd.DataFrame) -> dict[str, int]:
    """Document-level token frequencies across traces."""
    counts: dict[str, int] = {}
    for tokens in df["tokens"]:
        for token in set(tokens):
            counts[token] = counts.get(token, 0) + 1
    return counts


def summarize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Compute high-level corpus stats."""
    success_count = int(df["is_correct"].sum())
    failure_count = int((~df["is_correct"]).sum())
    avg_len = float(df["trace_len"].mean()) if not df.empty else 0.0

    by_question = (
        df.groupby("question_id")["is_correct"]
        .agg(total="count", success="sum")
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        "num_traces": int(len(df)),
        "num_success": success_count,
        "num_failure": failure_count,
        "success_rate": success_count / len(df) if len(df) else 0.0,
        "avg_trace_len": avg_len,
        "median_trace_len": float(df["trace_len"].median()) if not df.empty else 0.0,
        "num_questions": int(df["question_id"].nunique()),
        "token_doc_frequency": token_frequency(df),
        "question_level_counts": by_question,
    }


def save_cleaned_data(df: pd.DataFrame, path: str | Path) -> None:
    """Save cleaned intermediate dataset."""
    out = df.copy()
    out["tokens"] = out["tokens"].map(lambda toks: "|".join(toks))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write JSON with stable formatting."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
