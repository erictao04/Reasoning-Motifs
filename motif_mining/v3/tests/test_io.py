from __future__ import annotations

import pandas as pd

from analysis.io_utils import normalize_id_columns, parse_tokenized_trace, preprocess_traces


def test_parse_tokenized_trace() -> None:
    assert parse_tokenized_trace("analyze|compute|conclude") == ["analyze", "compute", "conclude"]
    assert parse_tokenized_trace("a||b|") == ["a", "b"]
    assert parse_tokenized_trace("MISSING") == []


def test_preprocess_filters_and_bool() -> None:
    df = pd.DataFrame(
        {
            "question_id": [1, 1, 2],
            "sample_id": [0, 1, 0],
            "tokenized_trace": ["a|b|c", "a", "a|b"],
            "is_correct": ["True", "0", "1"],
        }
    )
    out = preprocess_traces(df, min_trace_len=2)
    assert len(out) == 2
    assert out["is_correct"].tolist() == [True, True]
    assert "trace_id" in out.columns


def test_normalize_id_columns_typo_question_id() -> None:
    df = pd.DataFrame(
        {
            "quesiton_id": [45],
            "sample_id": [9],
            "tokenized_trace": ["analyze|compute"],
            "is_correct": [1],
        }
    )
    out = normalize_id_columns(df)
    assert "question_id" in out.columns
    assert out.loc[0, "trace_id"] == "45:9"
