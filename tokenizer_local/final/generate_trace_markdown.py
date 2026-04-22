#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_CSV = BASE_DIR / "math_motif_pilot_v6_qwen25_7b_hot30.csv"


def _normalize(value: str | None) -> str:
    return (value or "").strip()


def _read_row_by_key(
    csv_path: Path,
    question_id: str,
    sample_id: str,
    attempt_index: str,
) -> dict[str, str] | None:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                _normalize(row.get("question_id")) == question_id
                and _normalize(row.get("sample_id")) == sample_id
                and _normalize(row.get("attempt_index")) == attempt_index
            ):
                return row
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a markdown report containing question, correct answer, "
            "reasoning trace, and tokenized trace for a single row key."
        )
    )
    parser.add_argument("--question-id", required=True, help="Question ID (e.g., 1)")
    parser.add_argument("--sample-id", required=True, help="Sample ID (e.g., 0)")
    parser.add_argument(
        "--attempt-index", required=True, help="Attempt index (e.g., 0)"
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=DEFAULT_SOURCE_CSV,
        help=f"CSV containing question/reasoning_trace (default: {DEFAULT_SOURCE_CSV})",
    )
    parser.add_argument(
        "--tokenized-csv",
        type=Path,
        default=None,
        help=(
            "CSV containing tokenized_trace. Defaults to "
            "tokenizer/final/question_<question-id>_tokenized_traces.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output markdown path. Defaults to "
            "tokenizer/final/question_<qid>_sample_<sid>_attempt_<aid>.md"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    question_id = _normalize(args.question_id)
    sample_id = _normalize(args.sample_id)
    attempt_index = _normalize(args.attempt_index)

    source_csv = args.source_csv
    tokenized_csv = (
        args.tokenized_csv
        if args.tokenized_csv is not None
        else BASE_DIR / f"question_{question_id}_tokenized_traces.csv"
    )
    output_md = (
        args.output
        if args.output is not None
        else BASE_DIR / f"question_{question_id}_sample_{sample_id}_attempt_{attempt_index}.md"
    )

    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")
    if not tokenized_csv.exists():
        raise FileNotFoundError(f"Tokenized CSV not found: {tokenized_csv}")

    source_row = _read_row_by_key(source_csv, question_id, sample_id, attempt_index)
    if source_row is None:
        raise SystemExit(
            f"No matching row in source CSV: {source_csv} "
            f"for key ({question_id}, {sample_id}, {attempt_index})"
        )

    token_row = _read_row_by_key(tokenized_csv, question_id, sample_id, attempt_index)
    if token_row is None:
        raise SystemExit(
            f"No matching row in tokenized CSV: {tokenized_csv} "
            f"for key ({question_id}, {sample_id}, {attempt_index})"
        )

    question = source_row.get("question", "")
    correct_answer = source_row.get("gold_answer", "")
    predicted_answer = source_row.get("predicted_answer", "")
    reasoning_trace = source_row.get("reasoning_trace", "")
    tokenized_trace = token_row.get("tokenized_trace", "")

    markdown = (
        "# Trace Report\n\n"
        f"- question_id: `{question_id}`\n"
        f"- sample_id: `{sample_id}`\n"
        f"- attempt_index: `{attempt_index}`\n\n"
        "## Question\n\n"
        f"{question}\n\n"
        "## Correct Answer\n\n"
        f"{correct_answer}\n\n"
        "## Predicted Answer\n\n"
        f"{predicted_answer}\n\n"
        "## Reasoning Trace\n\n"
        f"{reasoning_trace}\n\n"
        "## Tokenized Trace\n\n"
        f"{tokenized_trace}\n"
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")
    print(f"Wrote markdown report: {output_md}")


if __name__ == "__main__":
    main()
