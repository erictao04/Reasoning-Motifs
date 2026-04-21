#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def _normalize(value: str) -> str:
    return (value or "").strip()


def _read_row_by_key(csv_path: Path, question_id: str, sample_id: str, attempt_index: str):
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a markdown file with question, reasoning_trace, and tokenized_trace "
            "for a specific question_id/sample_id/attempt_index."
        )
    )
    parser.add_argument("--question-id", required=True, help="Question ID (e.g., 22)")
    parser.add_argument("--sample-id", required=True, help="Sample ID (e.g., 3)")
    parser.add_argument("--attempt-index", required=True, help="Attempt index (e.g., 3)")
    parser.add_argument(
        "--source-csv",
        default="tokenizer/question_independent_model/math_motif_pilot_v2_hot30.csv",
        help="CSV containing question and reasoning_trace columns.",
    )
    parser.add_argument(
        "--tokenized-csv",
        default=None,
        help=(
            "CSV containing tokenized_trace. "
            "Defaults to tokenizer/question_independent_model/question_<question-id>_tokenized_traces.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output markdown path. Defaults to "
            "tokenizer/question_independent_model/question_<qid>_sample_<sid>_attempt_<aid>.md"
        ),
    )
    args = parser.parse_args()

    question_id = _normalize(args.question_id)
    sample_id = _normalize(args.sample_id)
    attempt_index = _normalize(args.attempt_index)

    source_csv = Path(args.source_csv)
    tokenized_csv = (
        Path(args.tokenized_csv)
        if args.tokenized_csv
        else Path(
            f"tokenizer/question_independent_model/question_{question_id}_tokenized_traces.csv"
        )
    )
    output_md = (
        Path(args.output)
        if args.output
        else Path(
            "tokenizer/question_independent_model/"
            f"question_{question_id}_sample_{sample_id}_attempt_{attempt_index}.md"
        )
    )

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
    reasoning_trace = source_row.get("reasoning_trace", "")
    tokenized_trace = token_row.get("tokenized_trace", "")

    markdown = (
        f"# Trace Report\n\n"
        f"- question_id: `{question_id}`\n"
        f"- sample_id: `{sample_id}`\n"
        f"- attempt_index: `{attempt_index}`\n\n"
        f"## Question\n\n"
        f"{question}\n\n"
        f"## Correct Answer\n\n"
        f"{correct_answer}\n\n"
        f"## Reasoning Trace\n\n"
        f"{reasoning_trace}\n\n"
        f"## Tokenized Trace\n\n"
        f"{tokenized_trace}\n"
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")
    print(f"Wrote markdown report: {output_md}")


if __name__ == "__main__":
    main()
