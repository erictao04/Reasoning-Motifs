from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT = Path(__file__).resolve().parent / "pilot_traces.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print succeeded/failed outcome for each sample_id and attempt_index "
            "for a given question_id."
        )
    )
    parser.add_argument("question_id", help="Question ID to match (string match).")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path (default: {DEFAULT_INPUT}).",
    )
    return parser.parse_args()


def as_int_if_possible(value: str) -> tuple[int, str]:
    try:
        return 0, int(value)
    except (TypeError, ValueError):
        return 1, value


def status_from_is_correct(value: str) -> str:
    normalized = str(value).strip().lower()
    return "succeeded" if normalized in {"true", "1", "yes"} else "failed"


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with args.input.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {args.input}")

    required_columns = ("question_id", "sample_id", "attempt_index", "is_correct")
    missing_columns = [col for col in required_columns if col not in rows[0]]
    if missing_columns:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing_columns)}. "
            f"Available columns: {', '.join(rows[0].keys())}"
        )

    matches = [
        row
        for row in rows
        if str(row.get("question_id", "")) == str(args.question_id)
    ]

    if not matches:
        print(f"No rows found for question_id={args.question_id}")
        return

    matches.sort(
        key=lambda row: (
            as_int_if_possible(str(row.get("sample_id", ""))),
            as_int_if_possible(str(row.get("attempt_index", ""))),
        )
    )

    for row in matches:
        sample_id = str(row.get("sample_id", ""))
        attempt_index = str(row.get("attempt_index", ""))
        status = status_from_is_correct(str(row.get("is_correct", "")))
        print(f"sample_id={sample_id} attempt_index={attempt_index}: {status}")


if __name__ == "__main__":
    main()
