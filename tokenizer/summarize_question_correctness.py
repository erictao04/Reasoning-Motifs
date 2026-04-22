#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QuestionStats:
    question_id: str
    question: str
    total: int = 0
    correct: int = 0

    @property
    def percent_correct(self) -> float:
        if self.total == 0:
            return 0.0
        return 100.0 * self.correct / self.total


def parse_is_correct(value: str) -> bool:
    normalized = (value or "").strip().lower()
    return normalized in {"1", "true", "t", "yes", "y"}


def summarize(input_csv: Path) -> list[QuestionStats]:
    by_question: dict[str, QuestionStats] = {}

    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {"question_id", "question", "is_correct"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns: {missing_list}")

        for row in reader:
            qid = (row.get("question_id") or "").strip()
            question = (row.get("question") or "").strip()
            key = qid or question

            if key not in by_question:
                by_question[key] = QuestionStats(question_id=qid, question=question)

            stats = by_question[key]
            stats.total += 1
            if parse_is_correct(row.get("is_correct", "")):
                stats.correct += 1

    return sorted(
        by_question.values(),
        key=lambda s: (
            int(s.question_id) if s.question_id.isdigit() else float("inf"),
            s.question_id,
            s.question,
        ),
    )


def write_csv(stats: list[QuestionStats], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id",
            "correct",
            "total",
            "percent_correct",
        ])
        for s in stats:
            writer.writerow([
                s.question_id,
                s.correct,
                s.total,
                f"{s.percent_correct:.2f}",
            ])


def print_table(stats: list[QuestionStats]) -> None:
    print("question_id,correct,total,percent_correct")
    for s in stats:
        print(
            f"{s.question_id},{s.correct},{s.total},{s.percent_correct:.2f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize correctness percentage for each question in a traces CSV."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to input CSV (must include question_id, question, is_correct).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path for the summary.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    stats = summarize(args.input_csv)

    if args.output_csv:
        write_csv(stats, args.output_csv)
        print(f"Wrote summary to: {args.output_csv}")
    else:
        print_table(stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
