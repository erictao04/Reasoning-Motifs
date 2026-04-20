#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a sampled reasoning-trace dataset and emit summary artifacts."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input trajectory CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where summary and per-question outputs will be written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top questions to keep in the summary for mixed/hard/easy slices.",
    )
    return parser.parse_args()


def is_truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def analyze_rows(rows: list[dict[str, str]], *, top_k: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("Input dataset is empty.")

    by_question: dict[int, list[dict[str, str]]] = defaultdict(list)
    trace_word_counts: list[int] = []
    trace_line_counts: list[int] = []
    trace_char_counts: list[int] = []

    total_correct = 0
    total_wrong = 0

    for row in rows:
        question_id = int(row["question_id"])
        by_question[question_id].append(row)

        trace = row.get("reasoning_trace", "") or ""
        trace_word_counts.append(len(trace.split()))
        trace_line_counts.append(len([line for line in trace.splitlines() if line.strip()]))
        trace_char_counts.append(len(trace))

        if is_truthy(row.get("is_correct", "")):
            total_correct += 1
        else:
            total_wrong += 1

    question_rows: list[dict[str, Any]] = []
    for question_id in sorted(by_question):
        group = by_question[question_id]
        total_traces = len(group)
        right_traces = sum(1 for row in group if is_truthy(row.get("is_correct", "")))
        wrong_traces = total_traces - right_traces
        predicted_answers = [normalize_text(row.get("predicted_answer", "")) for row in group]
        predicted_counter = Counter(answer for answer in predicted_answers if answer)
        majority_answer, majority_count = ("", 0)
        if predicted_counter:
            majority_answer, majority_count = predicted_counter.most_common(1)[0]

        trace_word_counts_question = [len((row.get("reasoning_trace", "") or "").split()) for row in group]
        trace_line_counts_question = [
            len([line for line in (row.get("reasoning_trace", "") or "").splitlines() if line.strip()])
            for row in group
        ]

        question_rows.append(
            {
                "question_id": question_id,
                "question": group[0]["question"],
                "gold_answer": group[0].get("gold_answer", ""),
                "total_traces": total_traces,
                "right_traces": right_traces,
                "wrong_traces": wrong_traces,
                "accuracy": right_traces / total_traces if total_traces else 0.0,
                "has_mixed_outcomes": right_traces > 0 and wrong_traces > 0,
                "distinct_predicted_answers": len(predicted_counter),
                "majority_predicted_answer": majority_answer,
                "majority_answer_count": majority_count,
                "avg_trace_words": mean(trace_word_counts_question),
                "avg_trace_lines": mean(trace_line_counts_question),
            }
        )

    mixed_questions = [row for row in question_rows if row["has_mixed_outcomes"]]
    easy_questions = [row for row in question_rows if row["accuracy"] == 1.0]
    failed_questions = [row for row in question_rows if row["accuracy"] == 0.0]
    hardest_questions = sorted(
        question_rows,
        key=lambda row: (row["accuracy"], -row["wrong_traces"], -row["total_traces"]),
    )
    most_diverse_answer_questions = sorted(
        question_rows,
        key=lambda row: (row["distinct_predicted_answers"], row["wrong_traces"], row["total_traces"]),
        reverse=True,
    )

    summary = {
        "input_path": None,
        "total_rows": len(rows),
        "total_questions": len(question_rows),
        "total_correct": total_correct,
        "total_wrong": total_wrong,
        "overall_accuracy": total_correct / len(rows),
        "avg_traces_per_question": mean([row["total_traces"] for row in question_rows]),
        "min_traces_per_question": min(row["total_traces"] for row in question_rows),
        "max_traces_per_question": max(row["total_traces"] for row in question_rows),
        "mixed_question_count": len(mixed_questions),
        "all_correct_question_count": len(easy_questions),
        "all_wrong_question_count": len(failed_questions),
        "avg_trace_words": mean(trace_word_counts),
        "avg_trace_lines": mean(trace_line_counts),
        "avg_trace_chars": mean(trace_char_counts),
        "top_mixed_questions": slice_for_summary(
            sorted(mixed_questions, key=lambda row: (-row["wrong_traces"], -row["total_traces"], row["question_id"])),
            top_k,
        ),
        "top_hard_questions": slice_for_summary(hardest_questions, top_k),
        "top_answer_diversity_questions": slice_for_summary(most_diverse_answer_questions, top_k),
    }

    return summary, question_rows


def mean(values: list[int] | list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def slice_for_summary(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    keep_fields = [
        "question_id",
        "total_traces",
        "right_traces",
        "wrong_traces",
        "accuracy",
        "distinct_predicted_answers",
        "majority_predicted_answer",
    ]
    sliced = []
    for row in rows[:top_k]:
        sliced.append({field: row[field] for field in keep_fields if field in row})
    return sliced


def write_question_stats(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "question_id",
        "question",
        "gold_answer",
        "total_traces",
        "right_traces",
        "wrong_traces",
        "accuracy",
        "has_mixed_outcomes",
        "distinct_predicted_answers",
        "majority_predicted_answer",
        "majority_answer_count",
        "avg_trace_words",
        "avg_trace_lines",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    with args.input.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    summary, question_rows = analyze_rows(rows, top_k=args.top_k)
    summary["input_path"] = str(args.input.resolve())

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    question_stats_path = output_dir / "question_stats.csv"
    mixed_only_path = output_dir / "mixed_questions.csv"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_question_stats(question_stats_path, question_rows)
    write_question_stats(mixed_only_path, [row for row in question_rows if row["has_mixed_outcomes"]])

    print(f"Wrote summary to {summary_path}")
    print(f"Wrote per-question stats to {question_stats_path}")
    print(f"Wrote mixed-question slice to {mixed_only_path}")
    print(f"Questions: {summary['total_questions']}")
    print(f"Rows: {summary['total_rows']}")
    print(f"Overall accuracy: {summary['overall_accuracy']:.3f}")
    print(f"Mixed questions: {summary['mixed_question_count']}")


if __name__ == "__main__":
    main()
