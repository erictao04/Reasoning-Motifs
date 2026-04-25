#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine multiple benchmark files into one JSON benchmark.")
    parser.add_argument("--input", type=Path, action="append", required=True, help="Input benchmark file. Repeatable.")
    parser.add_argument("--output", type=Path, required=True, help="Output combined JSON benchmark path.")
    parser.add_argument(
        "--dedupe-question",
        action="store_true",
        help="Keep only the first row for duplicate normalized question text.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"JSON benchmark must be a list: {path}")
        return [dict(row) for row in data]
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]
    raise ValueError(f"Unsupported benchmark format: {path}")


def main() -> None:
    args = parse_args()
    combined: list[dict[str, object]] = []
    seen_questions: set[str] = set()
    for input_path in args.input:
        rows = load_rows(input_path.resolve())
        source_name = input_path.stem
        for row in rows:
            row = dict(row)
            row.setdefault("source_dataset", source_name)
            if args.dedupe_question:
                question_key = normalize_question(str(row.get("question", "")))
                if question_key in seen_questions:
                    continue
                seen_questions.add(question_key)
            combined.append(row)

    for index, row in enumerate(combined):
        row["question_id"] = index

    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(combined, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Combined {len(combined)} questions into {args.output.resolve()}")


def normalize_question(question: str) -> str:
    return " ".join(question.split()).lower()


if __name__ == "__main__":
    main()
