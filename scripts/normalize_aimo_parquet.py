#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize an AI-MO parquet benchmark into the local question/answer JSON format."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input parquet file.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON benchmark file.")
    parser.add_argument("--source-dataset", required=True, help="Source dataset label to attach to each row.")
    parser.add_argument("--subject", default=None, help="Optional subject label.")
    parser.add_argument("--level", type=int, default=None, help="Optional numeric difficulty level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    frame = pd.read_parquet(args.input)
    required = {"problem", "answer"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Input parquet is missing required columns: {', '.join(missing)}")

    rows: list[dict[str, object]] = []
    for index, raw_row in frame.iterrows():
        question = str(raw_row["problem"]).strip()
        answer = str(raw_row["answer"]).strip()
        if not question or not answer:
            continue

        source_id = raw_row.get("url") if "url" in frame.columns else raw_row.get("id", index)
        row: dict[str, object] = {
            "question_id": len(rows),
            "question": question,
            "answer": answer,
            "source_dataset": args.source_dataset,
            "source_id": str(source_id),
        }
        if args.subject is not None:
            row["subject"] = args.subject
        if args.level is not None:
            row["level"] = args.level
        if "solution" in frame.columns and not pd.isna(raw_row["solution"]):
            row["solution"] = str(raw_row["solution"])
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Normalized {len(rows)} rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
