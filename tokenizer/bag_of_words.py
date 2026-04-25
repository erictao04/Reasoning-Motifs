#!/usr/bin/env python3
"""Build a bag-of-words frequency table from tokenized trace CSV files.

Examples:
  python tokenizer/bag_of_words.py \
    --input tokenizer/clean_expanded_pool_s100_seed73_qwen25_7b_hot30_tokenized_gpt-oss-120b_0.csv

  python tokenizer/bag_of_words.py \
    --input tokenizer/clean_expanded_pool_s100_seed73_qwen25_7b_hot30_tokenized_gpt-oss-120b_0.csv \
    --output tokenizer/token_bow.csv \
    --token-column tokenized_trace
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

DEFAULT_TOKEN_COLUMN_CANDIDATES = (
    "tokenized_trace",
    "tokens",
    "tokenized",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a bag-of-words over tokens in a tokenizer-style CSV and "
            "write token frequencies to a CSV file."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input CSV (e.g., tokenizer/clean_data_...csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to output CSV. Default: <input_stem>_bow.csv in the same directory."
        ),
    )
    parser.add_argument(
        "--token-column",
        default=None,
        help=(
            "Column containing token sequences. If omitted, auto-detects from: "
            f"{', '.join(DEFAULT_TOKEN_COLUMN_CANDIDATES)}"
        ),
    )
    return parser.parse_args()


def detect_token_column(fieldnames: list[str], user_choice: str | None) -> str:
    if user_choice:
        if user_choice not in fieldnames:
            cols = ", ".join(fieldnames)
            raise ValueError(
                f"Requested token column '{user_choice}' not found. Available columns: {cols}"
            )
        return user_choice

    for candidate in DEFAULT_TOKEN_COLUMN_CANDIDATES:
        if candidate in fieldnames:
            return candidate

    cols = ", ".join(fieldnames)
    choices = ", ".join(DEFAULT_TOKEN_COLUMN_CANDIDATES)
    raise ValueError(
        f"Could not auto-detect token column. Tried: {choices}. Available columns: {cols}"
    )


def parse_token_cell(value: str) -> list[str]:
    text = (value or "").strip()
    if not text or text.upper() == "MISSING":
        return []

    # Support JSON arrays in case tokens are stored as ["a", "b"].
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(tok).strip() for tok in parsed if str(tok).strip()]
        except json.JSONDecodeError:
            pass

    # Otherwise split on common separators and whitespace.
    parts = re.split(r"[\s,;|]+", text)
    return [p for p in parts if p]


def build_bow(input_csv: Path, token_column: str) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    row_count = 0

    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header: {input_csv}")

        if token_column not in reader.fieldnames:
            cols = ", ".join(reader.fieldnames)
            raise ValueError(
                f"Token column '{token_column}' not found in input CSV. Columns: {cols}"
            )

        for row in reader:
            row_count += 1
            tokens = parse_token_cell(row.get(token_column, ""))
            counts.update(tokens)

    return counts, row_count


def write_bow(output_csv: Path, counts: Counter[str], row_count: int) -> None:
    total_tokens = sum(counts.values())

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("token", "count", "relative_frequency", "input_rows"),
        )
        writer.writeheader()

        for token, count in counts.most_common():
            rel = (count / total_tokens) if total_tokens else 0.0
            writer.writerow(
                {
                    "token": token,
                    "count": count,
                    "relative_frequency": f"{rel:.8f}",
                    "input_rows": row_count,
                }
            )


def main() -> None:
    args = parse_args()

    input_csv = args.input.resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if args.output is None:
        output_csv = input_csv.with_name(f"{input_csv.stem}_bow.csv")
    else:
        output_csv = args.output.resolve()

    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

    token_column = detect_token_column(fieldnames, args.token_column)
    counts, row_count = build_bow(input_csv, token_column)
    write_bow(output_csv, counts, row_count)

    print(f"Input: {input_csv}")
    print(f"Token column: {token_column}")
    print(f"Rows processed: {row_count}")
    print(f"Unique tokens: {len(counts)}")
    print(f"Total tokens: {sum(counts.values())}")
    print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    main()
