from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = ROOT / "tokenizer" / "v2" / "tokenized_pilot_traces.csv"
OUTPUT_DIR = Path(__file__).resolve().parent


def is_truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def all_contiguous_subsequences(
    sequence: Sequence[str],
    *,
    min_len: int,
    max_len: int,
) -> Iterable[tuple[str, ...]]:
    capped_max = min(max_len, len(sequence))
    for length in range(min_len, capped_max + 1):
        for start in range(0, len(sequence) - length + 1):
            yield tuple(sequence[start : start + length])


def read_rows(input_path: Path) -> list[dict[str, str]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find tokenized traces file: {input_path}")
    with input_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_motif_stats(
    rows: list[dict[str, str]],
    *,
    min_len: int,
    max_len: int,
) -> list[dict[str, str]]:
    correct_counter: Counter[tuple[str, ...]] = Counter()
    incorrect_counter: Counter[tuple[str, ...]] = Counter()

    for row in rows:
        tokens = [token for token in (row.get("reasoning_trace", "") or "").split("|") if token]
        unique_subseqs = set(
            all_contiguous_subsequences(tokens, min_len=min_len, max_len=max_len)
        )
        if is_truthy(row.get("is_correct", "")):
            correct_counter.update(unique_subseqs)
        else:
            incorrect_counter.update(unique_subseqs)

    motif_stats: list[dict[str, str]] = []
    all_candidates = set(correct_counter) | set(incorrect_counter)
    for motif in all_candidates:
        correct = correct_counter[motif]
        incorrect = incorrect_counter[motif]
        total = correct + incorrect
        motif_stats.append(
            {
                "motif": "|".join(motif),
                "motif_pretty": " -> ".join(motif),
                "length": str(len(motif)),
                "support_total": str(total),
                "support_correct": str(correct),
                "support_incorrect": str(incorrect),
                "p_correct": f"{(correct / total):.6f}" if total else "0.000000",
                "p_incorrect": f"{(incorrect / total):.6f}" if total else "0.000000",
                "enrichment_correct": f"{(correct + 1) / (incorrect + 1):.6f}",
                "enrichment_incorrect": f"{(incorrect + 1) / (correct + 1):.6f}",
                "dominant_label": (
                    "correct"
                    if correct > incorrect
                    else "incorrect"
                    if incorrect > correct
                    else "tie"
                ),
            }
        )

    motif_stats.sort(
        key=lambda row: (
            int(row["support_total"]),
            float(row["enrichment_correct"]),
            int(row["length"]),
        ),
        reverse=True,
    )
    return motif_stats


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine contiguous motif subsequences from tokenized pilot traces."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to tokenized pilot traces. Defaults to {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=2,
        help="Minimum motif subsequence length.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=4,
        help="Maximum motif subsequence length.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=1,
        help="Minimum total support for a motif to be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_len < 1:
        raise ValueError("--min-len must be >= 1")
    if args.max_len < args.min_len:
        raise ValueError("--max-len must be >= --min-len")
    if args.min_support < 1:
        raise ValueError("--min-support must be >= 1")

    rows = read_rows(args.input)
    motif_stats = build_motif_stats(rows, min_len=args.min_len, max_len=args.max_len)
    filtered = [row for row in motif_stats if int(row["support_total"]) >= args.min_support]

    fieldnames = [
        "motif",
        "motif_pretty",
        "length",
        "support_total",
        "support_correct",
        "support_incorrect",
        "p_correct",
        "p_incorrect",
        "enrichment_correct",
        "enrichment_incorrect",
        "dominant_label",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_path = OUTPUT_DIR / "all_motif_subsequences.csv"
    correct_path = OUTPUT_DIR / "correct_motif_subsequences.csv"
    incorrect_path = OUTPUT_DIR / "incorrect_motif_subsequences.csv"

    correct_rows = [row for row in filtered if row["dominant_label"] == "correct"]
    incorrect_rows = [row for row in filtered if row["dominant_label"] == "incorrect"]

    write_csv(all_path, filtered, fieldnames)
    write_csv(correct_path, correct_rows, fieldnames)
    write_csv(incorrect_path, incorrect_rows, fieldnames)

    print(f"Read {len(rows)} traces from {args.input}")
    print(f"Wrote {len(filtered)} motifs to {all_path}")
    print(f"Wrote {len(correct_rows)} correct-leaning motifs to {correct_path}")
    print(f"Wrote {len(incorrect_rows)} incorrect-leaning motifs to {incorrect_path}")


if __name__ == "__main__":
    main()
