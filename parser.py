from __future__ import annotations

import argparse
import csv
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
ENCODED_TRACES_DIR = ROOT / "data" / "encoded_traces"


def read_encoded_traces(run_index: int) -> List[dict[str, str]]:
    input_path = ENCODED_TRACES_DIR / f"encoded_traces_{run_index}.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find encoded trace file: {input_path}")
    with input_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def all_contiguous_subsequences(sequence: Sequence[str], min_len: int = 2, max_len: int = 4) -> Iterable[Tuple[str, ...]]:
    max_len = min(max_len, len(sequence))
    for length in range(min_len, max_len + 1):
        for start in range(0, len(sequence) - length + 1):
            yield tuple(sequence[start : start + length])


def is_truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def learn_good_motifs(rows: List[dict[str, str]], min_support: int = 1) -> List[tuple[Tuple[str, ...], int, int, float]]:
    good_counter: Counter[Tuple[str, ...]] = Counter()
    bad_counter: Counter[Tuple[str, ...]] = Counter()

    for row in rows:
        sequence = [token for token in row["motif_sequence"].split("|") if token]
        unique_subsequences = set(all_contiguous_subsequences(sequence))
        target_counter = good_counter if is_truthy(row["is_correct"]) else bad_counter
        target_counter.update(unique_subsequences)

    scored: List[tuple[Tuple[str, ...], int, int, float]] = []
    all_candidates = set(good_counter) | set(bad_counter)
    for motif in all_candidates:
        good = good_counter[motif]
        bad = bad_counter[motif]
        total = good + bad
        if total < min_support:
            continue
        score = (good + 1) / (bad + 1)
        if good > bad:
            scored.append((motif, good, bad, score))

    scored.sort(key=lambda x: (x[3], x[1], len(x[0])), reverse=True)
    return scored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn good motif sequences from encoded traces.")
    parser.add_argument("index", type=int, help="Run index of the encoded trace file to parse.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of motifs to print.")
    parser.add_argument("--min-support", type=int, default=1, help="Minimum total support for a motif.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_encoded_traces(args.index)
    motifs = learn_good_motifs(rows, min_support=args.min_support)

    if not motifs:
        print("No good motifs found.")
        return

    print("Top good motif sequences:")
    for motif, good, bad, score in motifs[: args.top_k]:
        pretty = " -> ".join(motif)
        print(f"{pretty} | good={good}, bad={bad}, enrichment={score:.2f}")


if __name__ == "__main__":
    main()
