from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
TRACES_DIR = ROOT / "data" / "traces"
ENCODED_TRACES_DIR = ROOT / "data" / "encoded_traces"


MOTIF_RULES: "OrderedDict[str, str]" = OrderedDict(
    [
        ("instantiate", "instantiate"),
        ("rewrite", "rewrite"),
        ("compute", "compute"),
        ("check the constraint", "check_constraint"),
        ("check", "check_constraint"),
        ("backtrack", "backtrack"),
        ("conclude", "conclude"),
        ("read problem", "read_problem"),
        ("generic reasoning strategy", "generic_reasoning"),
    ]
)


def encode_trace(reasoning_trace: str) -> List[str]:
    motifs: List[str] = []
    for raw_line in reasoning_trace.splitlines():
        line = raw_line.strip().lower()
        if not line:
            continue
        matched = False
        for key, motif in MOTIF_RULES.items():
            if key in line:
                motifs.append(motif)
                matched = True
                break
        if not matched:
            motifs.append("other")
    return motifs


def read_traces(run_index: int) -> List[Dict[str, str]]:
    input_path = TRACES_DIR / f"traces_{run_index}.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find trace file: {input_path}")
    with input_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_encoded_traces(run_index: int, rows: List[Dict[str, str]]) -> Path:
    ENCODED_TRACES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ENCODED_TRACES_DIR / f"encoded_traces_{run_index}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "question",
                "motif_sequence",
                "is_correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode raw reasoning traces into motif sequences.")
    parser.add_argument("index", type=int, help="Run index of the trace file to encode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    traces = read_traces(args.index)
    encoded_rows: List[Dict[str, str]] = []

    for row in traces:
        motifs = encode_trace(row["reasoning_trace"])
        encoded_rows.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "motif_sequence": "|".join(motifs),
                "is_correct": row["is_correct"],
            }
        )

    output_path = save_encoded_traces(args.index, encoded_rows)
    print(f"Saved {len(encoded_rows)} encoded traces to {output_path}")


if __name__ == "__main__":
    main()
