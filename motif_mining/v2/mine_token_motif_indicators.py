from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = BASE_DIR / "sample_tokenized_traces.csv"
DEFAULT_OUTPUT_CSV = BASE_DIR / "motif_indicator_stats.csv"
DEFAULT_OUTPUT_MD = BASE_DIR / "motif_indicator_report.md"

TRUTHY_VALUES = {"true", "1", "yes", "y", "t"}
FALSY_VALUES = {"false", "0", "no", "n", "f"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine motif indicators from tokenized traces and estimate how strongly motifs "
            "indicate success or failure."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to CSV containing tokenized traces and correctness labels.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path for motif metrics (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help=f"Output markdown report path (default: {DEFAULT_OUTPUT_MD}).",
    )
    parser.add_argument(
        "--trace-column",
        default="tokenized_trace",
        help="Column name for whitespace-delimited tokenized traces (default: tokenized_trace).",
    )
    parser.add_argument(
        "--label-column",
        default="is_correct",
        help="Column name for correctness labels (default: is_correct).",
    )
    parser.add_argument(
        "--min-motif-len",
        type=int,
        default=1,
        help="Minimum contiguous motif length in tokens (default: 1).",
    )
    parser.add_argument(
        "--max-motif-len",
        type=int,
        default=3,
        help="Maximum contiguous motif length in tokens (default: 3).",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=3,
        help="Minimum number of traces containing a motif to include it (default: 3).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Top motifs to display per direction in markdown report (default: 25).",
    )
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in TRUTHY_VALUES:
        return True
    if normalized in FALSY_VALUES:
        return False
    raise ValueError(f"Unrecognized boolean value: {value!r}")


def tokenize_trace(raw_trace: str) -> list[str]:
    cleaned = raw_trace.strip()
    if not cleaned or cleaned.upper() == "MISSING":
        return []
    return [token for token in cleaned.split() if token]


def unique_contiguous_motifs(tokens: list[str], min_len: int, max_len: int) -> set[str]:
    motifs: set[str] = set()
    if not tokens:
        return motifs

    n_tokens = len(tokens)
    local_max = min(max_len, n_tokens)
    for motif_len in range(max(1, min_len), local_max + 1):
        for start in range(0, n_tokens - motif_len + 1):
            motifs.add(" ".join(tokens[start : start + motif_len]))
    return motifs


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def fmt_float(value: float) -> str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.4f}"


def load_rows(input_path: Path, trace_col: str, label_col: str) -> list[tuple[bool, list[str]]]:
    rows: list[tuple[bool, list[str]]] = []
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Input file has no header: {input_path}")

        missing_cols = [col for col in (trace_col, label_col) if col not in reader.fieldnames]
        if missing_cols:
            raise ValueError(
                f"Missing required columns {missing_cols} in input header {reader.fieldnames}."
            )

        for row in reader:
            label_raw = row[label_col]
            trace_raw = row[trace_col]
            if label_raw is None or trace_raw is None:
                continue
            try:
                is_correct = parse_bool(label_raw)
            except ValueError:
                continue
            rows.append((is_correct, tokenize_trace(trace_raw)))

    if not rows:
        raise ValueError("No valid rows were parsed from input.")
    return rows


def compute_metrics(
    rows: list[tuple[bool, list[str]]],
    min_len: int,
    max_len: int,
    min_support: int,
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
    total_traces = len(rows)
    total_correct = sum(1 for is_correct, _ in rows if is_correct)
    total_incorrect = total_traces - total_correct

    baseline_correct_rate = safe_div(total_correct, total_traces)
    baseline_incorrect_rate = safe_div(total_incorrect, total_traces)

    motif_correct_counts: Counter[str] = Counter()
    motif_incorrect_counts: Counter[str] = Counter()

    for is_correct, tokens in rows:
        motifs = unique_contiguous_motifs(tokens, min_len, max_len)
        if is_correct:
            motif_correct_counts.update(motifs)
        else:
            motif_incorrect_counts.update(motifs)

    metrics_rows: list[dict[str, float | int | str]] = []
    all_motifs = set(motif_correct_counts) | set(motif_incorrect_counts)

    for motif in all_motifs:
        correct_with = motif_correct_counts[motif]
        incorrect_with = motif_incorrect_counts[motif]
        total_with = correct_with + incorrect_with

        if total_with < min_support:
            continue

        p_correct_given_motif = safe_div(correct_with, total_with)
        p_incorrect_given_motif = safe_div(incorrect_with, total_with)

        # Lift relative to baseline class rate.
        lift_correct = safe_div(p_correct_given_motif, baseline_correct_rate)
        lift_incorrect = safe_div(p_incorrect_given_motif, baseline_incorrect_rate)

        # Signed score: positive favors correctness, negative favors failure.
        delta_correct_vs_baseline = p_correct_given_motif - baseline_correct_rate

        # Log odds ratio of motif prevalence in correct vs incorrect traces.
        # Haldane-Anscombe smoothing avoids divide-by-zero instability.
        odds_correct = (correct_with + 0.5) / ((total_correct - correct_with) + 0.5)
        odds_incorrect = (incorrect_with + 0.5) / ((total_incorrect - incorrect_with) + 0.5)
        log_odds_ratio = math.log(odds_correct / odds_incorrect)

        metrics_rows.append(
            {
                "motif": motif,
                "motif_len": len(motif.split()),
                "support_total": total_with,
                "support_correct": correct_with,
                "support_incorrect": incorrect_with,
                "support_rate": safe_div(total_with, total_traces),
                "p_correct_given_motif": p_correct_given_motif,
                "p_incorrect_given_motif": p_incorrect_given_motif,
                "baseline_correct_rate": baseline_correct_rate,
                "baseline_incorrect_rate": baseline_incorrect_rate,
                "delta_correct_vs_baseline": delta_correct_vs_baseline,
                "lift_correct": lift_correct,
                "lift_incorrect": lift_incorrect,
                "log_odds_ratio": log_odds_ratio,
                "indicator_direction": (
                    "success" if delta_correct_vs_baseline >= 0 else "failure"
                ),
                "indicator_strength": abs(delta_correct_vs_baseline),
            }
        )

    metrics_rows.sort(
        key=lambda row: (
            abs(float(row["delta_correct_vs_baseline"])),
            float(row["support_total"]),
        ),
        reverse=True,
    )

    baseline = {
        "total_traces": total_traces,
        "total_correct": total_correct,
        "total_incorrect": total_incorrect,
        "baseline_correct_rate": baseline_correct_rate,
        "baseline_incorrect_rate": baseline_incorrect_rate,
    }

    return baseline, metrics_rows


def write_metrics_csv(
    output_path: Path,
    baseline: dict[str, float | int],
    metrics_rows: Iterable[dict[str, float | int | str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "motif",
        "motif_len",
        "support_total",
        "support_correct",
        "support_incorrect",
        "support_rate",
        "p_correct_given_motif",
        "p_incorrect_given_motif",
        "baseline_correct_rate",
        "baseline_incorrect_rate",
        "delta_correct_vs_baseline",
        "lift_correct",
        "lift_incorrect",
        "log_odds_ratio",
        "indicator_direction",
        "indicator_strength",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "motif": "__BASELINE__",
                "motif_len": 0,
                "support_total": baseline["total_traces"],
                "support_correct": baseline["total_correct"],
                "support_incorrect": baseline["total_incorrect"],
                "support_rate": 1.0,
                "p_correct_given_motif": baseline["baseline_correct_rate"],
                "p_incorrect_given_motif": baseline["baseline_incorrect_rate"],
                "baseline_correct_rate": baseline["baseline_correct_rate"],
                "baseline_incorrect_rate": baseline["baseline_incorrect_rate"],
                "delta_correct_vs_baseline": 0.0,
                "lift_correct": 1.0,
                "lift_incorrect": 1.0,
                "log_odds_ratio": 0.0,
                "indicator_direction": "baseline",
                "indicator_strength": 0.0,
            }
        )
        for row in metrics_rows:
            writer.writerow(row)


def make_table(rows: list[dict[str, float | int | str]], limit: int) -> str:
    if not rows:
        return "No motifs passed filters."

    lines = [
        "| Motif | Support | P(correct given motif) | P(incorrect given motif) | Delta vs baseline | Lift(correct) | Lift(incorrect) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:limit]:
        lines.append(
            "| "
            + f"`{row['motif']}` | {row['support_total']} | {fmt_pct(float(row['p_correct_given_motif']))}"
            + f" | {fmt_pct(float(row['p_incorrect_given_motif']))} | "
            + f"{fmt_pct(float(row['delta_correct_vs_baseline']))} | "
            + f"{fmt_float(float(row['lift_correct']))} | {fmt_float(float(row['lift_incorrect']))} |"
        )
    return "\n".join(lines)


def write_markdown_report(
    output_path: Path,
    input_path: Path,
    baseline: dict[str, float | int],
    metrics_rows: list[dict[str, float | int | str]],
    top_k: int,
    min_support: int,
    min_len: int,
    max_len: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success_rows = [
        row
        for row in metrics_rows
        if str(row["indicator_direction"]) == "success"
    ]
    success_rows.sort(
        key=lambda row: (
            float(row["delta_correct_vs_baseline"]),
            float(row["support_total"]),
        ),
        reverse=True,
    )

    failure_rows = [
        row
        for row in metrics_rows
        if str(row["indicator_direction"]) == "failure"
    ]
    failure_rows.sort(
        key=lambda row: (
            float(row["p_incorrect_given_motif"]),
            float(row["support_total"]),
        ),
        reverse=True,
    )

    single_token_success_rows = [row for row in success_rows if int(row["motif_len"]) == 1]
    single_token_failure_rows = [row for row in failure_rows if int(row["motif_len"]) == 1]

    report_lines = [
        "# Motif Indicator Report",
        "",
        f"- Input: `{input_path}`",
        f"- Total traces: {baseline['total_traces']}",
        f"- Correct traces: {baseline['total_correct']} ({fmt_pct(float(baseline['baseline_correct_rate']))})",
        f"- Incorrect traces: {baseline['total_incorrect']} ({fmt_pct(float(baseline['baseline_incorrect_rate']))})",
        f"- Motif settings: contiguous n-grams, length {min_len}-{max_len}, min support {min_support}",
        "",
        "## Strongest Success Indicators",
        "",
        make_table(success_rows, top_k),
        "",
        "## Strongest Failure Indicators",
        "",
        make_table(failure_rows, top_k),
        "",
        "## Strongest Single-Token Success Indicators",
        "",
        make_table(single_token_success_rows, top_k),
        "",
        "## Strongest Single-Token Failure Indicators",
        "",
        make_table(single_token_failure_rows, top_k),
        "",
        "## Notes",
        "",
        "- `Delta vs baseline` is `P(correct given motif) - baseline_correct_rate`.",
        "- Positive delta implies success-associated motif; negative delta implies failure-associated motif.",
        "- Rare motifs can look extreme; use support counts to judge reliability.",
    ]

    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.min_motif_len <= 0 or args.max_motif_len <= 0:
        raise ValueError("Motif lengths must be positive integers.")
    if args.min_motif_len > args.max_motif_len:
        raise ValueError("--min-motif-len cannot be greater than --max-motif-len.")
    if args.min_support <= 0:
        raise ValueError("--min-support must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    rows = load_rows(args.input, args.trace_column, args.label_column)
    baseline, metrics_rows = compute_metrics(
        rows,
        min_len=args.min_motif_len,
        max_len=args.max_motif_len,
        min_support=args.min_support,
    )

    write_metrics_csv(args.output_csv, baseline, metrics_rows)
    write_markdown_report(
        args.output_md,
        input_path=args.input,
        baseline=baseline,
        metrics_rows=metrics_rows,
        top_k=args.top_k,
        min_support=args.min_support,
        min_len=args.min_motif_len,
        max_len=args.max_motif_len,
    )

    print(f"Wrote motif metrics CSV to: {args.output_csv}")
    print(f"Wrote markdown report to: {args.output_md}")


if __name__ == "__main__":
    main()
