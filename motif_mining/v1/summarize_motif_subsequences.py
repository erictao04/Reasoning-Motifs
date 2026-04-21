from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path(__file__).resolve().parent / "all_motif_subsequences.csv"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "motif_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize mined motif subsequences and print high-signal statistics."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to all_motif_subsequences.csv (default: {DEFAULT_INPUT_PATH}).",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=10,
        help="Minimum support_total for 'best/worst' motif ranking.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many motifs to print in each top list.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Markdown report output path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    return parser.parse_args()


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def as_int(row: dict[str, str], key: str) -> int:
    return int(row[key])


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def histogram(values: list[float], bins: int = 10) -> list[tuple[float, float, int]]:
    # Bins are [0.0, 0.1), [0.1, 0.2), ... [0.9, 1.0]
    counts = [0 for _ in range(bins)]
    for value in values:
        idx = int(value * bins)
        if idx >= bins:
            idx = bins - 1
        if idx < 0:
            idx = 0
        counts[idx] += 1

    out: list[tuple[float, float, int]] = []
    width = 1.0 / bins
    for i, count in enumerate(counts):
        left = i * width
        right = (i + 1) * width
        out.append((left, right, count))
    return out


def summarize_distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "mean": 0.0, "p75": 0.0, "max": 0.0}

    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p25": percentile(ordered, 0.25),
        "median": percentile(ordered, 0.50),
        "mean": statistics.fmean(values),
        "p75": percentile(ordered, 0.75),
        "max": ordered[-1],
    }


def print_motif_list(title: str, rows: list[dict[str, Any]], top_k: int) -> None:
    print(title)
    for row in rows[:top_k]:
        print(
            f"- {row['motif_pretty']} | support={row['support_total']} | "
            f"p_correct={row['p_correct']:.3f} | p_incorrect={row['p_incorrect']:.3f} | "
            f"enrich_correct={row['enrichment_correct']:.3f} | enrich_incorrect={row['enrichment_incorrect']:.3f}"
        )
    print()


def markdown_motif_section(title: str, rows: list[dict[str, Any]], top_k: int) -> list[str]:
    lines = [f"## {title}", ""]
    for row in rows[:top_k]:
        lines.append(
            f"- **{row['motif_pretty']}**  "
            f"(support={row['support_total']}, "
            f"p_success={row['p_correct']:.3f}, "
            f"p_fail={row['p_incorrect']:.3f}, "
            f"enrich_success={row['enrichment_correct']:.3f}, "
            f"enrich_fail={row['enrichment_incorrect']:.3f})"
        )
    lines.append("")
    return lines


def markdown_histogram_section(title: str, values: list[float], summary: dict[str, float]) -> list[str]:
    lines = [f"## {title}", ""]
    lines.append(
        f"- min={summary['min']:.3f}, p25={summary['p25']:.3f}, "
        f"median={summary['median']:.3f}, mean={summary['mean']:.3f}, "
        f"p75={summary['p75']:.3f}, max={summary['max']:.3f}"
    )
    lines.append("")
    lines.append("| Bin | Count |")
    lines.append("| --- | ---: |")
    for left, right, count in histogram(values, bins=10):
        bracket = "]" if right == 1.0 else ")"
        lines.append(f"| [{left:.1f}, {right:.1f}{bracket} | {count} |")
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with args.input.open("r", newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    if not raw_rows:
        raise ValueError(f"No rows found in {args.input}")

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        rows.append(
            {
                "motif": raw["motif"],
                "motif_pretty": raw["motif_pretty"],
                "length": as_int(raw, "length"),
                "support_total": as_int(raw, "support_total"),
                "support_correct": as_int(raw, "support_correct"),
                "support_incorrect": as_int(raw, "support_incorrect"),
                "p_correct": as_float(raw, "p_correct"),
                "p_incorrect": as_float(raw, "p_incorrect"),
                "enrichment_correct": as_float(raw, "enrichment_correct"),
                "enrichment_incorrect": as_float(raw, "enrichment_incorrect"),
                "dominant_label": raw["dominant_label"],
            }
        )

    filtered = [row for row in rows if row["support_total"] >= args.min_support]
    if not filtered:
        raise ValueError(
            f"No motifs meet --min-support={args.min_support}. Lower the threshold."
        )

    p_correct_values = [row["p_correct"] for row in rows]
    p_incorrect_values = [row["p_incorrect"] for row in rows]
    p_correct_summary = summarize_distribution(p_correct_values)
    p_incorrect_summary = summarize_distribution(p_incorrect_values)

    # Best motifs: high p_correct, high support as tie-breaker.
    best_motifs = sorted(
        filtered,
        key=lambda r: (r["p_correct"], r["support_total"], r["enrichment_correct"]),
        reverse=True,
    )
    # Worst motifs: high p_incorrect, high support as tie-breaker.
    worst_motifs = sorted(
        filtered,
        key=lambda r: (r["p_incorrect"], r["support_total"], r["enrichment_incorrect"]),
        reverse=True,
    )

    # Highly enriched motifs by class.
    most_correct_enriched = sorted(
        filtered,
        key=lambda r: (r["enrichment_correct"], r["support_total"]),
        reverse=True,
    )
    most_incorrect_enriched = sorted(
        filtered,
        key=lambda r: (r["enrichment_incorrect"], r["support_total"]),
        reverse=True,
    )

    # Most frequent motifs.
    most_frequent = sorted(filtered, key=lambda r: r["support_total"], reverse=True)

    print(f"Input file: {args.input}")
    print(f"Total motifs: {len(rows)}")
    print(f"Motifs with support >= {args.min_support}: {len(filtered)}")
    print()

    print("Label split (all motifs):")
    correct_dominant = sum(1 for row in rows if row["dominant_label"] == "correct")
    incorrect_dominant = sum(1 for row in rows if row["dominant_label"] == "incorrect")
    tie_dominant = sum(1 for row in rows if row["dominant_label"] == "tie")
    print(
        f"- correct-dominant={correct_dominant}, incorrect-dominant={incorrect_dominant}, tie={tie_dominant}"
    )
    print()

    print("Distribution of p_success (p_correct):")
    print(
        f"- min={p_correct_summary['min']:.3f}, p25={p_correct_summary['p25']:.3f}, "
        f"median={p_correct_summary['median']:.3f}, mean={p_correct_summary['mean']:.3f}, "
        f"p75={p_correct_summary['p75']:.3f}, max={p_correct_summary['max']:.3f}"
    )
    print("Histogram bins [left,right):")
    for left, right, count in histogram(p_correct_values, bins=10):
        bracket = "]" if right == 1.0 else ")"
        print(f"- [{left:.1f},{right:.1f}{bracket} -> {count}")
    print()

    print("Distribution of p_fail (p_incorrect):")
    print(
        f"- min={p_incorrect_summary['min']:.3f}, p25={p_incorrect_summary['p25']:.3f}, "
        f"median={p_incorrect_summary['median']:.3f}, mean={p_incorrect_summary['mean']:.3f}, "
        f"p75={p_incorrect_summary['p75']:.3f}, max={p_incorrect_summary['max']:.3f}"
    )
    print("Histogram bins [left,right):")
    for left, right, count in histogram(p_incorrect_values, bins=10):
        bracket = "]" if right == 1.0 else ")"
        print(f"- [{left:.1f},{right:.1f}{bracket} -> {count}")
    print()

    print_motif_list("Best motifs (highest p_success):", best_motifs, args.top_k)
    print_motif_list("Worst motifs (highest p_fail):", worst_motifs, args.top_k)
    print_motif_list(
        "Most correct-enriched motifs:", most_correct_enriched, args.top_k
    )
    print_motif_list(
        "Most incorrect-enriched motifs:", most_incorrect_enriched, args.top_k
    )
    print_motif_list("Most frequent motifs:", most_frequent, args.top_k)

    report_lines: list[str] = []
    report_lines.append("# Motif Subsequences Summary")
    report_lines.append("")
    report_lines.append(f"- Input file: `{args.input}`")
    report_lines.append(f"- Total motifs: **{len(rows)}**")
    report_lines.append(
        f"- Motifs with support >= {args.min_support}: **{len(filtered)}**"
    )
    report_lines.append("")
    report_lines.append("## Label Split")
    report_lines.append("")
    report_lines.append(
        f"- correct-dominant={correct_dominant}, incorrect-dominant={incorrect_dominant}, tie={tie_dominant}"
    )
    report_lines.append("")
    report_lines.extend(
        markdown_histogram_section(
            "Distribution of p_success (p_correct)",
            p_correct_values,
            p_correct_summary,
        )
    )
    report_lines.extend(
        markdown_histogram_section(
            "Distribution of p_fail (p_incorrect)",
            p_incorrect_values,
            p_incorrect_summary,
        )
    )
    report_lines.extend(
        markdown_motif_section("Best motifs (highest p_success)", best_motifs, args.top_k)
    )
    report_lines.extend(
        markdown_motif_section("Worst motifs (highest p_fail)", worst_motifs, args.top_k)
    )
    report_lines.extend(
        markdown_motif_section(
            "Most correct-enriched motifs", most_correct_enriched, args.top_k
        )
    )
    report_lines.extend(
        markdown_motif_section(
            "Most incorrect-enriched motifs", most_incorrect_enriched, args.top_k
        )
    )
    report_lines.extend(
        markdown_motif_section("Most frequent motifs", most_frequent, args.top_k)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved markdown summary to: {args.output}")


if __name__ == "__main__":
    main()
