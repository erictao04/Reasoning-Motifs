from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_INPUT_PATH = Path(__file__).resolve().parent / "trace_critical_steps.csv"
DEFAULT_REPORT_PATH = Path(__file__).resolve().parent / "signal_correlation_report.md"
DEFAULT_METRICS_PATH = Path(__file__).resolve().parent / "signal_correlation_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze correlation between per-trace critical-step signals and correctness."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input critical steps CSV path (default: {DEFAULT_INPUT_PATH}).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Markdown report output path (default: {DEFAULT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=f"Metrics CSV output path (default: {DEFAULT_METRICS_PATH}).",
    )
    return parser.parse_args()


def is_truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - 1)


def stddev(values: list[float]) -> float:
    return math.sqrt(max(0.0, variance(values)))


def pearson_corr(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx = mean(x)
    my = mean(y)
    sx = stddev(x)
    sy = stddev(y)
    if sx == 0.0 or sy == 0.0:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (len(x) - 1)
    return cov / (sx * sy)


def ranks(values: list[float]) -> list[float]:
    # Average rank for ties.
    indexed = sorted(enumerate(values), key=lambda t: t[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based average rank
        for k in range(i, j + 1):
            out[indexed[k][0]] = avg_rank
        i = j + 1
    return out


def spearman_corr(x: list[float], y: list[float]) -> float:
    return pearson_corr(ranks(x), ranks(y))


def auc_for_binary_labels(signal: list[float], labels: list[int]) -> float:
    # AUC via rank sum (Mann-Whitney U).
    positives = [i for i, label in enumerate(labels) if label == 1]
    negatives = [i for i, label in enumerate(labels) if label == 0]
    n_pos = len(positives)
    n_neg = len(negatives)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    signal_ranks = ranks(signal)
    rank_sum_pos = sum(signal_ranks[i] for i in positives)
    u = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return u / (n_pos * n_neg)


def best_threshold_accuracy(signal: list[float], labels: list[int]) -> tuple[float, float, str]:
    # Return best accuracy, threshold, direction where:
    # - "ge": predict correct if signal >= threshold
    # - "le": predict correct if signal <= threshold
    if not signal:
        return 0.0, 0.0, "ge"

    unique_vals = sorted(set(signal))
    candidates = [unique_vals[0] - 1e-12]
    for i in range(len(unique_vals) - 1):
        candidates.append((unique_vals[i] + unique_vals[i + 1]) / 2.0)
    candidates.append(unique_vals[-1] + 1e-12)

    best_acc = -1.0
    best_thresh = candidates[0]
    best_dir = "ge"
    n = len(signal)

    for thresh in candidates:
        for direction in ("ge", "le"):
            correct = 0
            for s, label in zip(signal, labels):
                pred = 1 if (s >= thresh if direction == "ge" else s <= thresh) else 0
                if pred == label:
                    correct += 1
            acc = correct / n
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
                best_dir = direction

    return best_acc, best_thresh, best_dir


def cohen_d(group1: list[float], group0: list[float]) -> float:
    # Effect size: (mean1 - mean0) / pooled_std
    n1 = len(group1)
    n0 = len(group0)
    if n1 < 2 or n0 < 2:
        return 0.0
    v1 = variance(group1)
    v0 = variance(group0)
    pooled = ((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2)
    if pooled <= 0:
        return 0.0
    return (mean(group1) - mean(group0)) / math.sqrt(pooled)


def format_float(x: float) -> str:
    return f"{x:.6f}"


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    with args.input.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {args.input}")

    labels = [1 if is_truthy(row["is_correct"]) else 0 for row in rows]

    signals: dict[str, list[float]] = {
        "success_signal": [float(row["success_signal"]) for row in rows],
        "failure_signal": [float(row["failure_signal"]) for row in rows],
        "net_signal": [float(row["net_signal"]) for row in rows],
        "trace_length": [float(row["trace_length"]) for row in rows],
    }
    # Derived signals.
    signals["success_to_failure_ratio"] = [
        (s + 1e-9) / (f + 1e-9) for s, f in zip(signals["success_signal"], signals["failure_signal"])
    ]
    signals["log_success_to_failure_ratio"] = [
        math.log((s + 1e-9) / (f + 1e-9))
        for s, f in zip(signals["success_signal"], signals["failure_signal"])
    ]

    metric_rows: list[dict[str, str]] = []
    pos_idx = [i for i, y in enumerate(labels) if y == 1]
    neg_idx = [i for i, y in enumerate(labels) if y == 0]

    for signal_name, values in signals.items():
        values_pos = [values[i] for i in pos_idx]
        values_neg = [values[i] for i in neg_idx]

        pearson = pearson_corr(values, [float(y) for y in labels])
        spearman = spearman_corr(values, [float(y) for y in labels])
        auc = auc_for_binary_labels(values, labels)
        best_acc, best_thresh, best_dir = best_threshold_accuracy(values, labels)
        d = cohen_d(values_pos, values_neg)

        metric_rows.append(
            {
                "signal": signal_name,
                "n": str(len(values)),
                "mean_correct": format_float(mean(values_pos)),
                "mean_incorrect": format_float(mean(values_neg)),
                "pearson_with_correct": format_float(pearson),
                "spearman_with_correct": format_float(spearman),
                "auc_correct_when_higher": format_float(auc),
                "cohen_d_correct_minus_incorrect": format_float(d),
                "best_threshold_accuracy": format_float(best_acc),
                "best_threshold": format_float(best_thresh),
                "best_threshold_direction": best_dir,
            }
        )

    # Sort by absolute linear correlation with correctness.
    metric_rows.sort(
        key=lambda row: abs(float(row["pearson_with_correct"])),
        reverse=True,
    )

    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "signal",
                "n",
                "mean_correct",
                "mean_incorrect",
                "pearson_with_correct",
                "spearman_with_correct",
                "auc_correct_when_higher",
                "cohen_d_correct_minus_incorrect",
                "best_threshold_accuracy",
                "best_threshold",
                "best_threshold_direction",
            ],
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    correct_rate = mean([float(y) for y in labels])
    report_lines: list[str] = []
    report_lines.append("# Signal-Correctness Correlation Report")
    report_lines.append("")
    report_lines.append(f"- Input: `{args.input}`")
    report_lines.append(f"- Rows: **{len(rows)}**")
    report_lines.append(f"- Correct rate: **{correct_rate:.3f}**")
    report_lines.append(f"- Metrics CSV: `{args.metrics_csv}`")
    report_lines.append("")
    report_lines.append("## Ranked Signals (by |Pearson|)")
    report_lines.append("")
    report_lines.append("| Signal | mean(correct) | mean(incorrect) | Pearson | Spearman | AUC (higher=>correct) | Cohen d | Best threshold acc | Direction |")
    report_lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in metric_rows:
        report_lines.append(
            f"| {row['signal']} | {row['mean_correct']} | {row['mean_incorrect']} | "
            f"{row['pearson_with_correct']} | {row['spearman_with_correct']} | "
            f"{row['auc_correct_when_higher']} | {row['cohen_d_correct_minus_incorrect']} | "
            f"{row['best_threshold_accuracy']} | {row['best_threshold_direction']} @ {row['best_threshold']} |"
        )
    report_lines.append("")
    report_lines.append("## Notes")
    report_lines.append("")
    report_lines.append("- Positive Pearson/Spearman means higher signal is associated with correct answers.")
    report_lines.append("- AUC near 1.0 means strong separation where larger signal implies correctness.")
    report_lines.append("- AUC near 0.0 means inverse separation (larger implies incorrectness).")
    report_lines.append("- Best threshold accuracy is single-signal classification quality on this dataset.")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Read {len(rows)} rows from {args.input}")
    print(f"Wrote metrics CSV to {args.metrics_csv}")
    print(f"Wrote markdown report to {args.report}")


if __name__ == "__main__":
    main()
