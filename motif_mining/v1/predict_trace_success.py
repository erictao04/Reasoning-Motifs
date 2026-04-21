from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRACES_PATH = BASE_DIR / "trace_critical_steps.csv"
DEFAULT_METRICS_PATH = BASE_DIR / "signal_correlation_metrics.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "trace_success_predictions.csv"
DEFAULT_REPORT_PATH = BASE_DIR / "trace_success_prediction_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict trace success using correlation metrics and per-trace critical-step signals."
        )
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=DEFAULT_TRACES_PATH,
        help=f"Input trace critical steps CSV (default: {DEFAULT_TRACES_PATH}).",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=f"Input signal metrics CSV (default: {DEFAULT_METRICS_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output predictions CSV (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Output markdown report (default: {DEFAULT_REPORT_PATH}).",
    )
    return parser.parse_args()


def is_truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def load_metrics(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"No rows found in metrics file: {path}")

    metrics: dict[str, dict[str, float]] = {}
    for row in rows:
        signal = row["signal"]
        metrics[signal] = {
            "pearson": float(row["pearson_with_correct"]),
            "mean_correct": float(row["mean_correct"]),
            "mean_incorrect": float(row["mean_incorrect"]),
            "threshold": float(row["best_threshold"]),
            "best_acc": float(row["best_threshold_accuracy"]),
        }
    return metrics


def derive_signal_map(trace_row: dict[str, str]) -> dict[str, float]:
    success = float(trace_row["success_signal"])
    failure = float(trace_row["failure_signal"])
    trace_length = float(trace_row["trace_length"])
    net = float(trace_row["net_signal"])

    ratio = (success + 1e-9) / (failure + 1e-9)
    log_ratio = math.log(ratio)
    return {
        "success_signal": success,
        "failure_signal": failure,
        "net_signal": net,
        "trace_length": trace_length,
        "success_to_failure_ratio": ratio,
        "log_success_to_failure_ratio": log_ratio,
    }


def score_trace(
    signal_values: dict[str, float],
    metrics: dict[str, dict[str, float]],
) -> tuple[float, float, str]:
    total_score = 0.0
    contribution_parts: list[str] = []

    for signal_name, signal_metric in metrics.items():
        if signal_name not in signal_values:
            continue
        value = signal_values[signal_name]
        pearson = signal_metric["pearson"]
        threshold = signal_metric["threshold"]
        scale = abs(signal_metric["mean_correct"] - signal_metric["mean_incorrect"])
        if scale < 1e-9:
            scale = 1.0

        direction = 1.0 if pearson >= 0 else -1.0
        weight = abs(pearson)
        normalized_delta = direction * (value - threshold) / scale
        contribution = weight * normalized_delta
        total_score += contribution
        contribution_parts.append(f"{signal_name}:{contribution:.3f}")

    probability = sigmoid(total_score)
    return total_score, probability, " || ".join(contribution_parts)


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.metrics)

    if not args.traces.exists():
        raise FileNotFoundError(f"Traces file not found: {args.traces}")
    trace_rows = list(csv.DictReader(args.traces.open("r", newline="", encoding="utf-8")))
    if not trace_rows:
        raise ValueError(f"No rows found in traces file: {args.traces}")

    output_rows: list[dict[str, Any]] = []
    correct = 0
    for row in trace_rows:
        signal_values = derive_signal_map(row)
        raw_score, p_success, contrib = score_trace(signal_values, metrics)
        predicted_success = p_success >= 0.5
        actual_success = is_truthy(row.get("is_correct", ""))
        if predicted_success == actual_success:
            correct += 1

        output_rows.append(
            {
                "question_id": row.get("question_id", ""),
                "sample_id": row.get("sample_id", ""),
                "attempt_index": row.get("attempt_index", ""),
                "actual_is_correct": row.get("is_correct", ""),
                "predicted_is_correct": str(predicted_success),
                "p_success": f"{p_success:.6f}",
                "raw_score": f"{raw_score:.6f}",
                "success_signal": row.get("success_signal", ""),
                "failure_signal": row.get("failure_signal", ""),
                "net_signal": row.get("net_signal", ""),
                "trace_length": row.get("trace_length", ""),
                "signal_contributions": contrib,
                "reasoning_trace": row.get("reasoning_trace", ""),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "sample_id",
                "attempt_index",
                "actual_is_correct",
                "predicted_is_correct",
                "p_success",
                "raw_score",
                "success_signal",
                "failure_signal",
                "net_signal",
                "trace_length",
                "signal_contributions",
                "reasoning_trace",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    n = len(output_rows)
    accuracy = correct / n if n else 0.0
    avg_p_success = sum(float(row["p_success"]) for row in output_rows) / n if n else 0.0
    report_lines = [
        "# Trace Success Prediction Report",
        "",
        f"- Input traces: `{args.traces}`",
        f"- Input metrics: `{args.metrics}`",
        f"- Output predictions: `{args.output}`",
        f"- Rows scored: **{n}**",
        f"- Accuracy at threshold 0.5: **{accuracy:.4f}**",
        f"- Mean predicted p_success: **{avg_p_success:.4f}**",
        "",
        "Model details:",
        "- Uses signal weights from |Pearson correlation with correctness|.",
        "- Uses threshold-centered normalization based on |mean_correct - mean_incorrect|.",
        "- Final `raw_score` is converted with sigmoid to `p_success`.",
    ]
    args.report.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Scored {n} traces.")
    print(f"Accuracy (p_success >= 0.5): {accuracy:.4f}")
    print(f"Saved predictions to {args.output}")
    print(f"Saved report to {args.report}")


if __name__ == "__main__":
    main()
