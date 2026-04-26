#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_paper_experiments import (
    compute_mixed_question_ids,
    load_rows,
    run_question_holdout_experiment,
)

DEFAULT_INPUT = Path("gpt-oss-tokenized-traces.csv")
DEFAULT_OUTPUT_DIR = Path("paper_experiments/experiment1_gpt_oss_tokenized_traces")
DEFAULT_RAW_TRACE_INPUT = Path("tokenizer/expanded_pool_s100_seed73_qwen25_7b_hot30.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Experiment 1 (held-out question prediction) on a single tokenized "
            "trace CSV using the paper's logistic-regression motif baseline."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input tokenized trace CSV (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of question-holdout folds (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=73,
        help="Random seed for fold assignment (default: 73).",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=12,
        help="Minimum support threshold for motif features (default: 12).",
    )
    parser.add_argument(
        "--min-support-list",
        default="",
        help=(
            "Optional comma-separated support thresholds to sweep. "
            "When provided, runs one experiment per threshold."
        ),
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=1,
        help="Minimum motif length (default: 1).",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=3,
        help="Maximum motif length (default: 3).",
    )
    parser.add_argument(
        "--baseline-source",
        choices=["tokenized", "raw_words", "raw_chars"],
        default="tokenized",
        help=(
            "Baseline length source: tokenized motif-sequence length, "
            "raw reasoning-trace word count, or raw reasoning-trace character count."
        ),
    )
    parser.add_argument(
        "--raw-trace-input",
        type=Path,
        default=DEFAULT_RAW_TRACE_INPUT,
        help=(
            "Raw trace CSV used for raw baseline lengths "
            f"(default: {DEFAULT_RAW_TRACE_INPUT})."
        ),
    )
    return parser.parse_args()


def write_report(
    output_path: Path,
    *,
    input_path: Path,
    runs: list[dict[str, object]],
    total_rows: int,
    total_questions: int,
    mixed_rows: int,
    mixed_questions: int,
    baseline_source: str,
) -> None:
    lines = [
        "# Experiment 1 Report",
        "",
        f"Input: `{input_path}`",
        f"Baseline source: `{baseline_source}`",
        "",
        "## Dataset",
        "",
        f"- Total traces: {total_rows}",
        f"- Total questions: {total_questions}",
        f"- Mixed-outcome traces: {mixed_rows}",
        f"- Mixed-outcome questions: {mixed_questions}",
        "",
        "## Held-out Question Prediction Support Sweep",
        "",
        "| Support Rule | Slice | Accuracy | Balanced Accuracy | AUC | Q-Local AUC | Length Q-Local AUC | Avg Selected Features |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        "",
    ]
    for run in runs:
        support_label = str(run["support_rule"])
        full_agg = run["full"]["aggregate"]
        mixed_agg = run["mixed_questions"]["aggregate"]
        lines.append(
            "| "
            f"{support_label} | "
            f"full | "
            f"{float(full_agg['motif_accuracy']):.4f} | "
            f"{float(full_agg['motif_balanced_accuracy']):.4f} | "
            f"{float(full_agg['motif_auc']):.4f} | "
            f"{float(full_agg['motif_question_local_auc_mean']):.4f} | "
            f"{float(full_agg['length_question_local_auc_mean']):.4f} | "
            f"{float(full_agg['avg_selected_feature_count']):.1f} |"
        )
        lines.append(
            "| "
            f"{support_label} | "
            f"mixed_questions | "
            f"{float(mixed_agg['motif_accuracy']):.4f} | "
            f"{float(mixed_agg['motif_balanced_accuracy']):.4f} | "
            f"{float(mixed_agg['motif_auc']):.4f} | "
            f"{float(mixed_agg['motif_question_local_auc_mean']):.4f} | "
            f"{float(mixed_agg['length_question_local_auc_mean']):.4f} | "
            f"{float(mixed_agg['avg_selected_feature_count']):.1f} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_raw_length_lookup(raw_trace_input: Path, mode: str) -> dict[tuple[str, str], int]:
    if mode not in {"raw_words", "raw_chars"}:
        raise ValueError(f"Unsupported raw baseline mode: {mode}")

    lookup: dict[tuple[str, str], int] = {}
    with raw_trace_input.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"question_id", "sample_id", "reasoning_trace"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing required columns in {raw_trace_input}: {required}")

        for row in reader:
            trace = row["reasoning_trace"] or ""
            if mode == "raw_words":
                length = len(trace.split())
            else:
                length = len(trace)
            lookup[(str(row["question_id"]), str(row["sample_id"]))] = length
    return lookup


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mixed_question_ids = compute_mixed_question_ids(rows)
    mixed_rows = [row for row in rows if row.question_id in mixed_question_ids]
    question_ids = {row.question_id for row in rows}

    baseline_length_fn = None
    if args.baseline_source != "tokenized":
        raw_length_lookup = build_raw_length_lookup(args.raw_trace_input, args.baseline_source)
        missing_keys = [
            (row.question_id, row.sample_id)
            for row in rows
            if (row.question_id, row.sample_id) not in raw_length_lookup
        ]
        if missing_keys:
            preview = ", ".join(f"{qid}:{sid}" for qid, sid in missing_keys[:5])
            raise ValueError(
                f"Missing raw baseline lengths for {len(missing_keys)} rows. First few: {preview}"
            )

        def baseline_length_fn(row):
            return raw_length_lookup[(row.question_id, row.sample_id)]

    if args.min_support_list.strip():
        support_thresholds = [
            int(item.strip()) for item in args.min_support_list.split(",") if item.strip()
        ]
    else:
        support_thresholds = [args.min_support]

    runs: list[dict[str, object]] = []
    for min_support in support_thresholds:
        full_result = run_question_holdout_experiment(
            rows,
            folds=args.folds,
            seed=args.seed,
            min_support=min_support,
            prefix_fraction=1.0,
            min_len=args.min_len,
            max_len=args.max_len,
            experiment_name="motif_1to3_full_trace",
            baseline_length_fn=baseline_length_fn,
        )
        mixed_result = run_question_holdout_experiment(
            mixed_rows,
            folds=args.folds,
            seed=args.seed,
            min_support=min_support,
            prefix_fraction=1.0,
            min_len=args.min_len,
            max_len=args.max_len,
            experiment_name="motif_1to3_full_trace",
            baseline_length_fn=baseline_length_fn,
        )
        runs.append(
            {
                "support_rule": f"support >= {min_support}",
                "min_support": min_support,
                "full": full_result,
                "mixed_questions": mixed_result,
            }
        )

    payload = {
        "config": {
            "input": str(args.input),
            "folds": args.folds,
            "seed": args.seed,
            "min_support": args.min_support,
            "min_support_list": support_thresholds,
            "min_len": args.min_len,
            "max_len": args.max_len,
            "baseline_source": args.baseline_source,
            "raw_trace_input": str(args.raw_trace_input),
        },
        "dataset": {
            "total_rows": len(rows),
            "total_questions": len(question_ids),
            "mixed_rows": len(mixed_rows),
            "mixed_questions": len(mixed_question_ids),
        },
        "question_holdout_runs": runs,
    }

    summary_path = output_dir / "experiment1_summary.json"
    report_path = output_dir / "experiment1_report.md"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(
        report_path,
        input_path=args.input,
        runs=runs,
        total_rows=len(rows),
        total_questions=len(question_ids),
        mixed_rows=len(mixed_rows),
        mixed_questions=len(mixed_question_ids),
        baseline_source=args.baseline_source,
    )

    print(f"Wrote summary JSON to {summary_path}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
