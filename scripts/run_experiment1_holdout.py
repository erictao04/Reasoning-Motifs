#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_paper_experiments import (
    compute_mixed_question_ids,
    load_rows,
    run_question_holdout_experiment,
)

DEFAULT_INPUT = Path("gpt-oss-tokenized-traces.csv")
DEFAULT_OUTPUT_DIR = Path("paper_experiments/experiment1_gpt_oss_tokenized_traces")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Experiment 1 (held-out question prediction) on a single tokenized "
            "trace CSV using the paper's Bernoulli motif baseline."
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
    return parser.parse_args()


def write_report(
    output_path: Path,
    *,
    input_path: Path,
    full_result: dict[str, object],
    mixed_result: dict[str, object],
    total_rows: int,
    total_questions: int,
    mixed_rows: int,
    mixed_questions: int,
) -> None:
    full_agg = full_result["aggregate"]
    mixed_agg = mixed_result["aggregate"]

    lines = [
        "# Experiment 1 Report",
        "",
        f"Input: `{input_path}`",
        "",
        "## Dataset",
        "",
        f"- Total traces: {total_rows}",
        f"- Total questions: {total_questions}",
        f"- Mixed-outcome traces: {mixed_rows}",
        f"- Mixed-outcome questions: {mixed_questions}",
        "",
        "## Held-out Question Prediction",
        "",
        "| Slice | Accuracy | Balanced Accuracy | AUC | Q-Local AUC | Length Q-Local AUC | Avg Selected Features |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            "| full | "
            f"{float(full_agg['motif_accuracy']):.4f} | "
            f"{float(full_agg['motif_balanced_accuracy']):.4f} | "
            f"{float(full_agg['motif_auc']):.4f} | "
            f"{float(full_agg['motif_question_local_auc_mean']):.4f} | "
            f"{float(full_agg['length_question_local_auc_mean']):.4f} | "
            f"{float(full_agg['avg_selected_feature_count']):.1f} |"
        ),
        (
            "| mixed_questions | "
            f"{float(mixed_agg['motif_accuracy']):.4f} | "
            f"{float(mixed_agg['motif_balanced_accuracy']):.4f} | "
            f"{float(mixed_agg['motif_auc']):.4f} | "
            f"{float(mixed_agg['motif_question_local_auc_mean']):.4f} | "
            f"{float(mixed_agg['length_question_local_auc_mean']):.4f} | "
            f"{float(mixed_agg['avg_selected_feature_count']):.1f} |"
        ),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mixed_question_ids = compute_mixed_question_ids(rows)
    mixed_rows = [row for row in rows if row.question_id in mixed_question_ids]
    question_ids = {row.question_id for row in rows}

    full_result = run_question_holdout_experiment(
        rows,
        folds=args.folds,
        seed=args.seed,
        min_support=args.min_support,
        prefix_fraction=1.0,
        min_len=args.min_len,
        max_len=args.max_len,
        experiment_name="motif_1to3_full_trace",
    )
    mixed_result = run_question_holdout_experiment(
        mixed_rows,
        folds=args.folds,
        seed=args.seed,
        min_support=args.min_support,
        prefix_fraction=1.0,
        min_len=args.min_len,
        max_len=args.max_len,
        experiment_name="motif_1to3_full_trace",
    )

    payload = {
        "config": {
            "input": str(args.input),
            "folds": args.folds,
            "seed": args.seed,
            "min_support": args.min_support,
            "min_len": args.min_len,
            "max_len": args.max_len,
        },
        "dataset": {
            "total_rows": len(rows),
            "total_questions": len(question_ids),
            "mixed_rows": len(mixed_rows),
            "mixed_questions": len(mixed_question_ids),
        },
        "question_holdout_results": {
            "full": full_result,
            "mixed_questions": mixed_result,
        },
    }

    summary_path = output_dir / "experiment1_summary.json"
    report_path = output_dir / "experiment1_report.md"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(
        report_path,
        input_path=args.input,
        full_result=full_result,
        mixed_result=mixed_result,
        total_rows=len(rows),
        total_questions=len(question_ids),
        mixed_rows=len(mixed_rows),
        mixed_questions=len(mixed_question_ids),
    )

    print(f"Wrote summary JSON to {summary_path}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
