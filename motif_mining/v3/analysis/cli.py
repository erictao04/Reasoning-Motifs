from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd

from .io_utils import (
    load_trace_csv,
    preprocess_traces,
    save_cleaned_data,
    save_json,
    split_success_failure,
    summarize_dataframe,
)
from .rules import mine_sequential_rules, top_rule_tables
from .sequential_patterns import discriminative_pattern_table, reduce_redundant_patterns, top_pattern_tables
from .skipgrams import mine_skipgrams, top_skipgram_tables


def _common_preprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True, help="Input CSV with tokenized traces.")
    parser.add_argument("--delimiter", default=" ", help='Token delimiter in tokenized_trace (default: " ").')
    parser.add_argument("--deduplicate", action="store_true", help="Drop duplicate tokenized traces globally.")
    parser.add_argument(
        "--dedupe-per-question",
        action="store_true",
        help="Drop duplicate tokenized traces within each question_id.",
    )
    parser.add_argument("--min-trace-len", type=int, default=None)
    parser.add_argument("--max-trace-len", type=int, default=None)
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit analysis to the first N unique question_id values (default: no limit).",
    )


def _load_cleaned(args: argparse.Namespace) -> pd.DataFrame:
    raw = load_trace_csv(args.input)
    cleaned = preprocess_traces(
        raw,
        delimiter=args.delimiter,
        deduplicate=args.deduplicate,
        dedupe_per_question=args.dedupe_per_question,
        min_trace_len=args.min_trace_len,
        max_trace_len=args.max_trace_len,
    )
    if args.max_questions is not None:
        if args.max_questions <= 0:
            raise ValueError("--max-questions must be a positive integer when provided.")
        question_ids = cleaned["question_id"].drop_duplicates().head(args.max_questions)
        cleaned = cleaned[cleaned["question_id"].isin(set(question_ids))].reset_index(drop=True)
    return cleaned


def _ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _plot_length_histogram(df: pd.DataFrame, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(df["trace_len"], bins=30)
    plt.title("Trace Length Distribution")
    plt.xlabel("Trace length (tokens)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def _plot_top_differences(df: pd.DataFrame, outpath: Path, title: str, top_k: int = 20) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    top = df.head(top_k).copy()
    plt.figure(figsize=(8, max(4, min(12, 0.35 * len(top) + 1))))
    plt.barh(range(len(top)), top["support_difference"])
    plt.yticks(range(len(top)), top["motif"])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Support difference (success - failure)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "No rows to display.\n"

    cols = [c for c in columns if c in df.columns]
    view = df[cols].head(max_rows).copy()
    view = view.fillna("")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in view.iterrows():
        values = [str(row[c]).replace("\n", " ").strip() for c in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _write_skipgrams_report(outdir: Path, all_df: pd.DataFrame, success_df: pd.DataFrame, failure_df: pd.DataFrame) -> None:
    lines = [
        "# Skip-gram Analysis Report",
        "",
        f"- Total mined skip-grams: {len(all_df)}",
        f"- Top success-enriched rows: {len(success_df)}",
        f"- Top failure-enriched rows: {len(failure_df)}",
        "",
        "## Top Success-Enriched Skip-grams",
        "",
        _markdown_table(
            success_df,
            ["motif", "length", "success_count", "failure_count", "support_difference", "lift", "log_odds_ratio", "q_value"],
            max_rows=20,
        ),
        "",
        "## Top Failure-Enriched Skip-grams",
        "",
        _markdown_table(
            failure_df,
            ["motif", "length", "success_count", "failure_count", "support_difference", "lift", "log_odds_ratio", "q_value"],
            max_rows=20,
        ),
    ]
    (outdir / "skipgrams_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_patterns_report(outdir: Path, all_df: pd.DataFrame, success_df: pd.DataFrame, failure_df: pd.DataFrame) -> None:
    lines = [
        "# Sequential Pattern Analysis Report",
        "",
        f"- Total mined patterns: {len(all_df)}",
        f"- Top success-enriched rows: {len(success_df)}",
        f"- Top failure-enriched rows: {len(failure_df)}",
        "",
        "## Top Success-Enriched Patterns",
        "",
        _markdown_table(
            success_df,
            ["motif", "length", "success_count", "failure_count", "support_difference", "lift", "log_odds_ratio", "q_value"],
            max_rows=20,
        ),
        "",
        "## Top Failure-Enriched Patterns",
        "",
        _markdown_table(
            failure_df,
            ["motif", "length", "success_count", "failure_count", "support_difference", "lift", "log_odds_ratio", "q_value"],
            max_rows=20,
        ),
    ]
    (outdir / "sequential_patterns_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_rules_report(outdir: Path, all_df: pd.DataFrame, success_df: pd.DataFrame, failure_df: pd.DataFrame) -> None:
    lines = [
        "# Sequential Rule Analysis Report",
        "",
        f"- Total mined rules: {len(all_df)}",
        f"- Top success-enriched rows: {len(success_df)}",
        f"- Top failure-enriched rows: {len(failure_df)}",
        "",
        "## Top Success-Enriched Rules",
        "",
        _markdown_table(
            success_df,
            [
                "motif",
                "success_count",
                "failure_count",
                "support_difference",
                "success_confidence",
                "failure_confidence",
                "success_rule_lift",
                "failure_rule_lift",
                "q_value",
            ],
            max_rows=20,
        ),
        "",
        "## Top Failure-Enriched Rules",
        "",
        _markdown_table(
            failure_df,
            [
                "motif",
                "success_count",
                "failure_count",
                "support_difference",
                "success_confidence",
                "failure_confidence",
                "success_rule_lift",
                "failure_rule_lift",
                "q_value",
            ],
            max_rows=20,
        ),
    ]
    (outdir / "rules_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_run_report(outdir: Path) -> None:
    summary_path = outdir / "summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    lines = [
        "# Motif Mining Run Report",
        "",
        "## Dataset Summary",
        "",
    ]
    if summary:
        lines.extend(
            [
                f"- num_traces: {summary.get('num_traces', 0)}",
                f"- num_success: {summary.get('num_success', 0)}",
                f"- num_failure: {summary.get('num_failure', 0)}",
                f"- success_rate: {summary.get('success_rate', 0)}",
                f"- avg_trace_len: {summary.get('avg_trace_len', 0)}",
                f"- median_trace_len: {summary.get('median_trace_len', 0)}",
            ]
        )
    else:
        lines.append("Summary not found.")

    lines.extend(
        [
            "",
            "## Generated Reports",
            "",
            "- [skipgrams_report.md](./skipgrams_report.md)",
            "- [sequential_patterns_report.md](./sequential_patterns_report.md)",
            "- [rules_report.md](./rules_report.md)",
        ]
    )
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_summarize(args: argparse.Namespace) -> None:
    outdir = _ensure_outdir(args.outdir)
    df = _load_cleaned(args)

    save_cleaned_data(df, outdir / "cleaned_traces.csv")
    summary = summarize_dataframe(df)
    save_json(summary, outdir / "summary.json")

    if args.plots:
        _plot_length_histogram(df, outdir / "plots" / "trace_length_histogram.png")

    print(f"Wrote summary to: {outdir / 'summary.json'}")


def cmd_skipgrams(args: argparse.Namespace) -> None:
    outdir = _ensure_outdir(args.outdir)
    df = _load_cleaned(args)
    success_df, failure_df = split_success_failure(df)

    table = mine_skipgrams(
        success_df["tokens"].tolist(),
        failure_df["tokens"].tolist(),
        min_len=args.min_len,
        max_len=args.max_len,
        max_gap=args.max_gap,
        min_support_count=args.min_support,
    )
    success_top, failure_top = top_skipgram_tables(table, top_k=args.top_k)

    _write_table(table, outdir / "skipgrams_all.csv")
    _write_table(success_top, outdir / "skipgrams_success.csv")
    _write_table(failure_top, outdir / "skipgrams_failure.csv")
    _write_skipgrams_report(outdir, table, success_top, failure_top)

    if args.plots and not table.empty:
        _plot_top_differences(
            success_top,
            outdir / "plots" / "skipgrams_top_success.png",
            "Top Success-Enriched Skip-grams",
            top_k=min(20, len(success_top)),
        )
        _plot_top_differences(
            failure_top,
            outdir / "plots" / "skipgrams_top_failure.png",
            "Top Failure-Enriched Skip-grams",
            top_k=min(20, len(failure_top)),
        )

    print(f"Wrote skip-gram tables under: {outdir}")


def cmd_patterns(args: argparse.Namespace) -> None:
    outdir = _ensure_outdir(args.outdir)
    df = _load_cleaned(args)
    success_df, failure_df = split_success_failure(df)

    table = discriminative_pattern_table(
        success_df["tokens"].tolist(),
        failure_df["tokens"].tolist(),
        backend=args.backend,
        min_support_count=args.min_support,
        min_len=args.min_len,
        max_len=args.max_len,
    )

    if args.reduce_redundancy and not table.empty:
        table = reduce_redundant_patterns(
            table,
            support_tolerance=args.redundancy_support_tol,
            score_column="abs_log_odds_ratio",
        )

    success_top, failure_top = top_pattern_tables(table, top_k=args.top_k)
    _write_table(table, outdir / "sequential_patterns_all.csv")
    _write_table(success_top, outdir / "sequential_patterns_success.csv")
    _write_table(failure_top, outdir / "sequential_patterns_failure.csv")
    _write_patterns_report(outdir, table, success_top, failure_top)

    if args.plots and not table.empty:
        _plot_top_differences(
            success_top,
            outdir / "plots" / "patterns_top_success.png",
            "Top Success-Enriched Sequential Patterns",
            top_k=min(20, len(success_top)),
        )
        _plot_top_differences(
            failure_top,
            outdir / "plots" / "patterns_top_failure.png",
            "Top Failure-Enriched Sequential Patterns",
            top_k=min(20, len(failure_top)),
        )

    print(f"Wrote sequential-pattern tables under: {outdir}")


def cmd_rules(args: argparse.Namespace) -> None:
    outdir = _ensure_outdir(args.outdir)
    df = _load_cleaned(args)
    success_df, failure_df = split_success_failure(df)

    table = mine_sequential_rules(
        success_df["tokens"].tolist(),
        failure_df["tokens"].tolist(),
        min_support_count=args.min_support,
        max_len=args.max_len,
        backend=args.backend,
        max_candidates=args.max_candidates,
    )
    success_top, failure_top = top_rule_tables(table, top_k=args.top_k)

    _write_table(table, outdir / "rules_all.csv")
    _write_table(success_top, outdir / "rules_success.csv")
    _write_table(failure_top, outdir / "rules_failure.csv")
    _write_rules_report(outdir, table, success_top, failure_top)
    print(f"Wrote rule tables under: {outdir}")


def cmd_all(args: argparse.Namespace) -> None:
    outdir = _ensure_outdir(args.outdir)

    # Reuse same namespace across specialized commands.
    cmd_summarize(args)
    cmd_skipgrams(args)
    cmd_patterns(args)
    cmd_rules(args)
    _write_run_report(outdir)

    print(f"Completed all analyses under: {outdir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reasoning motif mining toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize", help="Load, clean, and summarize traces")
    _common_preprocess_args(summarize)
    summarize.add_argument("--outdir", type=Path, required=True)
    summarize.add_argument("--plots", action="store_true")
    summarize.set_defaults(func=cmd_summarize)

    skipgrams = subparsers.add_parser("skipgrams", help="Mine bounded-gap skip-grams")
    _common_preprocess_args(skipgrams)
    skipgrams.add_argument("--outdir", type=Path, required=True)
    skipgrams.add_argument("--min-len", type=int, default=2)
    skipgrams.add_argument("--max-len", type=int, default=4)
    skipgrams.add_argument("--max-gap", type=int, default=2)
    skipgrams.add_argument("--min-support", type=int, default=5)
    skipgrams.add_argument("--top-k", type=int, default=100)
    skipgrams.add_argument("--plots", action="store_true")
    skipgrams.set_defaults(func=cmd_skipgrams)

    patterns = subparsers.add_parser("patterns", help="Mine sequential patterns with gaps")
    _common_preprocess_args(patterns)
    patterns.add_argument("--outdir", type=Path, required=True)
    patterns.add_argument("--backend", choices=["auto", "python", "prefixspan"], default="auto")
    patterns.add_argument("--min-len", type=int, default=1)
    patterns.add_argument("--max-len", type=int, default=4)
    patterns.add_argument("--min-support", type=int, default=5)
    patterns.add_argument("--top-k", type=int, default=100)
    patterns.add_argument("--reduce-redundancy", action="store_true")
    patterns.add_argument("--redundancy-support-tol", type=float, default=0.005)
    patterns.add_argument("--plots", action="store_true")
    patterns.set_defaults(func=cmd_patterns)

    rules = subparsers.add_parser("rules", help="Mine sequential rules antecedent => consequent")
    _common_preprocess_args(rules)
    rules.add_argument("--outdir", type=Path, required=True)
    rules.add_argument("--backend", choices=["auto", "python", "prefixspan"], default="auto")
    rules.add_argument("--max-len", type=int, default=4)
    rules.add_argument("--min-support", type=int, default=5)
    rules.add_argument("--max-candidates", type=int, default=80)
    rules.add_argument("--top-k", type=int, default=100)
    rules.set_defaults(func=cmd_rules)

    all_cmd = subparsers.add_parser("all", help="Run summary + skipgrams + patterns + rules")
    _common_preprocess_args(all_cmd)
    all_cmd.add_argument("--outdir", type=Path, required=True)
    all_cmd.add_argument("--backend", choices=["auto", "python", "prefixspan"], default="auto")
    all_cmd.add_argument("--min-len", type=int, default=2)
    all_cmd.add_argument("--max-len", type=int, default=4)
    all_cmd.add_argument("--max-gap", type=int, default=2)
    all_cmd.add_argument("--min-support", type=int, default=5)
    all_cmd.add_argument("--top-k", type=int, default=100)
    all_cmd.add_argument("--max-candidates", type=int, default=80)
    all_cmd.add_argument("--reduce-redundancy", action="store_true")
    all_cmd.add_argument("--redundancy-support-tol", type=float, default=0.005)
    all_cmd.add_argument("--plots", action="store_true")
    all_cmd.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    fn: Callable[[argparse.Namespace], None] = args.func
    fn(args)


if __name__ == "__main__":
    main()
