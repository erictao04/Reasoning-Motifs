from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACES_PATH = ROOT / "tokenizer" / "v2" / "tokenized_pilot_traces.csv"
DEFAULT_MOTIFS_PATH = Path(__file__).resolve().parent / "all_motif_subsequences.csv"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "trace_critical_steps.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze each tokenized reasoning trace using motif subsequence statistics "
            "to identify critical steps linked to success vs failure."
        )
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=DEFAULT_TRACES_PATH,
        help=f"Tokenized traces CSV path (default: {DEFAULT_TRACES_PATH}).",
    )
    parser.add_argument(
        "--motifs",
        type=Path,
        default=DEFAULT_MOTIFS_PATH,
        help=f"Motif subsequences CSV path (default: {DEFAULT_MOTIFS_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output per-trace critical steps CSV path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=5,
        help="Ignore motifs with support_total below this threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many motifs/steps to keep for each trace.",
    )
    return parser.parse_args()


def is_truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_motif_index(
    motif_rows: list[dict[str, str]], *, min_support: int
) -> tuple[dict[tuple[str, ...], dict[str, Any]], int, int]:
    motif_index: dict[tuple[str, ...], dict[str, Any]] = {}
    min_len = 10**9
    max_len = 0

    for row in motif_rows:
        support_total = int(row["support_total"])
        if support_total < min_support:
            continue

        motif_tokens = tuple(token for token in row["motif"].split("|") if token)
        if not motif_tokens:
            continue

        p_correct = float(row["p_correct"])
        p_incorrect = float(row["p_incorrect"])
        enrichment_correct = float(row["enrichment_correct"])
        enrichment_incorrect = float(row["enrichment_incorrect"])

        # Weighted log-signal: stronger when enrichment is high and support is reliable.
        support_weight = math.log1p(support_total)
        success_signal = max(0.0, math.log(enrichment_correct)) * support_weight
        failure_signal = max(0.0, math.log(enrichment_incorrect)) * support_weight

        motif_index[motif_tokens] = {
            "motif": row["motif"],
            "motif_pretty": row["motif_pretty"],
            "length": len(motif_tokens),
            "support_total": support_total,
            "p_correct": p_correct,
            "p_incorrect": p_incorrect,
            "success_signal": success_signal,
            "failure_signal": failure_signal,
        }

        min_len = min(min_len, len(motif_tokens))
        max_len = max(max_len, len(motif_tokens))

    if not motif_index:
        raise ValueError("No motifs available after filtering. Lower --min-support.")

    return motif_index, min_len, max_len


def format_top_motifs(
    motif_counter: Counter[tuple[str, ...]],
    motif_score: dict[tuple[str, ...], float],
    motif_index: dict[tuple[str, ...], dict[str, Any]],
    *,
    top_k: int,
) -> str:
    ranked = sorted(
        motif_counter.keys(),
        key=lambda motif: (motif_score[motif], motif_counter[motif], len(motif)),
        reverse=True,
    )[:top_k]

    parts: list[str] = []
    for motif in ranked:
        meta = motif_index[motif]
        parts.append(
            f"{meta['motif']} (count={motif_counter[motif]}, score={motif_score[motif]:.3f}, "
            f"p_success={meta['p_correct']:.3f}, p_fail={meta['p_incorrect']:.3f})"
        )
    return " || ".join(parts)


def format_top_steps(step_score: dict[str, float], *, top_k: int) -> str:
    ranked = sorted(step_score.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return " || ".join(f"{token} ({score:.3f})" for token, score in ranked)


def analyze_trace(
    trace_tokens: list[str],
    motif_index: dict[tuple[str, ...], dict[str, Any]],
    *,
    min_len: int,
    max_len: int,
    top_k: int,
) -> dict[str, Any]:
    success_motif_count: Counter[tuple[str, ...]] = Counter()
    failure_motif_count: Counter[tuple[str, ...]] = Counter()
    success_motif_score: defaultdict[tuple[str, ...], float] = defaultdict(float)
    failure_motif_score: defaultdict[tuple[str, ...], float] = defaultdict(float)
    success_step_score: defaultdict[str, float] = defaultdict(float)
    failure_step_score: defaultdict[str, float] = defaultdict(float)

    for length in range(min_len, min(max_len, len(trace_tokens)) + 1):
        for start in range(0, len(trace_tokens) - length + 1):
            motif = tuple(trace_tokens[start : start + length])
            if motif not in motif_index:
                continue

            meta = motif_index[motif]
            s_signal = float(meta["success_signal"])
            f_signal = float(meta["failure_signal"])

            if s_signal > 0:
                success_motif_count[motif] += 1
                success_motif_score[motif] += s_signal
                for token in motif:
                    success_step_score[token] += s_signal

            if f_signal > 0:
                failure_motif_count[motif] += 1
                failure_motif_score[motif] += f_signal
                for token in motif:
                    failure_step_score[token] += f_signal

    total_success_signal = sum(success_motif_score.values())
    total_failure_signal = sum(failure_motif_score.values())
    net_signal = total_success_signal - total_failure_signal

    return {
        "success_signal": total_success_signal,
        "failure_signal": total_failure_signal,
        "net_signal": net_signal,
        "top_success_motifs": format_top_motifs(
            success_motif_count, success_motif_score, motif_index, top_k=top_k
        ),
        "top_failure_motifs": format_top_motifs(
            failure_motif_count, failure_motif_score, motif_index, top_k=top_k
        ),
        "top_success_steps": format_top_steps(success_step_score, top_k=top_k),
        "top_failure_steps": format_top_steps(failure_step_score, top_k=top_k),
    }


def main() -> None:
    args = parse_args()
    motif_rows = read_csv_rows(args.motifs)
    trace_rows = read_csv_rows(args.traces)
    motif_index, min_len, max_len = build_motif_index(
        motif_rows, min_support=args.min_support
    )

    output_rows: list[dict[str, Any]] = []
    for row in trace_rows:
        trace_tokens = [token for token in (row.get("reasoning_trace", "") or "").split("|") if token]
        analysis = analyze_trace(
            trace_tokens,
            motif_index,
            min_len=min_len,
            max_len=max_len,
            top_k=args.top_k,
        )

        is_correct_value = row.get("is_correct", "")
        is_correct = is_truthy(is_correct_value)
        if is_correct:
            critical_for_outcome = analysis["top_success_steps"]
        else:
            critical_for_outcome = analysis["top_failure_steps"]

        output_rows.append(
            {
                "question_id": row.get("question_id", ""),
                "sample_id": row.get("sample_id", ""),
                "attempt_index": row.get("attempt_index", ""),
                "is_correct": is_correct_value,
                "trace_length": len(trace_tokens),
                "success_signal": f"{analysis['success_signal']:.6f}",
                "failure_signal": f"{analysis['failure_signal']:.6f}",
                "net_signal": f"{analysis['net_signal']:.6f}",
                "top_success_motifs": analysis["top_success_motifs"],
                "top_failure_motifs": analysis["top_failure_motifs"],
                "top_success_steps": analysis["top_success_steps"],
                "top_failure_steps": analysis["top_failure_steps"],
                "critical_steps_for_observed_outcome": critical_for_outcome,
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
                "is_correct",
                "trace_length",
                "success_signal",
                "failure_signal",
                "net_signal",
                "top_success_motifs",
                "top_failure_motifs",
                "top_success_steps",
                "top_failure_steps",
                "critical_steps_for_observed_outcome",
                "reasoning_trace",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Read {len(trace_rows)} tokenized traces from {args.traces}")
    print(f"Read {len(motif_rows)} motif subsequences from {args.motifs}")
    print(
        f"Used {len(motif_index)} motifs after support filter (min_len={min_len}, max_len={max_len}, "
        f"min_support={args.min_support})"
    )
    print(f"Wrote per-trace critical-step analysis to {args.output}")


if __name__ == "__main__":
    main()
