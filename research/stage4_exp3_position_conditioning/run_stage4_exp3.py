#!/usr/bin/env python3
"""Stage 4 / Experiment 3: position-conditioned token delta analysis.

Implements Stage 4 from docs/research_plan.md:
1) Restrict to mixed-outcome questions.
2) Bin token occurrences by trace-relative position (first/middle/last third).
3) Run Stage-2-style within-question deltas for each (token, position_bin).
4) Compare position-conditioned |delta| against position-collapsed |delta|.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


TOKEN_COL_CANDIDATES = ("tokenized_trace", "tokens", "tokenized")
QUESTION_COL_CANDIDATES = ("question_id", "qid", "question")
CORRECT_COL_CANDIDATES = ("is_correct", "correct", "label")
POSITION_BINS = ("first_third", "middle_third", "last_third")


@dataclass
class TraceRow:
    question_id: str
    is_correct: int
    tokens: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 4 position-conditioned within-question delta analysis."
    )
    parser.add_argument("--input", type=Path, required=True, help="Tokenized CSV input.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("research/stage4_exp3_position_conditioning/results"),
        help="Output directory for Stage 4 artifacts.",
    )
    parser.add_argument(
        "--min-questions",
        type=int,
        default=10,
        help="Minimum mixed-outcome question presence for a feature.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="Bootstrap resamples for 95%% CI and paired token-level aggregate tests.",
    )
    parser.add_argument(
        "--fdr",
        type=float,
        default=0.10,
        help="FDR threshold for significance marking.",
    )
    parser.add_argument(
        "--lift-threshold",
        type=float,
        default=1.5,
        help="Position-lift threshold multiplier over position-collapsed |mean_delta|.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=73,
        help="Random seed for bootstrap.",
    )
    return parser.parse_args()


def detect_column(fieldnames: list[str], candidates: tuple[str, ...], kind: str) -> str:
    for cand in candidates:
        if cand in fieldnames:
            return cand
    options = ", ".join(fieldnames)
    raise ValueError(f"Could not find {kind} column. Available: {options}")


def parse_bool(value: str) -> int:
    text = (value or "").strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return 1
    if text in {"0", "false", "f", "no", "n"}:
        return 0
    raise ValueError(f"Unrecognized correctness label: {value!r}")


def parse_tokens(value: str) -> list[str]:
    text = (value or "").strip()
    if not text or text.upper() == "MISSING":
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(tok).strip() for tok in parsed if str(tok).strip()]
        except json.JSONDecodeError:
            pass
    return [tok for tok in text.replace(",", " ").split() if tok]


def load_rows(input_csv: Path) -> tuple[list[TraceRow], dict[str, str]]:
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError(f"CSV has no header: {input_csv}")
        question_col = detect_column(fieldnames, QUESTION_COL_CANDIDATES, "question id")
        correct_col = detect_column(fieldnames, CORRECT_COL_CANDIDATES, "correctness")
        token_col = detect_column(fieldnames, TOKEN_COL_CANDIDATES, "token sequence")

        rows: list[TraceRow] = []
        for row in reader:
            try:
                is_correct = parse_bool(row.get(correct_col, ""))
            except ValueError:
                continue
            rows.append(
                TraceRow(
                    question_id=str(row.get(question_col, "")).strip(),
                    is_correct=is_correct,
                    tokens=parse_tokens(row.get(token_col, "")),
                )
            )
    return rows, {"question": question_col, "correct": correct_col, "token": token_col}


def group_by_question(rows: Iterable[TraceRow]) -> dict[str, list[TraceRow]]:
    grouped: dict[str, list[TraceRow]] = defaultdict(list)
    for row in rows:
        if row.question_id:
            grouped[row.question_id].append(row)
    return grouped


def mixed_outcome_questions(grouped: dict[str, list[TraceRow]]) -> set[str]:
    mixed = set()
    for qid, qrows in grouped.items():
        labels = {r.is_correct for r in qrows}
        if labels == {0, 1}:
            mixed.add(qid)
    return mixed


def sign_test_p_value(deltas: list[float]) -> tuple[float, int, int, int]:
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0, pos, neg, n
    k = min(pos, neg)
    cdf = 0.0
    for i in range(0, k + 1):
        cdf += math.comb(n, i) * (0.5**n)
    p = min(1.0, 2.0 * cdf)
    return p, pos, neg, n


def bootstrap_ci_mean(values: list[float], rng: random.Random, samples: int) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    n = len(values)
    means: list[float] = []
    for _ in range(samples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()
    lo_idx = int(0.025 * (samples - 1))
    hi_idx = int(0.975 * (samples - 1))
    return means[lo_idx], means[hi_idx]


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if m == 0:
        return []
    ranked = sorted(enumerate(pvals), key=lambda x: x[1])
    qvals = [1.0] * m
    running_min = 1.0
    for rank in range(m, 0, -1):
        idx, p = ranked[rank - 1]
        q = (p * m) / rank
        running_min = min(running_min, q)
        qvals[idx] = min(1.0, running_min)
    return qvals


def token_position_bin(index: int, n_tokens: int) -> str:
    if n_tokens <= 0:
        return ""
    rel = (index + 0.5) / n_tokens
    if rel < (1.0 / 3.0):
        return "first_third"
    if rel < (2.0 / 3.0):
        return "middle_third"
    return "last_third"


def feature_set_collapsed(row: TraceRow) -> set[str]:
    return set(row.tokens)


def feature_set_positioned(row: TraceRow) -> set[str]:
    features: set[str] = set()
    n = len(row.tokens)
    for i, tok in enumerate(row.tokens):
        pos = token_position_bin(i, n)
        if not pos:
            continue
        features.add(f"{tok}@@{pos}")
    return features


def compute_feature_deltas(
    mixed_rows_by_q: dict[str, list[TraceRow]],
    feature_extractor: Callable[[TraceRow], set[str]],
    min_questions: int,
    bootstrap_samples: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    feature_questions: dict[str, set[str]] = defaultdict(set)
    feature_cache: dict[tuple[str, int], set[str]] = {}

    for qid, qrows in mixed_rows_by_q.items():
        present: set[str] = set()
        for i, r in enumerate(qrows):
            feats = feature_extractor(r)
            feature_cache[(qid, i)] = feats
            present.update(feats)
        for feat in present:
            feature_questions[feat].add(qid)

    records: list[dict[str, object]] = []
    for feature, qids_present in feature_questions.items():
        if len(qids_present) < min_questions:
            continue
        deltas: list[float] = []
        questions_used = 0
        for qid, qrows in mixed_rows_by_q.items():
            with_feat: list[TraceRow] = []
            without_feat: list[TraceRow] = []
            for i, r in enumerate(qrows):
                feats = feature_cache[(qid, i)]
                if feature in feats:
                    with_feat.append(r)
                else:
                    without_feat.append(r)
            if not with_feat or not without_feat:
                continue
            succ_with = sum(r.is_correct for r in with_feat) / len(with_feat)
            succ_without = sum(r.is_correct for r in without_feat) / len(without_feat)
            deltas.append(succ_with - succ_without)
            questions_used += 1

        if not deltas:
            continue
        ci_lo, ci_hi = bootstrap_ci_mean(deltas, rng, bootstrap_samples)
        p, pos, neg, n_nonzero = sign_test_p_value(deltas)
        records.append(
            {
                "feature": feature,
                "question_presence": len(qids_present),
                "questions_used": questions_used,
                "mean_delta": sum(deltas) / len(deltas),
                "median_delta": statistics.median(deltas),
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
                "sign_test_p": p,
                "sign_pos": pos,
                "sign_neg": neg,
                "sign_n_nonzero": n_nonzero,
            }
        )

    pvals = [float(r["sign_test_p"]) for r in records]
    qvals = benjamini_hochberg(pvals)
    for rec, q in zip(records, qvals):
        rec["bh_q"] = q
    return records


def parse_position_feature(feature: str) -> tuple[str, str]:
    if "@@" not in feature:
        return feature, ""
    tok, pos = feature.rsplit("@@", 1)
    return tok, pos


def bootstrap_lift_summary(
    token_lifts: list[dict[str, float]],
    rng: random.Random,
    samples: int,
    threshold: float,
) -> dict[str, float]:
    if not token_lifts:
        return {
            "n_tokens_evaluated": 0,
            "n_tokens_ge_threshold": 0,
            "fraction_ge_threshold": 0.0,
            "fraction_ge_threshold_ci95_low": 0.0,
            "fraction_ge_threshold_ci95_high": 0.0,
            "mean_excess_over_threshold": 0.0,
            "mean_excess_ci95_low": 0.0,
            "mean_excess_ci95_high": 0.0,
            "paired_bootstrap_p_excess_le_zero": 1.0,
        }

    n = len(token_lifts)
    frac_values: list[float] = []
    excess_values: list[float] = []
    for _ in range(samples):
        draw = [token_lifts[rng.randrange(n)] for _ in range(n)]
        indicators = [1.0 if d["lift_ratio"] >= threshold else 0.0 for d in draw]
        excess = [d["max_position_abs"] - (threshold * d["collapsed_abs"]) for d in draw]
        frac_values.append(sum(indicators) / n)
        excess_values.append(sum(excess) / n)

    frac_values.sort()
    excess_values.sort()
    lo_idx = int(0.025 * (samples - 1))
    hi_idx = int(0.975 * (samples - 1))
    p_excess_le_zero = sum(1 for v in excess_values if v <= 0.0) / len(excess_values)

    n_ge = sum(1 for d in token_lifts if d["lift_ratio"] >= threshold)
    return {
        "n_tokens_evaluated": n,
        "n_tokens_ge_threshold": n_ge,
        "fraction_ge_threshold": n_ge / n,
        "fraction_ge_threshold_ci95_low": frac_values[lo_idx],
        "fraction_ge_threshold_ci95_high": frac_values[hi_idx],
        "mean_excess_over_threshold": sum(
            d["max_position_abs"] - (threshold * d["collapsed_abs"]) for d in token_lifts
        )
        / n,
        "mean_excess_ci95_low": excess_values[lo_idx],
        "mean_excess_ci95_high": excess_values[hi_idx],
        "paired_bootstrap_p_excess_le_zero": p_excess_le_zero,
    }


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_position_lift_plot(path: Path, token_lifts: list[dict[str, float]], threshold: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not token_lifts:
        return

    xs = [d["collapsed_abs"] for d in token_lifts]
    ys = [d["max_position_abs"] for d in token_lifts]
    labels = [d["token"] for d in token_lifts]
    colors = ["#2E7D32" if d["lift_ratio"] >= threshold else "#1F77B4" for d in token_lifts]

    max_axis = max(max(xs), max(ys), 1e-6)
    line = [0.0, max_axis * 1.05]
    thresh_line = [v * threshold for v in line]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(xs, ys, c=colors, alpha=0.85, s=22, linewidths=0)
    ax.plot(line, line, linestyle="--", color="#555555", linewidth=1.1, label="1.0x")
    ax.plot(line, thresh_line, linestyle=":", color="#B22222", linewidth=1.3, label=f"{threshold:.1f}x")
    ax.set_xlabel("|mean delta| (position-collapsed)")
    ax.set_ylabel("max |mean delta| across position bins")
    ax.set_title("Stage 4 position lift by token")
    ax.legend(loc="best")

    top = sorted(token_lifts, key=lambda d: d["lift_ratio"], reverse=True)[:12]
    for item in top:
        x = item["collapsed_abs"]
        y = item["max_position_abs"]
        if x <= 0 and y <= 0:
            continue
        ax.annotate(item["token"], (x, y), fontsize=7, alpha=0.85, xytext=(3, 3), textcoords="offset points")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    rows, columns_used = load_rows(args.input)
    by_q = group_by_question(rows)
    mixed_q = mixed_outcome_questions(by_q)
    mixed_rows_by_q = {qid: by_q[qid] for qid in mixed_q}

    collapsed_stats = compute_feature_deltas(
        mixed_rows_by_q=mixed_rows_by_q,
        feature_extractor=feature_set_collapsed,
        min_questions=args.min_questions,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    collapsed_by_token = {
        str(r["feature"]): float(r["mean_delta"])
        for r in collapsed_stats
    }

    positioned_stats = compute_feature_deltas(
        mixed_rows_by_q=mixed_rows_by_q,
        feature_extractor=feature_set_positioned,
        min_questions=args.min_questions,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )

    rows_out: list[dict[str, object]] = []
    by_token_absmax: dict[str, float] = defaultdict(float)
    for rec in positioned_stats:
        token, position_bin = parse_position_feature(str(rec["feature"]))
        if position_bin not in POSITION_BINS:
            continue
        position_abs = abs(float(rec["mean_delta"]))
        by_token_absmax[token] = max(by_token_absmax[token], position_abs)
        collapsed_mean = collapsed_by_token.get(token, 0.0)
        collapsed_abs = abs(collapsed_mean)
        lift_ratio = (position_abs / collapsed_abs) if collapsed_abs > 0 else 0.0
        rows_out.append(
            {
                "token": token,
                "position_bin": position_bin,
                "feature": rec["feature"],
                "question_presence": rec["question_presence"],
                "questions_used": rec["questions_used"],
                "mean_delta": rec["mean_delta"],
                "median_delta": rec["median_delta"],
                "ci95_low": rec["ci95_low"],
                "ci95_high": rec["ci95_high"],
                "sign_test_p": rec["sign_test_p"],
                "bh_q": rec["bh_q"],
                "sign_pos": rec["sign_pos"],
                "sign_neg": rec["sign_neg"],
                "sign_n_nonzero": rec["sign_n_nonzero"],
                "collapsed_mean_delta": collapsed_mean,
                "collapsed_abs_mean_delta": collapsed_abs,
                "position_abs_mean_delta": position_abs,
                "abs_delta_lift_ratio": lift_ratio,
                "is_ge_lift_threshold": int(lift_ratio >= args.lift_threshold),
            }
        )

    rows_out.sort(
        key=lambda r: (abs(float(r["mean_delta"])), float(r["question_presence"])),
        reverse=True,
    )

    token_lifts: list[dict[str, float]] = []
    for token, max_position_abs in by_token_absmax.items():
        collapsed_abs = abs(collapsed_by_token.get(token, 0.0))
        if collapsed_abs <= 0.0:
            continue
        token_lifts.append(
            {
                "token": token,
                "collapsed_abs": collapsed_abs,
                "max_position_abs": max_position_abs,
                "lift_ratio": max_position_abs / collapsed_abs,
            }
        )

    aggregate = bootstrap_lift_summary(
        token_lifts=token_lifts,
        rng=rng,
        samples=args.bootstrap_samples,
        threshold=args.lift_threshold,
    )

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    columns = [
        "token",
        "position_bin",
        "feature",
        "question_presence",
        "questions_used",
        "mean_delta",
        "median_delta",
        "ci95_low",
        "ci95_high",
        "sign_test_p",
        "bh_q",
        "sign_pos",
        "sign_neg",
        "sign_n_nonzero",
        "collapsed_mean_delta",
        "collapsed_abs_mean_delta",
        "position_abs_mean_delta",
        "abs_delta_lift_ratio",
        "is_ge_lift_threshold",
    ]
    write_csv(outdir / "position_conditioned_deltas.csv", rows_out, columns)
    make_position_lift_plot(outdir / "position_lift_plot.png", token_lifts, threshold=args.lift_threshold)

    summary = {
        "input_csv": str(args.input.resolve()),
        "columns_used": columns_used,
        "n_rows": len(rows),
        "n_questions": len(by_q),
        "n_mixed_outcome_questions": len(mixed_q),
        "min_questions_threshold": args.min_questions,
        "fdr_threshold": args.fdr,
        "lift_threshold": args.lift_threshold,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "n_position_conditioned_features": len(rows_out),
        "n_position_tokens_evaluated_for_lift": aggregate["n_tokens_evaluated"],
        "n_tokens_ge_lift_threshold": aggregate["n_tokens_ge_threshold"],
        "fraction_tokens_ge_lift_threshold": aggregate["fraction_ge_threshold"],
        "fraction_tokens_ge_lift_threshold_ci95_low": aggregate["fraction_ge_threshold_ci95_low"],
        "fraction_tokens_ge_lift_threshold_ci95_high": aggregate["fraction_ge_threshold_ci95_high"],
        "paired_bootstrap_mean_excess_over_threshold": aggregate["mean_excess_over_threshold"],
        "paired_bootstrap_mean_excess_ci95_low": aggregate["mean_excess_ci95_low"],
        "paired_bootstrap_mean_excess_ci95_high": aggregate["mean_excess_ci95_high"],
        "paired_bootstrap_p_excess_le_zero": aggregate["paired_bootstrap_p_excess_le_zero"],
        "top_tokens_by_lift_ratio": sorted(token_lifts, key=lambda d: d["lift_ratio"], reverse=True)[:10],
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote Stage 4 outputs to: {outdir.resolve()}")
    print(f"Mixed-outcome questions |M|: {len(mixed_q)}")
    print(f"Position-conditioned features analyzed: {len(rows_out)}")
    print(
        "Tokens with max position-bin |delta| >= "
        f"{args.lift_threshold:.2f}x collapsed |delta|: "
        f"{aggregate['n_tokens_ge_threshold']}/{aggregate['n_tokens_evaluated']}"
    )


if __name__ == "__main__":
    main()
