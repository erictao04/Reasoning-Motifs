#!/usr/bin/env python3
"""Stage 2 / Experiment 1: within-question single-token delta analysis.

This script implements the Stage 2 protocol from docs/research_plan.md:
1) Restrict to mixed-outcome questions.
2) Compute per-question delta for each token.
3) Aggregate with sign test, bootstrap CI, and BH q-values.
4) Run a within-question permutation sanity check.
5) Run token count bucket analysis (0 vs 1 vs >=2).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TOKEN_COL_CANDIDATES = ("tokenized_trace", "tokens", "tokenized")
QUESTION_COL_CANDIDATES = ("question_id", "qid", "question")
CORRECT_COL_CANDIDATES = ("is_correct", "correct", "label")


@dataclass
class TraceRow:
    question_id: str
    is_correct: int
    tokens: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 2 single-token within-question delta analysis."
    )
    parser.add_argument("--input", type=Path, required=True, help="Tokenized CSV input.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("research/stage2_exp1_single_token_delta/results"),
        help="Output directory for Stage 2 artifacts.",
    )
    parser.add_argument(
        "--min-questions",
        type=int,
        default=10,
        help="Minimum mixed-outcome question presence for a token.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="Bootstrap resamples for 95%% CI.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=500,
        help="Number of within-question label shuffles for null sanity check.",
    )
    parser.add_argument(
        "--fdr",
        type=float,
        default=0.10,
        help="FDR threshold for leaderboard significance.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=73,
        help="Random seed for bootstrap/permutation.",
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
            tokens = parse_tokens(row.get(token_col, ""))
            rows.append(
                TraceRow(
                    question_id=str(row.get(question_col, "")).strip(),
                    is_correct=is_correct,
                    tokens=tokens,
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


def bootstrap_ci_mean(deltas: list[float], rng: random.Random, samples: int) -> tuple[float, float]:
    if not deltas:
        return (0.0, 0.0)
    n = len(deltas)
    means: list[float] = []
    for _ in range(samples):
        draw = [deltas[rng.randrange(n)] for _ in range(n)]
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


def linear_residuals(xs: list[float], ys: list[float]) -> list[float]:
    if not xs:
        return []
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        return [y - mean_y for y in ys]
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    b = cov_xy / var_x
    a = mean_y - b * mean_x
    return [y - (a + b * x) for x, y in zip(xs, ys)]


def compute_token_deltas(
    mixed_rows_by_q: dict[str, list[TraceRow]],
    min_questions: int,
    bootstrap_samples: int,
    rng: random.Random,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    token_questions: dict[str, set[str]] = defaultdict(set)
    for qid, qrows in mixed_rows_by_q.items():
        present = {tok for r in qrows for tok in set(r.tokens)}
        for tok in present:
            token_questions[tok].add(qid)

    records: list[dict[str, object]] = []
    per_question_rows: list[dict[str, object]] = []
    for token, qids_present in token_questions.items():
        if len(qids_present) < min_questions:
            continue
        deltas: list[float] = []
        baselines: list[float] = []
        qcount = 0
        avg_len_acc = 0.0
        for qid, qrows in mixed_rows_by_q.items():
            with_tok = [r for r in qrows if token in set(r.tokens)]
            without_tok = [r for r in qrows if token not in set(r.tokens)]
            if not with_tok or not without_tok:
                continue
            question_baseline = sum(r.is_correct for r in qrows) / len(qrows)
            succ_with = sum(r.is_correct for r in with_tok) / len(with_tok)
            succ_without = sum(r.is_correct for r in without_tok) / len(without_tok)
            delta_q = succ_with - succ_without
            deltas.append(delta_q)
            baselines.append(question_baseline)
            qcount += 1
            avg_len_acc += sum(len(r.tokens) for r in qrows) / len(qrows)
            per_question_rows.append(
                {
                    "token": token,
                    "question_id": qid,
                    "question_total_runs": len(qrows),
                    "question_success_count": sum(r.is_correct for r in qrows),
                    "question_baseline_success_rate": question_baseline,
                    "n_with_token": len(with_tok),
                    "n_without_token": len(without_tok),
                    "succ_with_token": succ_with,
                    "succ_without_token": succ_without,
                    "delta_q": delta_q,
                }
            )
        if not deltas:
            continue
        mean_delta = sum(deltas) / len(deltas)
        median_delta = statistics.median(deltas)
        mean_baseline = sum(baselines) / len(baselines)
        median_baseline = statistics.median(baselines)
        ci_lo, ci_hi = bootstrap_ci_mean(deltas, rng, bootstrap_samples)
        p, pos, neg, n_nonzero = sign_test_p_value(deltas)
        avg_trace_len = avg_len_acc / qcount if qcount else 0.0
        records.append(
            {
                "token": token,
                "question_presence": len(qids_present),
                "questions_used": qcount,
                "mean_delta": mean_delta,
                "median_delta": median_delta,
                "mean_question_baseline": mean_baseline,
                "median_question_baseline": median_baseline,
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
                "sign_test_p": p,
                "sign_pos": pos,
                "sign_neg": neg,
                "sign_n_nonzero": n_nonzero,
                "avg_question_trace_len": avg_trace_len,
            }
        )

    pvals = [float(r["sign_test_p"]) for r in records]
    qvals = benjamini_hochberg(pvals)
    for rec, q in zip(records, qvals):
        rec["bh_q"] = q

    xs = [float(r["avg_question_trace_len"]) for r in records]
    ys = [float(r["mean_delta"]) for r in records]
    residuals = linear_residuals(xs, ys)
    for rec, resid in zip(records, residuals):
        rec["length_control_residual"] = resid
    return records, per_question_rows


def compute_count_bucket_effects(
    mixed_rows_by_q: dict[str, list[TraceRow]],
    min_questions: int,
) -> list[dict[str, object]]:
    token_questions: dict[str, set[str]] = defaultdict(set)
    for qid, qrows in mixed_rows_by_q.items():
        present = {tok for r in qrows for tok in set(r.tokens)}
        for tok in present:
            token_questions[tok].add(qid)

    results: list[dict[str, object]] = []
    for token, qids_present in token_questions.items():
        if len(qids_present) < min_questions:
            continue
        d1: list[float] = []
        d2: list[float] = []
        n1 = 0
        n2 = 0
        for _, qrows in mixed_rows_by_q.items():
            b0 = [r for r in qrows if Counter(r.tokens)[token] == 0]
            b1 = [r for r in qrows if Counter(r.tokens)[token] == 1]
            b2 = [r for r in qrows if Counter(r.tokens)[token] >= 2]
            if b0 and b1:
                s0 = sum(r.is_correct for r in b0) / len(b0)
                s1 = sum(r.is_correct for r in b1) / len(b1)
                d1.append(s1 - s0)
                n1 += 1
            if b0 and b2:
                s0 = sum(r.is_correct for r in b0) / len(b0)
                s2 = sum(r.is_correct for r in b2) / len(b2)
                d2.append(s2 - s0)
                n2 += 1
        if not d1 and not d2:
            continue
        p1, _, _, _ = sign_test_p_value(d1) if d1 else (1.0, 0, 0, 0)
        p2, _, _, _ = sign_test_p_value(d2) if d2 else (1.0, 0, 0, 0)
        results.append(
            {
                "token": token,
                "question_presence": len(qids_present),
                "n_q_1_vs_0": n1,
                "mean_delta_1_vs_0": (sum(d1) / len(d1)) if d1 else 0.0,
                "median_delta_1_vs_0": statistics.median(d1) if d1 else 0.0,
                "p_1_vs_0": p1,
                "n_q_2plus_vs_0": n2,
                "mean_delta_2plus_vs_0": (sum(d2) / len(d2)) if d2 else 0.0,
                "median_delta_2plus_vs_0": statistics.median(d2) if d2 else 0.0,
                "p_2plus_vs_0": p2,
            }
        )
    pvals = [float(r["p_1_vs_0"]) for r in results] + [float(r["p_2plus_vs_0"]) for r in results]
    qvals = benjamini_hochberg(pvals)
    half = len(results)
    for i, rec in enumerate(results):
        rec["q_1_vs_0"] = qvals[i]
        rec["q_2plus_vs_0"] = qvals[i + half]
    return results


def permutation_null(
    mixed_rows_by_q: dict[str, list[TraceRow]],
    base_token_stats: list[dict[str, object]],
    n_perm: int,
    min_questions: int,
    rng: random.Random,
) -> dict[str, object]:
    tracked_tokens = [str(r["token"]) for r in base_token_stats]
    max_abs_deltas: list[float] = []
    top_token_counts: Counter[str] = Counter()

    question_ids = list(mixed_rows_by_q.keys())
    for _ in range(n_perm):
        shuffled: dict[str, list[TraceRow]] = {}
        for qid in question_ids:
            qrows = mixed_rows_by_q[qid]
            labels = [r.is_correct for r in qrows]
            rng.shuffle(labels)
            shuffled[qid] = [
                TraceRow(question_id=r.question_id, is_correct=labels[i], tokens=r.tokens)
                for i, r in enumerate(qrows)
            ]
        perm_stats, _ = compute_token_deltas(
            shuffled,
            min_questions=min_questions,
            bootstrap_samples=200,
            rng=rng,
        )
        keep = [r for r in perm_stats if str(r["token"]) in tracked_tokens]
        if not keep:
            max_abs_deltas.append(0.0)
            continue
        top = max(keep, key=lambda r: abs(float(r["mean_delta"])))
        max_abs_deltas.append(abs(float(top["mean_delta"])))
        top_token_counts[str(top["token"])] += 1

    max_abs_deltas.sort()
    return {
        "n_permutations": n_perm,
        "max_abs_mean_delta": {
            "mean": (sum(max_abs_deltas) / len(max_abs_deltas)) if max_abs_deltas else 0.0,
            "p95": max_abs_deltas[int(0.95 * (len(max_abs_deltas) - 1))] if max_abs_deltas else 0.0,
            "max": max(max_abs_deltas) if max_abs_deltas else 0.0,
        },
        "most_frequent_top_tokens": top_token_counts.most_common(20),
    }


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_forest_plot(path: Path, lifesavers: list[dict[str, object]], killers: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    top_life = sorted(lifesavers, key=lambda r: float(r["mean_delta"]), reverse=True)[:15]
    top_kill = sorted(killers, key=lambda r: float(r["mean_delta"]))[:15]
    rows = top_life + top_kill
    if not rows:
        return
    labels = [str(r["token"]) for r in rows]
    means = [float(r["mean_delta"]) for r in rows]
    lo = [float(r["ci95_low"]) for r in rows]
    hi = [float(r["ci95_high"]) for r in rows]
    left_err = [m - l for m, l in zip(means, lo)]
    right_err = [h - m for m, h in zip(means, hi)]
    y = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.errorbar(means, y, xerr=[left_err, right_err], fmt="o", capsize=3)
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean within-question delta (success with token - without token)")
    ax.set_title("Stage 2 single-token diagnostics")
    ax.invert_yaxis()
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

    token_stats, per_question_token_rows = compute_token_deltas(
        mixed_rows_by_q=mixed_rows_by_q,
        min_questions=args.min_questions,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    token_stats_sorted = sorted(token_stats, key=lambda r: float(r["mean_delta"]), reverse=True)

    lifesavers = [r for r in token_stats_sorted if float(r["mean_delta"]) > 0 and float(r["bh_q"]) <= args.fdr]
    killers = [r for r in token_stats_sorted if float(r["mean_delta"]) < 0 and float(r["bh_q"]) <= args.fdr]
    killers = sorted(killers, key=lambda r: float(r["mean_delta"]))

    bucket_stats = compute_count_bucket_effects(
        mixed_rows_by_q=mixed_rows_by_q,
        min_questions=args.min_questions,
    )
    bucket_stats = sorted(bucket_stats, key=lambda r: float(r["mean_delta_2plus_vs_0"]))

    perm = permutation_null(
        mixed_rows_by_q=mixed_rows_by_q,
        base_token_stats=token_stats_sorted,
        n_perm=args.permutations,
        min_questions=args.min_questions,
        rng=rng,
    )

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    columns = [
        "token",
        "question_presence",
        "questions_used",
        "mean_delta",
        "median_delta",
        "mean_question_baseline",
        "median_question_baseline",
        "ci95_low",
        "ci95_high",
        "sign_test_p",
        "bh_q",
        "sign_pos",
        "sign_neg",
        "sign_n_nonzero",
        "avg_question_trace_len",
        "length_control_residual",
    ]
    write_csv(outdir / "token_deltas_all.csv", token_stats_sorted, columns)
    write_csv(outdir / "leaderboard_lifesavers.csv", lifesavers, columns)
    write_csv(outdir / "leaderboard_killers.csv", killers, columns)
    write_csv(
        outdir / "per_question_token_deltas.csv",
        per_question_token_rows,
        [
            "token",
            "question_id",
            "question_total_runs",
            "question_success_count",
            "question_baseline_success_rate",
            "n_with_token",
            "n_without_token",
            "succ_with_token",
            "succ_without_token",
            "delta_q",
        ],
    )
    write_csv(
        outdir / "question_baselines.csv",
        [
            {
                "question_id": qid,
                "question_total_runs": len(qrows),
                "question_success_count": sum(r.is_correct for r in qrows),
                "question_baseline_success_rate": sum(r.is_correct for r in qrows) / len(qrows),
            }
            for qid, qrows in sorted(mixed_rows_by_q.items(), key=lambda kv: kv[0])
        ],
        [
            "question_id",
            "question_total_runs",
            "question_success_count",
            "question_baseline_success_rate",
        ],
    )

    bucket_columns = [
        "token",
        "question_presence",
        "n_q_1_vs_0",
        "mean_delta_1_vs_0",
        "median_delta_1_vs_0",
        "p_1_vs_0",
        "q_1_vs_0",
        "n_q_2plus_vs_0",
        "mean_delta_2plus_vs_0",
        "median_delta_2plus_vs_0",
        "p_2plus_vs_0",
        "q_2plus_vs_0",
    ]
    write_csv(outdir / "count_bucket_effects.csv", bucket_stats, bucket_columns)

    with (outdir / "permutation_null.json").open("w", encoding="utf-8") as handle:
        json.dump(perm, handle, indent=2)

    make_forest_plot(outdir / "forest_plot.png", lifesavers=lifesavers, killers=killers)

    summary = {
        "input_csv": str(args.input.resolve()),
        "columns_used": columns_used,
        "n_rows": len(rows),
        "n_questions": len(by_q),
        "n_mixed_outcome_questions": len(mixed_q),
        "mean_question_baseline_over_mixed": (
            sum(sum(r.is_correct for r in qrows) / len(qrows) for qrows in mixed_rows_by_q.values())
            / len(mixed_rows_by_q)
            if mixed_rows_by_q
            else 0.0
        ),
        "token_threshold_min_questions": args.min_questions,
        "n_tokens_analyzed": len(token_stats_sorted),
        "fdr_threshold": args.fdr,
        "n_lifesaver_tokens": len(lifesavers),
        "n_killer_tokens": len(killers),
        "bootstrap_samples": args.bootstrap_samples,
        "permutations": args.permutations,
        "seed": args.seed,
        "top_3_lifesavers": [
            {
                "token": r["token"],
                "mean_delta": r["mean_delta"],
                "ci95_low": r["ci95_low"],
                "ci95_high": r["ci95_high"],
                "bh_q": r["bh_q"],
            }
            for r in lifesavers[:3]
        ],
        "top_3_killers": [
            {
                "token": r["token"],
                "mean_delta": r["mean_delta"],
                "ci95_low": r["ci95_low"],
                "ci95_high": r["ci95_high"],
                "bh_q": r["bh_q"],
            }
            for r in killers[:3]
        ],
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote Stage 2 outputs to: {outdir.resolve()}")
    print(f"Mixed-outcome questions |M|: {len(mixed_q)}")
    print(f"Analyzed tokens: {len(token_stats_sorted)}")
    print(f"Lifesavers (q <= {args.fdr}): {len(lifesavers)}")
    print(f"Killers (q <= {args.fdr}): {len(killers)}")


if __name__ == "__main__":
    main()
