#!/usr/bin/env python3
"""Stage 8 / Experiment 7: first-action bag-of-words analysis.

Analyzes whether the first action token in a trace carries predictive signal.

Models (grouped CV by question_id):
- M0: oracle per-question baseline
- M1: has-first-action-only logistic baseline
- M2: first-action token BOW + question fixed effects (L1 logistic)
- M3: first-action qualifier-word BOW + question fixed effects (L1 logistic)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TOKEN_COL_CANDIDATES = ("tokenized_trace", "tokens", "tokenized")
QUESTION_COL_CANDIDATES = ("question_id", "qid", "question")
CORRECT_COL_CANDIDATES = ("is_correct", "correct", "label")


@dataclass
class Row:
    question_id: str
    y: int
    first_action_token: str
    first_action_bow_text: str
    has_first_action: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 8 first-action BOW analysis.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("research/gpt_oss_subset_15q_all_traces_tokenized_gpt-oss-chunked_live_1.csv"),
        help="Tokenized CSV input.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("experiments/exp7_first_action_bow"),
        help="Output directory.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Outer GroupKFold split count.",
    )
    parser.add_argument(
        "--inner-splits",
        type=int,
        default=3,
        help="Inner GroupKFold split count for C tuning.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=73,
        help="Random seed.",
    )
    parser.add_argument(
        "--use-all-questions",
        action="store_true",
        help="Use all questions, not just mixed-outcome questions.",
    )
    parser.add_argument(
        "--min-questions",
        type=int,
        default=2,
        help="Minimum mixed-outcome question presence for first-action token leaderboard.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Bootstrap resamples for 95%% CI in leaderboard deltas.",
    )
    parser.add_argument(
        "--fdr",
        type=float,
        default=0.10,
        help="FDR threshold for leaderboard significance.",
    )
    return parser.parse_args()


def detect_column(fieldnames: list[str], candidates: tuple[str, ...], kind: str) -> str:
    for cand in candidates:
        if cand in fieldnames:
            return cand
    raise ValueError(f"Missing {kind} column. Available: {', '.join(fieldnames)}")


def parse_bool(value: str) -> int:
    text = (value or "").strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return 1
    if text in {"0", "false", "f", "no", "n"}:
        return 0
    raise ValueError(f"Bad label: {value!r}")


def first_action_from_tokens(tokens: list[str]) -> str:
    for tok in tokens:
        if tok.startswith("action:"):
            return tok
    return "__NO_FIRST_ACTION__"


def action_bow_text(action_token: str) -> str:
    if action_token == "__NO_FIRST_ACTION__":
        return "no_first_action"
    qualifier = action_token.split(":", 1)[1] if ":" in action_token else action_token
    words = [w for w in re.split(r"[^a-zA-Z0-9]+", qualifier.lower()) if w]
    return " ".join(words) if words else "unknown"


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        q_col = detect_column(fields, QUESTION_COL_CANDIDATES, "question_id")
        y_col = detect_column(fields, CORRECT_COL_CANDIDATES, "correctness")
        t_col = detect_column(fields, TOKEN_COL_CANDIDATES, "token sequence")
        for record in reader:
            try:
                y = parse_bool(record.get(y_col, ""))
            except ValueError:
                continue
            qid = str(record.get(q_col, "")).strip()
            if not qid:
                continue
            token_text = (record.get(t_col, "") or "").strip()
            tokens = [tok for tok in token_text.split() if tok]
            first_action = first_action_from_tokens(tokens)
            rows.append(
                Row(
                    question_id=qid,
                    y=y,
                    first_action_token=first_action,
                    first_action_bow_text=action_bow_text(first_action),
                    has_first_action=int(first_action != "__NO_FIRST_ACTION__"),
                )
            )
    return rows


def filter_mixed(rows: list[Row]) -> list[Row]:
    by_q: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_q[r.question_id].append(r)
    mixed_q = {q for q, qrows in by_q.items() if {x.y for x in qrows} == {0, 1}}
    return [r for r in rows if r.question_id in mixed_q]


def question_mode_label(labels: np.ndarray) -> int:
    ones = int(np.sum(labels))
    zeros = int(labels.size - ones)
    # deterministic tie-break toward positive class
    return 1 if ones >= zeros else 0


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def ece_score(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    if total == 0:
        return 0.0
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(probs[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / total) * abs(acc - conf)
    return float(ece)


def summarize_model(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    preds = (probs >= 0.5).astype(int)
    return {
        "auc": safe_auc(y, probs),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "ece_10bin": ece_score(y, probs),
    }


def fit_m1(train_has: np.ndarray, train_y: np.ndarray) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    x = scaler.fit_transform(train_has.reshape(-1, 1))
    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=73,
    )
    clf.fit(x, train_y)
    return clf, scaler


def fit_l1(x: csr_matrix, y: np.ndarray, c_value: float) -> LogisticRegression:
    clf = LogisticRegression(
        penalty="l1",
        C=c_value,
        solver="liblinear",
        max_iter=4000,
        random_state=73,
    )
    clf.fit(x, y)
    return clf


def build_features(
    train_text: np.ndarray,
    test_text: np.ndarray,
    train_qids: np.ndarray,
    test_qids: np.ndarray,
) -> tuple[csr_matrix, csr_matrix, list[str], CountVectorizer, OneHotEncoder]:
    vec = CountVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\S+\b",
        ngram_range=(1, 1),
        binary=False,
    )
    x_tr_tok = vec.fit_transform(train_text.tolist())
    x_te_tok = vec.transform(test_text.tolist())
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    x_tr_q = ohe.fit_transform(train_qids.reshape(-1, 1))
    x_te_q = ohe.transform(test_qids.reshape(-1, 1))
    x_tr = hstack([x_tr_tok, x_tr_q], format="csr")
    x_te = hstack([x_te_tok, x_te_q], format="csr")
    names = [f"token::{n}" for n in vec.get_feature_names_out()] + [f"qid::{q}" for q in ohe.categories_[0].tolist()]
    return x_tr, x_te, names, vec, ohe


def tune_c(
    text: np.ndarray,
    y: np.ndarray,
    qids: np.ndarray,
    groups: np.ndarray,
    inner_splits: int,
    c_grid: list[float],
) -> float:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return 1.0
    splits = min(inner_splits, len(unique_groups))
    if splits < 2:
        return 1.0
    gkf = GroupKFold(n_splits=splits)
    best_c = c_grid[0]
    best_score = -math.inf
    for c in c_grid:
        scores: list[float] = []
        for tr_idx, va_idx in gkf.split(text, y, groups):
            x_tr, x_va, _, _, _ = build_features(text[tr_idx], text[va_idx], qids[tr_idx], qids[va_idx])
            clf = fit_l1(x_tr, y[tr_idx], c)
            prob = clf.predict_proba(x_va)[:, 1]
            score = safe_auc(y[va_idx], prob)
            if math.isnan(score):
                score = balanced_accuracy_score(y[va_idx], (prob >= 0.5).astype(int))
            scores.append(float(score))
        mean_score = float(np.mean(scores)) if scores else -math.inf
        if mean_score > best_score:
            best_score = mean_score
            best_c = c
    return best_c


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


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_forest_plot(path: Path, lifesavers: list[dict[str, object]], killers: list[dict[str, object]]) -> None:
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
    ax.set_title("Stage 8 first-action diagnostics")
    ax.invert_yaxis()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def compute_first_action_token_deltas(
    rows: list[Row],
    min_questions: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_q: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_q[r.question_id].append(r)

    token_questions: dict[str, set[str]] = defaultdict(set)
    for qid, qrows in by_q.items():
        present = {r.first_action_token for r in qrows}
        for tok in present:
            token_questions[tok].add(qid)

    rng = random.Random(seed)
    records: list[dict[str, object]] = []
    per_question_rows: list[dict[str, object]] = []
    for token, qids_present in token_questions.items():
        if len(qids_present) < min_questions:
            continue
        deltas: list[float] = []
        baselines: list[float] = []
        qcount = 0
        for qid, qrows in by_q.items():
            with_tok = [r for r in qrows if r.first_action_token == token]
            without_tok = [r for r in qrows if r.first_action_token != token]
            if not with_tok or not without_tok:
                continue
            question_baseline = sum(r.y for r in qrows) / len(qrows)
            succ_with = sum(r.y for r in with_tok) / len(with_tok)
            succ_without = sum(r.y for r in without_tok) / len(without_tok)
            delta_q = succ_with - succ_without
            deltas.append(delta_q)
            baselines.append(question_baseline)
            qcount += 1
            per_question_rows.append(
                {
                    "token": token,
                    "question_id": qid,
                    "question_total_runs": len(qrows),
                    "question_success_count": sum(r.y for r in qrows),
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
                "avg_question_trace_len": 1.0,
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

    records = sorted(records, key=lambda r: float(r["mean_delta"]), reverse=True)
    return records, per_question_rows


def write_first_action_summary(path: Path, rows: list[Row]) -> None:
    ctr = Counter(r.first_action_token for r in rows)
    y_by_action: dict[str, list[int]] = defaultdict(list)
    q_by_action: dict[str, set[str]] = defaultdict(set)
    by_q: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        y_by_action[r.first_action_token].append(r.y)
        q_by_action[r.first_action_token].add(r.question_id)
        by_q[r.question_id].append(r)

    out_rows: list[dict[str, object]] = []
    for action, n in ctr.items():
        deltas: list[float] = []
        for qid, qrows in by_q.items():
            with_a = [x.y for x in qrows if x.first_action_token == action]
            without_a = [x.y for x in qrows if x.first_action_token != action]
            if not with_a or not without_a:
                continue
            deltas.append(float(np.mean(with_a) - np.mean(without_a)))
        mean_delta = float(np.mean(deltas)) if deltas else 0.0
        out_rows.append(
            {
                "first_action": action,
                "count": int(n),
                "fraction": float(n / len(rows)),
                "n_questions": int(len(q_by_action[action])),
                "success_rate": float(np.mean(y_by_action[action])),
                "mean_within_question_delta": mean_delta,
            }
        )
    out_rows.sort(key=lambda r: (-int(r["count"]), str(r["first_action"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "first_action",
                "count",
                "fraction",
                "n_questions",
                "success_rate",
                "mean_within_question_delta",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)


def plot_first_action_success(path: Path, rows: list[Row]) -> None:
    data = defaultdict(list)
    for r in rows:
        data[r.first_action_token].append(r.y)
    items = sorted(data.items(), key=lambda kv: len(kv[1]), reverse=True)[:12]
    labels = [k for k, _ in items]
    rates = [float(np.mean(v)) for _, v in items]
    counts = [len(v) for _, v in items]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, rates, color="#2b8cbe")
    ax.set_xticks(x, labels, rotation=40, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Success rate")
    ax.set_title("Success rate by first action (top 12 by count)")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, str(n), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*'penalty' was deprecated in version 1\.8.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*Inconsistent values: penalty=l1 with l1_ratio=0\.0.*",
    )
    rows = load_rows(args.input)
    if not args.use_all_questions:
        rows = filter_mixed(rows)
    if not rows:
        raise ValueError("No rows available after filtering.")

    qids = np.array([r.question_id for r in rows], dtype=object)
    y = np.array([r.y for r in rows], dtype=int)
    first_action_token_text = np.array([r.first_action_token for r in rows], dtype=object)
    first_action_word_text = np.array([r.first_action_bow_text for r in rows], dtype=object)
    has_action = np.array([r.has_first_action for r in rows], dtype=float)

    unique_q = np.unique(qids)
    n_splits = min(args.n_splits, len(unique_q))
    if n_splits < 2:
        raise ValueError("Need at least 2 unique questions for GroupKFold.")
    outer = GroupKFold(n_splits=n_splits)
    c_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

    oof_m0 = np.zeros(len(rows), dtype=float)
    oof_m1 = np.zeros(len(rows), dtype=float)
    oof_m2 = np.zeros(len(rows), dtype=float)
    oof_m3 = np.zeros(len(rows), dtype=float)
    oof_shuf = np.zeros(len(rows), dtype=float)
    fold_rows: list[dict[str, object]] = []

    q_mode = {q: float(question_mode_label(y[qids == q])) for q in unique_q}
    y_shuf = np.random.default_rng(args.seed).permutation(y.copy())

    for fold_idx, (tr_idx, te_idx) in enumerate(outer.split(first_action_token_text, y, groups=qids), start=1):
        tr_y, te_y = y[tr_idx], y[te_idx]
        tr_q, te_q = qids[tr_idx], qids[te_idx]

        # M0
        m0_prob = np.array([q_mode[q] for q in te_q], dtype=float)
        oof_m0[te_idx] = m0_prob

        # M1 has-first-action only
        clf1, sc1 = fit_m1(has_action[tr_idx], tr_y)
        m1_prob = clf1.predict_proba(sc1.transform(has_action[te_idx].reshape(-1, 1)))[:, 1]
        oof_m1[te_idx] = m1_prob

        # M2 first-action token + qid FE
        c_m2 = tune_c(
            text=first_action_token_text[tr_idx],
            y=tr_y,
            qids=tr_q,
            groups=tr_q,
            inner_splits=args.inner_splits,
            c_grid=c_grid,
        )
        x2_tr, x2_te, _, _, _ = build_features(first_action_token_text[tr_idx], first_action_token_text[te_idx], tr_q, te_q)
        clf2 = fit_l1(x2_tr, tr_y, c_m2)
        m2_prob = clf2.predict_proba(x2_te)[:, 1]
        oof_m2[te_idx] = m2_prob

        # M3 first-action qualifier words + qid FE
        c_m3 = tune_c(
            text=first_action_word_text[tr_idx],
            y=tr_y,
            qids=tr_q,
            groups=tr_q,
            inner_splits=args.inner_splits,
            c_grid=c_grid,
        )
        x3_tr, x3_te, _, _, _ = build_features(first_action_word_text[tr_idx], first_action_word_text[te_idx], tr_q, te_q)
        clf3 = fit_l1(x3_tr, tr_y, c_m3)
        m3_prob = clf3.predict_proba(x3_te)[:, 1]
        oof_m3[te_idx] = m3_prob

        # Shuffled-label sanity on M2
        c_shuf = tune_c(
            text=first_action_token_text[tr_idx],
            y=y_shuf[tr_idx],
            qids=tr_q,
            groups=tr_q,
            inner_splits=args.inner_splits,
            c_grid=c_grid,
        )
        xs_tr, xs_te, _, _, _ = build_features(first_action_token_text[tr_idx], first_action_token_text[te_idx], tr_q, te_q)
        clf_s = fit_l1(xs_tr, y_shuf[tr_idx], c_shuf)
        oof_shuf[te_idx] = clf_s.predict_proba(xs_te)[:, 1]

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "m2_best_c": float(c_m2),
                "m3_best_c": float(c_m3),
                "m0_auc": safe_auc(te_y, m0_prob),
                "m1_auc": safe_auc(te_y, m1_prob),
                "m2_auc": safe_auc(te_y, m2_prob),
                "m3_auc": safe_auc(te_y, m3_prob),
            }
        )

    models = {
        "M0_question_mode_baseline": summarize_model(y, oof_m0),
        "M1_has_first_action_only": summarize_model(y, oof_m1),
        "M2_first_action_token_plus_qid_fe": summarize_model(y, oof_m2),
        "M3_first_action_word_bow_plus_qid_fe": summarize_model(y, oof_m3),
    }
    models["AUC_lift_vs_M0"] = {
        "M1_minus_M0": float(models["M1_has_first_action_only"]["auc"] - models["M0_question_mode_baseline"]["auc"]),
        "M2_minus_M0": float(
            models["M2_first_action_token_plus_qid_fe"]["auc"] - models["M0_question_mode_baseline"]["auc"]
        ),
        "M3_minus_M0": float(
            models["M3_first_action_word_bow_plus_qid_fe"]["auc"] - models["M0_question_mode_baseline"]["auc"]
        ),
    }

    # Full-fit coefficients for interpretability.
    c_m2_full = tune_c(
        text=first_action_token_text,
        y=y,
        qids=qids,
        groups=qids,
        inner_splits=args.inner_splits,
        c_grid=c_grid,
    )
    x2_full, _, names2, _, _ = build_features(first_action_token_text, first_action_token_text, qids, qids)
    final_m2 = fit_l1(x2_full, y, c_m2_full)

    c_m3_full = tune_c(
        text=first_action_word_text,
        y=y,
        qids=qids,
        groups=qids,
        inner_splits=args.inner_splits,
        c_grid=c_grid,
    )
    x3_full, _, names3, _, _ = build_features(first_action_word_text, first_action_word_text, qids, qids)
    final_m3 = fit_l1(x3_full, y, c_m3_full)

    coef_rows: list[dict[str, object]] = []
    for model_name, names, coefs in (
        ("M2_first_action_token_plus_qid_fe", names2, final_m2.coef_.ravel()),
        ("M3_first_action_word_bow_plus_qid_fe", names3, final_m3.coef_.ravel()),
    ):
        for n, c in zip(names, coefs):
            c = float(c)
            if abs(c) < 1e-12:
                continue
            ftype, feat = n.split("::", 1)
            coef_rows.append(
                {
                    "model": model_name,
                    "feature_type": ftype,
                    "feature": feat,
                    "coefficient": c,
                    "abs_coefficient": abs(c),
                }
            )
    coef_rows.sort(key=lambda r: (r["model"], -float(r["abs_coefficient"])))

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "cv_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_csv": str(args.input.resolve()),
                "n_rows": int(len(rows)),
                "n_questions": int(len(unique_q)),
                "used_mixed_outcome_only": (not args.use_all_questions),
                "outer_groupkfold_splits": int(n_splits),
                "inner_groupkfold_splits": int(min(args.inner_splits, len(unique_q))),
                "models": models,
                "fold_details": fold_rows,
                "sanity_checks": {"m2_shuffled_label_auc": safe_auc(y_shuf, oof_shuf)},
                "selected_C": {"M2_full_fit": c_m2_full, "M3_full_fit": c_m3_full},
            },
            handle,
            indent=2,
        )

    with (outdir / "coefficients.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "feature_type", "feature", "coefficient", "abs_coefficient"],
        )
        writer.writeheader()
        writer.writerows(coef_rows)

    write_first_action_summary(outdir / "first_action_summary.csv", rows)
    plot_first_action_success(outdir / "first_action_success_rates.png", rows)

    token_stats, per_question_rows = compute_first_action_token_deltas(
        rows=rows,
        min_questions=args.min_questions,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    lifesavers = [r for r in token_stats if float(r["mean_delta"]) > 0 and float(r["bh_q"]) <= args.fdr]
    killers = [r for r in token_stats if float(r["mean_delta"]) < 0 and float(r["bh_q"]) <= args.fdr]
    killers = sorted(killers, key=lambda r: float(r["mean_delta"]))
    delta_columns = [
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
    write_csv(outdir / "token_deltas_all.csv", token_stats, delta_columns)
    write_csv(outdir / "leaderboard_lifesavers.csv", lifesavers, delta_columns)
    write_csv(outdir / "leaderboard_killers.csv", killers, delta_columns)
    write_csv(
        outdir / "per_question_token_deltas.csv",
        per_question_rows,
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
    make_forest_plot(outdir / "forest_plot.png", lifesavers=lifesavers, killers=killers)

    print(f"Wrote Stage 8 outputs to: {outdir.resolve()}")
    print(f"Rows: {len(rows)} | Questions: {len(unique_q)}")
    print(f"M0 AUC: {models['M0_question_mode_baseline']['auc']:.4f}")
    print(f"M2 AUC: {models['M2_first_action_token_plus_qid_fe']['auc']:.4f}")
    print(f"M3 AUC: {models['M3_first_action_word_bow_plus_qid_fe']['auc']:.4f}")
    print(f"Lifesavers (q <= {args.fdr}): {len(lifesavers)}")
    print(f"Killers (q <= {args.fdr}): {len(killers)}")


if __name__ == "__main__":
    main()
