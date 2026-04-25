#!/usr/bin/env python3
"""Stage 7 / Experiment 6: prefix-based early prediction (failure forecasting).

Implements grouped-CV prediction using partial prefixes of tokenized traces.

Prefix levels:
- p in {0.25, 0.50, 0.75, 1.00}

Models per prefix p:
- M0_p: oracle per-question baseline
- M1_p: logistic regression on prefix length only
- M2_p: L1 logistic regression on token counts + question fixed effects
- M3_p: M2_p + bigrams

Main outputs:
- prefix_metrics.json
- auc_vs_prefix.png
- early_token_leaderboard.csv (optional)
- time_to_detection.json (optional)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import defaultdict
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
class TraceRow:
    trace_id: str
    question_id: str
    y: int
    tokens: list[str]


@dataclass
class PrefixRow:
    trace_id: str
    question_id: str
    y: int
    prefix_fraction: float
    prefix_length: int
    original_length: int
    token_text: str
    tokens: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 7 prefix prediction experiment.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "tokenizer/clean_expanded_pool_s100_seed73_qwen25_7b_hot30_tokenized_gpt-oss-120b_0.csv"
        ),
        help="Tokenized CSV input.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("experiments/exp6_prefix_prediction"),
        help="Output directory.",
    )
    parser.add_argument(
        "--prefix-fractions",
        type=str,
        default="0.25,0.50,0.75,1.00",
        help="Comma-separated prefix fractions (0,1].",
    )
    parser.add_argument(
        "--early-prefix-fractions",
        type=str,
        default="0.25,0.50",
        help="Comma-separated early-prefix fractions for optional token leaderboard.",
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
        "--disable-optional-analysis",
        action="store_true",
        help="Disable early-token leaderboard and time-to-detection outputs.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.70,
        help="Confidence threshold for time-to-detection analysis.",
    )
    parser.add_argument(
        "--early-min-questions",
        type=int,
        default=10,
        help="Minimum #questions in which token must appear for early-token leaderboard.",
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


def parse_fraction_list(text: str) -> list[float]:
    values: list[float] = []
    for part in text.split(","):
        s = part.strip()
        if not s:
            continue
        val = float(s)
        if not (0.0 < val <= 1.0):
            raise ValueError(f"Prefix fraction must be in (0, 1], got: {val}")
        values.append(float(val))
    if not values:
        raise ValueError("At least one prefix fraction is required.")
    return sorted(set(values))


def load_rows(path: Path) -> list[TraceRow]:
    rows: list[TraceRow] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        q_col = detect_column(fields, QUESTION_COL_CANDIDATES, "question_id")
        y_col = detect_column(fields, CORRECT_COL_CANDIDATES, "correctness")
        t_col = detect_column(fields, TOKEN_COL_CANDIDATES, "token sequence")

        for i, r in enumerate(reader):
            try:
                y = parse_bool(r.get(y_col, ""))
            except ValueError:
                continue
            qid = str(r.get(q_col, "")).strip()
            if not qid:
                continue
            token_text = (r.get(t_col, "") or "").strip()
            tokens = [tok for tok in token_text.split() if tok]
            rows.append(
                TraceRow(
                    trace_id=f"trace_{i}",
                    question_id=qid,
                    y=y,
                    tokens=tokens,
                )
            )
    return rows


def filter_mixed(rows: list[TraceRow]) -> list[TraceRow]:
    by_q: dict[str, list[TraceRow]] = defaultdict(list)
    for r in rows:
        by_q[r.question_id].append(r)
    mixed_q = {q for q, qrows in by_q.items() if {row.y for row in qrows} == {0, 1}}
    return [r for r in rows if r.question_id in mixed_q]


def build_prefix_rows(rows: list[TraceRow], prefix_fraction: float) -> list[PrefixRow]:
    out: list[PrefixRow] = []
    for r in rows:
        orig_len = len(r.tokens)
        pref_len = int(math.ceil(prefix_fraction * orig_len))
        pref_tokens = r.tokens[:pref_len]
        out.append(
            PrefixRow(
                trace_id=r.trace_id,
                question_id=r.question_id,
                y=r.y,
                prefix_fraction=prefix_fraction,
                prefix_length=pref_len,
                original_length=orig_len,
                token_text=" ".join(pref_tokens),
                tokens=pref_tokens,
            )
        )
    return out


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


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def build_token_and_qid_features(
    train_text: np.ndarray,
    test_text: np.ndarray,
    train_qids: np.ndarray,
    test_qids: np.ndarray,
    ngram_range: tuple[int, int],
) -> tuple[csr_matrix, csr_matrix]:
    vec = CountVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\S+\b",
        ngram_range=ngram_range,
        binary=False,
    )
    x_train_tok = vec.fit_transform(train_text.tolist())
    x_test_tok = vec.transform(test_text.tolist())

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    x_train_q = ohe.fit_transform(train_qids.reshape(-1, 1))
    x_test_q = ohe.transform(test_qids.reshape(-1, 1))
    return hstack([x_train_tok, x_train_q], format="csr"), hstack([x_test_tok, x_test_q], format="csr")


def fit_m1_length(train_len: np.ndarray, train_y: np.ndarray) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    x = scaler.fit_transform(train_len.reshape(-1, 1))
    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=73,
    )
    clf.fit(x, train_y)
    return clf, scaler


def fit_l1_logistic(x: csr_matrix, y: np.ndarray, c_value: float) -> LogisticRegression:
    clf = LogisticRegression(
        penalty="l1",
        C=c_value,
        solver="liblinear",
        max_iter=4000,
        random_state=73,
    )
    clf.fit(x, y)
    return clf


def tune_c(
    text: np.ndarray,
    y: np.ndarray,
    qids: np.ndarray,
    groups: np.ndarray,
    ngram_range: tuple[int, int],
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
        fold_scores: list[float] = []
        for tr_idx, va_idx in gkf.split(text, y, groups):
            x_tr, x_va = build_token_and_qid_features(
                text[tr_idx],
                text[va_idx],
                qids[tr_idx],
                qids[va_idx],
                ngram_range=ngram_range,
            )
            clf = fit_l1_logistic(x_tr, y[tr_idx], c_value=c)
            va_prob = clf.predict_proba(x_va)[:, 1]
            score = safe_auc(y[va_idx], va_prob)
            if math.isnan(score):
                score = balanced_accuracy_score(y[va_idx], (va_prob >= 0.5).astype(int))
            fold_scores.append(float(score))
        mean_score = float(np.mean(fold_scores)) if fold_scores else -math.inf
        if mean_score > best_score:
            best_score = mean_score
            best_c = c
    return best_c


def summarize_model(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    preds = (probs >= 0.5).astype(int)
    return {
        "auc": safe_auc(y, probs),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "ece_10bin": ece_score(y, probs, n_bins=10),
    }


def shuffled_prefix_text(token_text: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out: list[str] = []
    for text in token_text.tolist():
        toks = [t for t in text.split() if t]
        if len(toks) <= 1:
            out.append(" ".join(toks))
            continue
        shuffled = toks.copy()
        rng.shuffle(shuffled)
        out.append(" ".join(shuffled))
    return np.array(out, dtype=object)


def evaluate_prefix(
    prefix_rows: list[PrefixRow],
    n_splits: int,
    inner_splits: int,
    c_grid: list[float],
    seed: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    qids = np.array([r.question_id for r in prefix_rows], dtype=object)
    y = np.array([r.y for r in prefix_rows], dtype=int)
    token_text = np.array([r.token_text for r in prefix_rows], dtype=object)
    token_len = np.array([r.prefix_length for r in prefix_rows], dtype=float)
    trace_ids = np.array([r.trace_id for r in prefix_rows], dtype=object)
    orig_len = np.array([r.original_length for r in prefix_rows], dtype=int)
    prefix_fraction = float(prefix_rows[0].prefix_fraction)

    unique_q = np.unique(qids)
    split_count = min(n_splits, len(unique_q))
    if split_count < 2:
        raise ValueError("Need at least 2 unique questions for GroupKFold.")

    oof_m0 = np.zeros(len(prefix_rows), dtype=float)
    oof_m1 = np.zeros(len(prefix_rows), dtype=float)
    oof_m2 = np.zeros(len(prefix_rows), dtype=float)
    oof_m3 = np.zeros(len(prefix_rows), dtype=float)
    oof_m2_shuf = np.zeros(len(prefix_rows), dtype=float)
    oof_m3_rand = np.zeros(len(prefix_rows), dtype=float)

    q_baseline = {q: float(np.mean(y[qids == q])) for q in unique_q}
    outer = GroupKFold(n_splits=split_count)
    fold_rows: list[dict[str, object]] = []

    token_text_rand = shuffled_prefix_text(token_text, seed=seed + int(prefix_fraction * 1000))
    y_shuf = np.random.default_rng(seed + 1 + int(prefix_fraction * 1000)).permutation(y.copy())

    for fold_idx, (tr_idx, te_idx) in enumerate(outer.split(token_text, y, groups=qids), start=1):
        tr_text, te_text = token_text[tr_idx], token_text[te_idx]
        tr_y, te_y = y[tr_idx], y[te_idx]
        tr_q, te_q = qids[tr_idx], qids[te_idx]
        tr_len, te_len = token_len[tr_idx], token_len[te_idx]

        # M0
        m0_prob = np.array([q_baseline[q] for q in te_q], dtype=float)
        oof_m0[te_idx] = m0_prob

        # M1
        m1_clf, m1_scaler = fit_m1_length(tr_len, tr_y)
        m1_prob = m1_clf.predict_proba(m1_scaler.transform(te_len.reshape(-1, 1)))[:, 1]
        oof_m1[te_idx] = m1_prob

        # M2
        c_m2 = tune_c(
            text=tr_text,
            y=tr_y,
            qids=tr_q,
            groups=tr_q,
            ngram_range=(1, 1),
            inner_splits=inner_splits,
            c_grid=c_grid,
        )
        x2_tr, x2_te = build_token_and_qid_features(tr_text, te_text, tr_q, te_q, ngram_range=(1, 1))
        m2 = fit_l1_logistic(x2_tr, tr_y, c_value=c_m2)
        m2_prob = m2.predict_proba(x2_te)[:, 1]
        oof_m2[te_idx] = m2_prob

        # M3
        c_m3 = tune_c(
            text=tr_text,
            y=tr_y,
            qids=tr_q,
            groups=tr_q,
            ngram_range=(1, 2),
            inner_splits=inner_splits,
            c_grid=c_grid,
        )
        x3_tr, x3_te = build_token_and_qid_features(tr_text, te_text, tr_q, te_q, ngram_range=(1, 2))
        m3 = fit_l1_logistic(x3_tr, tr_y, c_value=c_m3)
        m3_prob = m3.predict_proba(x3_te)[:, 1]
        oof_m3[te_idx] = m3_prob

        # Shuffled-label sanity for M2
        c_m2_shuf = tune_c(
            text=tr_text,
            y=y_shuf[tr_idx],
            qids=tr_q,
            groups=tr_q,
            ngram_range=(1, 1),
            inner_splits=inner_splits,
            c_grid=c_grid,
        )
        x2s_tr, x2s_te = build_token_and_qid_features(tr_text, te_text, tr_q, te_q, ngram_range=(1, 1))
        m2_shuf = fit_l1_logistic(x2s_tr, y_shuf[tr_idx], c_value=c_m2_shuf)
        oof_m2_shuf[te_idx] = m2_shuf.predict_proba(x2s_te)[:, 1]

        # Prefix-randomization sanity for M3 (order shuffled within prefix)
        tr_rand, te_rand = token_text_rand[tr_idx], token_text_rand[te_idx]
        c_m3_rand = tune_c(
            text=tr_rand,
            y=tr_y,
            qids=tr_q,
            groups=tr_q,
            ngram_range=(1, 2),
            inner_splits=inner_splits,
            c_grid=c_grid,
        )
        x3r_tr, x3r_te = build_token_and_qid_features(tr_rand, te_rand, tr_q, te_q, ngram_range=(1, 2))
        m3_rand = fit_l1_logistic(x3r_tr, tr_y, c_value=c_m3_rand)
        oof_m3_rand[te_idx] = m3_rand.predict_proba(x3r_te)[:, 1]

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "m2_best_c": float(c_m2),
                "m3_best_c": float(c_m3),
                "m2_shuf_best_c": float(c_m2_shuf),
                "m3_rand_best_c": float(c_m3_rand),
                "m0_auc": safe_auc(te_y, m0_prob),
                "m1_auc": safe_auc(te_y, m1_prob),
                "m2_auc": safe_auc(te_y, m2_prob),
                "m3_auc": safe_auc(te_y, m3_prob),
            }
        )

    models = {
        "M0_oracle_question_baseline": summarize_model(y, oof_m0),
        "M1_prefix_length_only": summarize_model(y, oof_m1),
        "M2_l1_tokens_plus_qid_fe": summarize_model(y, oof_m2),
        "M3_l1_tokens_bigrams_plus_qid_fe": summarize_model(y, oof_m3),
    }
    models["AUC_lift_vs_M0"] = {
        "M1_minus_M0": float(models["M1_prefix_length_only"]["auc"] - models["M0_oracle_question_baseline"]["auc"]),
        "M2_minus_M0": float(
            models["M2_l1_tokens_plus_qid_fe"]["auc"] - models["M0_oracle_question_baseline"]["auc"]
        ),
        "M3_minus_M0": float(
            models["M3_l1_tokens_bigrams_plus_qid_fe"]["auc"] - models["M0_oracle_question_baseline"]["auc"]
        ),
    }

    prefix_info = {
        "prefix_fraction": prefix_fraction,
        "n_rows": int(len(prefix_rows)),
        "n_questions": int(len(unique_q)),
        "mean_prefix_length": float(np.mean(token_len)),
        "mean_original_length": float(np.mean(orig_len)),
        "models": models,
        "sanity_checks": {
            "m2_shuffled_label_auc": safe_auc(y_shuf, oof_m2_shuf),
            "m3_prefix_randomized_order_auc": safe_auc(y, oof_m3_rand),
            "m3_randomization_auc_drop": float(
                models["M3_l1_tokens_bigrams_plus_qid_fe"]["auc"] - safe_auc(y, oof_m3_rand)
            ),
        },
        "fold_details": fold_rows,
    }
    oof = {
        "trace_id": trace_ids,
        "y": y,
        "m2_prob": oof_m2,
        "m3_prob": oof_m3,
    }
    return prefix_info, oof


def plot_auc_vs_prefix(path: Path, fractions: list[float], metrics_by_prefix: dict[str, dict[str, object]]) -> None:
    x = np.array(fractions, dtype=float)
    m0 = np.array(
        [metrics_by_prefix[f"{p:.2f}"]["models"]["M0_oracle_question_baseline"]["auc"] for p in fractions],
        dtype=float,
    )
    m1 = np.array(
        [metrics_by_prefix[f"{p:.2f}"]["models"]["M1_prefix_length_only"]["auc"] for p in fractions],
        dtype=float,
    )
    m2 = np.array(
        [metrics_by_prefix[f"{p:.2f}"]["models"]["M2_l1_tokens_plus_qid_fe"]["auc"] for p in fractions],
        dtype=float,
    )
    m3 = np.array(
        [metrics_by_prefix[f"{p:.2f}"]["models"]["M3_l1_tokens_bigrams_plus_qid_fe"]["auc"] for p in fractions],
        dtype=float,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, m0, marker="o", linewidth=2, label="M0 oracle baseline")
    ax.plot(x, m1, marker="o", linewidth=2, label="M1 length-only")
    ax.plot(x, m2, marker="o", linewidth=2, label="M2 tokens+qid")
    ax.plot(x, m3, marker="o", linewidth=2, label="M3 tokens+bigrams+qid")
    ax.set_title("Stage 7 AUC vs Prefix Fraction")
    ax.set_xlabel("Prefix fraction")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.45, 1.0)
    ax.set_xticks(x.tolist())
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def summarize_auc_trajectory(fractions: list[float], values: list[float]) -> dict[str, object]:
    x = np.array(fractions, dtype=float)
    y = np.array(values, dtype=float)
    if len(x) < 2:
        return {"slope": 0.0, "monotonic_non_decreasing": True}
    slope = float(np.polyfit(x, y, 1)[0])
    monotonic = all(values[i] <= values[i + 1] + 1e-12 for i in range(len(values) - 1))
    return {"slope": slope, "monotonic_non_decreasing": bool(monotonic)}


def earliest_prefix_at_or_above_threshold(
    fractions: list[float], auc_values: list[float], threshold: float
) -> float | None:
    for p, auc in zip(fractions, auc_values):
        if auc >= threshold:
            return float(p)
    return None


def write_early_token_leaderboard(
    path: Path,
    prefix_rows_map: dict[float, list[PrefixRow]],
    early_prefixes: list[float],
    min_questions: int,
) -> None:
    rows_out: list[dict[str, object]] = []
    for p in early_prefixes:
        if p not in prefix_rows_map:
            continue
        data = prefix_rows_map[p]
        by_q: dict[str, list[PrefixRow]] = defaultdict(list)
        for r in data:
            by_q[r.question_id].append(r)

        token_questions: dict[str, set[str]] = defaultdict(set)
        for qid, qrows in by_q.items():
            seen = set()
            for row in qrows:
                seen.update(row.tokens)
            for tok in seen:
                token_questions[tok].add(qid)

        candidate_tokens = [
            tok for tok, qs in token_questions.items() if len(qs) >= min_questions and tok.strip() != ""
        ]

        for tok in candidate_tokens:
            deltas: list[float] = []
            traces_with = 0
            traces_without = 0
            for qid, qrows in by_q.items():
                y_with = [r.y for r in qrows if tok in set(r.tokens)]
                y_without = [r.y for r in qrows if tok not in set(r.tokens)]
                traces_with += len(y_with)
                traces_without += len(y_without)
                if not y_with or not y_without:
                    continue
                deltas.append(float(np.mean(y_with) - np.mean(y_without)))
            if not deltas:
                continue
            arr = np.array(deltas, dtype=float)
            rows_out.append(
                {
                    "prefix_fraction": f"{p:.2f}",
                    "token": tok,
                    "n_questions_with_token": int(len(token_questions[tok])),
                    "n_questions_with_valid_delta": int(len(deltas)),
                    "mean_delta": float(np.mean(arr)),
                    "median_delta": float(np.median(arr)),
                    "positive_delta_fraction": float(np.mean(arr > 0)),
                    "support_traces_with_token": int(traces_with),
                    "support_traces_without_token": int(traces_without),
                    "abs_mean_delta": float(abs(np.mean(arr))),
                }
            )

    rows_out.sort(key=lambda r: (-float(r["abs_mean_delta"]), r["token"], r["prefix_fraction"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "prefix_fraction",
        "token",
        "n_questions_with_token",
        "n_questions_with_valid_delta",
        "mean_delta",
        "median_delta",
        "positive_delta_fraction",
        "support_traces_with_token",
        "support_traces_without_token",
        "abs_mean_delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows_out)


def write_time_to_detection(
    path: Path,
    fractions: list[float],
    prefix_oof: dict[float, dict[str, np.ndarray]],
    threshold: float,
) -> None:
    per_trace: dict[str, dict[str, object]] = {}
    for p in fractions:
        oof = prefix_oof[p]
        trace_ids = oof["trace_id"].tolist()
        ys = oof["y"].tolist()
        probs = oof["m2_prob"].tolist()
        for tid, y, prob in zip(trace_ids, ys, probs):
            if tid not in per_trace:
                per_trace[tid] = {"y": int(y), "probs": {}}
            per_trace[tid]["probs"][f"{p:.2f}"] = float(prob)

    rows: list[dict[str, object]] = []
    for tid, item in per_trace.items():
        y = int(item["y"])
        probs = item["probs"]
        earliest = None
        earliest_pred = None
        earliest_conf = None
        for p in fractions:
            key = f"{p:.2f}"
            prob = float(probs[key])
            conf = max(prob, 1.0 - prob)
            pred = int(prob >= 0.5)
            if conf >= threshold:
                earliest = float(p)
                earliest_pred = pred
                earliest_conf = float(conf)
                break
        rows.append(
            {
                "trace_id": tid,
                "y_true": y,
                "earliest_prefix": earliest,
                "earliest_prediction": earliest_pred,
                "earliest_confidence": earliest_conf,
                "detected": earliest is not None,
                "correct_at_detection": (
                    bool(earliest_pred == y) if earliest is not None and earliest_pred is not None else None
                ),
            }
        )

    def summarize_subset(subset: list[dict[str, object]]) -> dict[str, object]:
        n = len(subset)
        detected = [r for r in subset if bool(r["detected"])]
        counts: dict[str, int] = {f"{p:.2f}": 0 for p in fractions}
        for r in detected:
            counts[f"{float(r['earliest_prefix']):.2f}"] += 1
        med = None
        if detected:
            med = float(np.median([float(r["earliest_prefix"]) for r in detected]))
        correct_at_det = None
        if detected:
            vals = [bool(r["correct_at_detection"]) for r in detected if r["correct_at_detection"] is not None]
            if vals:
                correct_at_det = float(np.mean(vals))
        return {
            "n_traces": n,
            "n_detected": len(detected),
            "detection_rate": float(len(detected) / n) if n else 0.0,
            "median_earliest_prefix_detected": med,
            "counts_by_earliest_prefix": counts,
            "accuracy_at_detection": correct_at_det,
        }

    all_rows = rows
    y0_rows = [r for r in rows if int(r["y_true"]) == 0]
    y1_rows = [r for r in rows if int(r["y_true"]) == 1]
    out = {
        "model": "M2_l1_tokens_plus_qid_fe",
        "confidence_threshold": threshold,
        "fractions": [float(p) for p in fractions],
        "summary_all": summarize_subset(all_rows),
        "summary_incorrect_true_label_0": summarize_subset(y0_rows),
        "summary_correct_true_label_1": summarize_subset(y1_rows),
        "per_trace": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)


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

    prefix_fractions = parse_fraction_list(args.prefix_fractions)
    early_prefixes = parse_fraction_list(args.early_prefix_fractions)

    rows = load_rows(args.input)
    if not args.use_all_questions:
        rows = filter_mixed(rows)
    if not rows:
        raise ValueError("No rows available after filtering.")

    c_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

    metrics_by_prefix: dict[str, dict[str, object]] = {}
    prefix_rows_map: dict[float, list[PrefixRow]] = {}
    prefix_oof: dict[float, dict[str, np.ndarray]] = {}
    for p in prefix_fractions:
        pref_rows = build_prefix_rows(rows, p)
        prefix_rows_map[p] = pref_rows
        prefix_metrics, oof = evaluate_prefix(
            prefix_rows=pref_rows,
            n_splits=args.n_splits,
            inner_splits=args.inner_splits,
            c_grid=c_grid,
            seed=args.seed,
        )
        metrics_by_prefix[f"{p:.2f}"] = prefix_metrics
        prefix_oof[p] = oof

    m0_auc = [metrics_by_prefix[f"{p:.2f}"]["models"]["M0_oracle_question_baseline"]["auc"] for p in prefix_fractions]
    m1_auc = [metrics_by_prefix[f"{p:.2f}"]["models"]["M1_prefix_length_only"]["auc"] for p in prefix_fractions]
    m2_auc = [metrics_by_prefix[f"{p:.2f}"]["models"]["M2_l1_tokens_plus_qid_fe"]["auc"] for p in prefix_fractions]
    m3_auc = [
        metrics_by_prefix[f"{p:.2f}"]["models"]["M3_l1_tokens_bigrams_plus_qid_fe"]["auc"] for p in prefix_fractions
    ]

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    plot_auc_vs_prefix(outdir / "auc_vs_prefix.png", prefix_fractions, metrics_by_prefix)

    key_050 = 0.50 if 0.50 in prefix_fractions else min(prefix_fractions, key=lambda x: abs(x - 0.50))
    summary = {
        "input_csv": str(args.input.resolve()),
        "n_traces": int(len(rows)),
        "n_questions": int(len({r.question_id for r in rows})),
        "used_mixed_outcome_only": (not args.use_all_questions),
        "outer_groupkfold_splits": int(args.n_splits),
        "inner_groupkfold_splits": int(args.inner_splits),
        "prefix_fractions": [float(p) for p in prefix_fractions],
        "metrics_by_prefix": metrics_by_prefix,
        "trajectory": {
            "M0_auc_vs_prefix": summarize_auc_trajectory(prefix_fractions, m0_auc),
            "M1_auc_vs_prefix": summarize_auc_trajectory(prefix_fractions, m1_auc),
            "M2_auc_vs_prefix": summarize_auc_trajectory(prefix_fractions, m2_auc),
            "M3_auc_vs_prefix": summarize_auc_trajectory(prefix_fractions, m3_auc),
        },
        "paper_key_numbers": {
            "auc_at_prefix_0_50_model_M2": float(metrics_by_prefix[f"{key_050:.2f}"]["models"]["M2_l1_tokens_plus_qid_fe"]["auc"]),
            "auc_at_prefix_0_50_model_M3": float(metrics_by_prefix[f"{key_050:.2f}"]["models"]["M3_l1_tokens_bigrams_plus_qid_fe"]["auc"]),
            "delta_auc_at_prefix_0_50_M2_minus_M0": float(
                metrics_by_prefix[f"{key_050:.2f}"]["models"]["AUC_lift_vs_M0"]["M2_minus_M0"]
            ),
            "earliest_prefix_auc_ge_0_60_model_M2": earliest_prefix_at_or_above_threshold(
                prefix_fractions, m2_auc, threshold=0.60
            ),
            "earliest_prefix_auc_ge_0_60_model_M3": earliest_prefix_at_or_above_threshold(
                prefix_fractions, m3_auc, threshold=0.60
            ),
        },
    }
    with (outdir / "prefix_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if not args.disable_optional_analysis:
        write_early_token_leaderboard(
            outdir / "early_token_leaderboard.csv",
            prefix_rows_map=prefix_rows_map,
            early_prefixes=early_prefixes,
            min_questions=args.early_min_questions,
        )
        write_time_to_detection(
            outdir / "time_to_detection.json",
            fractions=prefix_fractions,
            prefix_oof=prefix_oof,
            threshold=args.confidence_threshold,
        )

    print(f"Wrote Stage 7 outputs to: {outdir.resolve()}")
    print(f"Traces: {len(rows)}")
    print(f"Questions: {len({r.question_id for r in rows})}")
    for p in prefix_fractions:
        k = f"{p:.2f}"
        m2 = metrics_by_prefix[k]["models"]["M2_l1_tokens_plus_qid_fe"]["auc"]
        m3 = metrics_by_prefix[k]["models"]["M3_l1_tokens_bigrams_plus_qid_fe"]["auc"]
        print(f"prefix={k}: M2 AUC={m2:.4f} | M3 AUC={m3:.4f}")


if __name__ == "__main__":
    main()
