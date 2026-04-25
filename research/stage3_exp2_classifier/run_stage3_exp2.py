#!/usr/bin/env python3
"""Stage 3 / Experiment 2: trace fingerprint classifier.

Implements a grouped-CV classifier pipeline:
- M0: oracle per-question majority baseline (difficulty reference)
- M1: logistic regression on trace length only
- M2: L1 logistic regression on token counts + question fixed effects
- M3: M2 + token bigrams

Outputs:
- cv_metrics.json
- feature_overlap.json
- coefficients.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

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
    token_text: str
    token_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3 classifier experiment.")
    parser.add_argument("--input", type=Path, required=True, help="Tokenized CSV input.")
    parser.add_argument(
        "--stage2-token-deltas",
        type=Path,
        default=Path(
            "research/stage2_exp1_single_token_delta/results_gpt-oss-120b_0/token_deltas_all.csv"
        ),
        help="Stage 2 token deltas CSV for overlap computation.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("research/stage3_exp2_classifier/results"),
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


def load_rows(path: Path) -> list[Row]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        q_col = detect_column(fields, QUESTION_COL_CANDIDATES, "question_id")
        y_col = detect_column(fields, CORRECT_COL_CANDIDATES, "correctness")
        t_col = detect_column(fields, TOKEN_COL_CANDIDATES, "token sequence")
        rows: list[Row] = []
        for r in reader:
            try:
                y = parse_bool(r.get(y_col, ""))
            except ValueError:
                continue
            qid = str(r.get(q_col, "")).strip()
            token_text = (r.get(t_col, "") or "").strip()
            if not qid:
                continue
            tokens = [tok for tok in token_text.split() if tok]
            rows.append(Row(question_id=qid, y=y, token_text=" ".join(tokens), token_count=len(tokens)))
    return rows


def filter_mixed(rows: list[Row]) -> list[Row]:
    by_q: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_q[r.question_id].append(r)
    mixed_q = {
        q
        for q, qrows in by_q.items()
        if {row.y for row in qrows} == {0, 1}
    }
    return [r for r in rows if r.question_id in mixed_q]


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
) -> tuple[csr_matrix, csr_matrix, list[str], CountVectorizer, OneHotEncoder]:
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

    x_train = hstack([x_train_tok, x_train_q], format="csr")
    x_test = hstack([x_test_tok, x_test_q], format="csr")
    feat_names = [f"token::{n}" for n in vec.get_feature_names_out()] + [
        f"qid::{q}" for q in ohe.categories_[0].tolist()
    ]
    return x_train, x_test, feat_names, vec, ohe


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
            x_tr, x_va, _, _, _ = build_token_and_qid_features(
                text[tr_idx], text[va_idx], qids[tr_idx], qids[va_idx], ngram_range=ngram_range
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


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    rows = load_rows(args.input)
    if not args.use_all_questions:
        rows = filter_mixed(rows)
    if not rows:
        raise ValueError("No rows available after filtering.")

    qids = np.array([r.question_id for r in rows])
    y = np.array([r.y for r in rows], dtype=int)
    token_text = np.array([r.token_text for r in rows], dtype=object)
    token_len = np.array([r.token_count for r in rows], dtype=float)

    unique_q = np.unique(qids)
    n_splits = min(args.n_splits, len(unique_q))
    if n_splits < 2:
        raise ValueError("Need at least 2 unique questions for GroupKFold.")

    oof_prob_m0 = np.zeros(len(rows), dtype=float)
    oof_prob_m1 = np.zeros(len(rows), dtype=float)
    oof_prob_m2 = np.zeros(len(rows), dtype=float)
    oof_prob_m3 = np.zeros(len(rows), dtype=float)

    fold_rows: list[dict[str, object]] = []
    c_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    outer = GroupKFold(n_splits=n_splits)

    # oracle per-question baseline for all rows (difficulty reference)
    q_baseline = {
        q: float(np.mean(y[qids == q]))
        for q in unique_q
    }

    for fold_idx, (tr_idx, te_idx) in enumerate(outer.split(token_text, y, groups=qids), start=1):
        tr_text, te_text = token_text[tr_idx], token_text[te_idx]
        tr_y, te_y = y[tr_idx], y[te_idx]
        tr_q, te_q = qids[tr_idx], qids[te_idx]
        tr_len, te_len = token_len[tr_idx], token_len[te_idx]

        # M0 (oracle question baseline)
        m0_prob = np.array([q_baseline[q] for q in te_q], dtype=float)
        oof_prob_m0[te_idx] = m0_prob

        # M1
        m1_clf, m1_scaler = fit_m1_length(tr_len, tr_y)
        m1_prob = m1_clf.predict_proba(m1_scaler.transform(te_len.reshape(-1, 1)))[:, 1]
        oof_prob_m1[te_idx] = m1_prob

        # M2 tune+fit
        c_m2 = tune_c(
            text=tr_text,
            y=tr_y,
            qids=tr_q,
            groups=tr_q,
            ngram_range=(1, 1),
            inner_splits=args.inner_splits,
            c_grid=c_grid,
        )
        x2_tr, x2_te, _, _, _ = build_token_and_qid_features(
            tr_text, te_text, tr_q, te_q, ngram_range=(1, 1)
        )
        m2_clf = fit_l1_logistic(x2_tr, tr_y, c_value=c_m2)
        m2_prob = m2_clf.predict_proba(x2_te)[:, 1]
        oof_prob_m2[te_idx] = m2_prob

        # M3 tune+fit
        c_m3 = tune_c(
            text=tr_text,
            y=tr_y,
            qids=tr_q,
            groups=tr_q,
            ngram_range=(1, 2),
            inner_splits=args.inner_splits,
            c_grid=c_grid,
        )
        x3_tr, x3_te, _, _, _ = build_token_and_qid_features(
            tr_text, te_text, tr_q, te_q, ngram_range=(1, 2)
        )
        m3_clf = fit_l1_logistic(x3_tr, tr_y, c_value=c_m3)
        m3_prob = m3_clf.predict_proba(x3_te)[:, 1]
        oof_prob_m3[te_idx] = m3_prob

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "m2_best_c": c_m2,
                "m3_best_c": c_m3,
                "m0_auc": safe_auc(te_y, m0_prob),
                "m1_auc": safe_auc(te_y, m1_prob),
                "m2_auc": safe_auc(te_y, m2_prob),
                "m3_auc": safe_auc(te_y, m3_prob),
            }
        )

    model_metrics = {
        "M0_oracle_question_baseline": summarize_model(y, oof_prob_m0),
        "M1_length_only": summarize_model(y, oof_prob_m1),
        "M2_l1_tokens_plus_qid_fe": summarize_model(y, oof_prob_m2),
        "M3_l1_tokens_bigrams_plus_qid_fe": summarize_model(y, oof_prob_m3),
    }
    model_metrics["AUC_lift_vs_M0"] = {
        "M1_minus_M0": model_metrics["M1_length_only"]["auc"] - model_metrics["M0_oracle_question_baseline"]["auc"],
        "M2_minus_M0": model_metrics["M2_l1_tokens_plus_qid_fe"]["auc"]
        - model_metrics["M0_oracle_question_baseline"]["auc"],
        "M3_minus_M0": model_metrics["M3_l1_tokens_bigrams_plus_qid_fe"]["auc"]
        - model_metrics["M0_oracle_question_baseline"]["auc"],
    }

    # Fit final models on full data for coefficients and overlap.
    c_m2_full = tune_c(
        text=token_text,
        y=y,
        qids=qids,
        groups=qids,
        ngram_range=(1, 1),
        inner_splits=args.inner_splits,
        c_grid=c_grid,
    )
    x2_full, _, names2, _, _ = build_token_and_qid_features(
        token_text, token_text, qids, qids, ngram_range=(1, 1)
    )
    final_m2 = fit_l1_logistic(x2_full, y, c_value=c_m2_full)
    coef2 = final_m2.coef_.ravel()

    c_m3_full = tune_c(
        text=token_text,
        y=y,
        qids=qids,
        groups=qids,
        ngram_range=(1, 2),
        inner_splits=args.inner_splits,
        c_grid=c_grid,
    )
    x3_full, _, names3, _, _ = build_token_and_qid_features(
        token_text, token_text, qids, qids, ngram_range=(1, 2)
    )
    final_m3 = fit_l1_logistic(x3_full, y, c_value=c_m3_full)
    coef3 = final_m3.coef_.ravel()

    coef_rows: list[dict[str, object]] = []
    for model_name, names, coef in (
        ("M2_l1_tokens_plus_qid_fe", names2, coef2),
        ("M3_l1_tokens_bigrams_plus_qid_fe", names3, coef3),
    ):
        for n, c in zip(names, coef):
            if abs(float(c)) < 1e-12:
                continue
            ftype, fname = n.split("::", 1)
            coef_rows.append(
                {
                    "model": model_name,
                    "feature_type": ftype,
                    "feature": fname,
                    "coefficient": float(c),
                    "abs_coefficient": abs(float(c)),
                }
            )
    coef_rows.sort(key=lambda r: (r["model"], -float(r["abs_coefficient"])))

    # Stage2 overlap with M2 non-zero token features.
    with args.stage2_token_deltas.open("r", newline="", encoding="utf-8") as f:
        s2 = list(csv.DictReader(f))
    s2_sorted = sorted(s2, key=lambda r: abs(float(r.get("mean_delta", 0.0))), reverse=True)
    s2_top = {r["token"] for r in s2_sorted[: min(50, len(s2_sorted))]}
    m2_nonzero_tokens = {
        r["feature"]
        for r in coef_rows
        if r["model"] == "M2_l1_tokens_plus_qid_fe" and r["feature_type"] == "token" and " " not in r["feature"]
    }
    inter = sorted(s2_top & m2_nonzero_tokens)
    union = s2_top | m2_nonzero_tokens
    jaccard = (len(inter) / len(union)) if union else 0.0

    # shuffled-label sanity check for M2
    y_shuf = np.random.permutation(y.copy())
    shuf_probs = np.zeros(len(rows), dtype=float)
    for tr_idx, te_idx in outer.split(token_text, y_shuf, groups=qids):
        tr_text, te_text = token_text[tr_idx], token_text[te_idx]
        tr_q, te_q = qids[tr_idx], qids[te_idx]
        c_shuf = tune_c(
            text=tr_text,
            y=y_shuf[tr_idx],
            qids=tr_q,
            groups=tr_q,
            ngram_range=(1, 1),
            inner_splits=args.inner_splits,
            c_grid=c_grid,
        )
        xtr, xte, _, _, _ = build_token_and_qid_features(tr_text, te_text, tr_q, te_q, ngram_range=(1, 1))
        clf = fit_l1_logistic(xtr, y_shuf[tr_idx], c_shuf)
        shuf_probs[te_idx] = clf.predict_proba(xte)[:, 1]
    shuffled_auc = safe_auc(y_shuf, shuf_probs)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / "cv_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_csv": str(args.input.resolve()),
                "n_rows": int(len(rows)),
                "n_questions": int(len(np.unique(qids))),
                "used_mixed_outcome_only": (not args.use_all_questions),
                "outer_groupkfold_splits": int(n_splits),
                "inner_groupkfold_splits": int(min(args.inner_splits, len(np.unique(qids)))),
                "fold_details": fold_rows,
                "models": model_metrics,
                "sanity_checks": {
                    "m2_shuffled_label_auc": shuffled_auc,
                },
                "selected_C": {
                    "M2_full_fit": c_m2_full,
                    "M3_full_fit": c_m3_full,
                },
            },
            f,
            indent=2,
        )

    with (outdir / "feature_overlap.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "stage2_token_deltas_csv": str(args.stage2_token_deltas.resolve()),
                "stage2_top_k": min(50, len(s2_sorted)),
                "stage2_top_tokens": sorted(s2_top),
                "m2_nonzero_token_features": sorted(m2_nonzero_tokens),
                "intersection_tokens": inter,
                "intersection_size": len(inter),
                "union_size": len(union),
                "jaccard": jaccard,
            },
            f,
            indent=2,
        )

    with (outdir / "coefficients.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "feature_type", "feature", "coefficient", "abs_coefficient"],
        )
        writer.writeheader()
        writer.writerows(coef_rows)

    print(f"Wrote: {outdir.resolve()}")
    print(f"M0 AUC: {model_metrics['M0_oracle_question_baseline']['auc']:.4f}")
    print(f"M2 AUC: {model_metrics['M2_l1_tokens_plus_qid_fe']['auc']:.4f}")
    print(f"M3 AUC: {model_metrics['M3_l1_tokens_bigrams_plus_qid_fe']['auc']:.4f}")
    print(f"Feature overlap Jaccard (M2 vs Stage2 top): {jaccard:.4f}")


if __name__ == "__main__":
    main()
