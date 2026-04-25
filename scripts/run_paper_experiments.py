#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


DEFAULT_GPT_OSS_INPUT = Path(
    "tokenizer/clean_expanded_pool_s100_seed73_qwen25_7b_hot30_tokenized_gpt-oss-120b_0.csv"
)
DEFAULT_DEEPSEEK_INPUT = Path(
    "tokenizer/clean_expanded_pool_s100_seed73_qwen25_7b_hot30_tokenized_DeepSeek-V3.1_0.csv"
)
DEFAULT_OUTPUT_DIR = Path("paper_experiments")

TRUTHY_VALUES = {"true", "1", "yes", "y", "t"}
FALSY_VALUES = {"false", "0", "no", "n", "f"}
EPSILON = 1e-9


@dataclass(frozen=True)
class TraceRow:
    question_id: str
    sample_id: str
    is_correct: bool
    tokens: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paper-oriented reasoning motif experiments: held-out question prediction, "
            "early-prefix prediction, and cross-model transfer."
        )
    )
    parser.add_argument(
        "--gpt-oss-input",
        type=Path,
        default=DEFAULT_GPT_OSS_INPUT,
        help=f"Input CSV for gpt-oss tokenized traces (default: {DEFAULT_GPT_OSS_INPUT}).",
    )
    parser.add_argument(
        "--deepseek-input",
        type=Path,
        default=DEFAULT_DEEPSEEK_INPUT,
        help=f"Input CSV for DeepSeek tokenized traces (default: {DEFAULT_DEEPSEEK_INPUT}).",
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
        help="Minimum training support for motifs used as Bernoulli features (default: 12).",
    )
    parser.add_argument(
        "--prefix-fractions",
        default="0.25,0.5,0.75,1.0",
        help="Comma-separated trace prefix fractions for early prediction (default: 0.25,0.5,0.75,1.0).",
    )
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in TRUTHY_VALUES:
        return True
    if normalized in FALSY_VALUES:
        return False
    raise ValueError(f"Unrecognized boolean value: {value!r}")


def load_rows(path: Path) -> list[TraceRow]:
    rows: list[TraceRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"question_id", "sample_id", "is_correct", "tokenized_trace"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing required columns in {path}: {required}")

        for row in reader:
            tokens = tuple(token for token in row["tokenized_trace"].split() if token)
            rows.append(
                TraceRow(
                    question_id=str(row["question_id"]),
                    sample_id=str(row["sample_id"]),
                    is_correct=parse_bool(row["is_correct"]),
                    tokens=tokens,
                )
            )
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def compute_mixed_question_ids(rows: list[TraceRow]) -> set[str]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        counts[row.question_id][1 if row.is_correct else 0] += 1
    return {qid for qid, (n_fail, n_success) in counts.items() if n_fail > 0 and n_success > 0}


def truncate_tokens(tokens: tuple[str, ...], fraction: float) -> tuple[str, ...]:
    if not tokens:
        return ()
    capped_fraction = min(max(fraction, 0.0), 1.0)
    keep = max(1, math.ceil(len(tokens) * capped_fraction))
    return tokens[:keep]


def unique_contiguous_motifs(
    tokens: tuple[str, ...],
    *,
    min_len: int,
    max_len: int,
) -> set[str]:
    motifs: set[str] = set()
    if not tokens:
        return motifs

    local_max = min(max_len, len(tokens))
    for motif_len in range(max(1, min_len), local_max + 1):
        for start in range(0, len(tokens) - motif_len + 1):
            motifs.add(" ".join(tokens[start : start + motif_len]))
    return motifs


def build_feature_sets(
    rows: list[TraceRow],
    *,
    prefix_fraction: float,
    min_len: int,
    max_len: int,
) -> list[set[str]]:
    feature_sets: list[set[str]] = []
    for row in rows:
        feature_sets.append(
            unique_contiguous_motifs(
                truncate_tokens(row.tokens, prefix_fraction),
                min_len=min_len,
                max_len=max_len,
            )
        )
    return feature_sets


def build_question_folds(rows: list[TraceRow], folds: int, seed: int) -> list[set[str]]:
    if folds <= 1:
        raise ValueError("--folds must be at least 2")
    question_ids = sorted({row.question_id for row in rows})
    rng = random.Random(seed)
    rng.shuffle(question_ids)
    assignments = [set() for _ in range(folds)]
    for idx, qid in enumerate(question_ids):
        assignments[idx % folds].add(qid)
    return assignments


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def roc_auc(rows: list[tuple[float, bool]]) -> float:
    if not rows:
        return 0.5
    positives = sum(1 for _, label in rows if label)
    negatives = len(rows) - positives
    if positives == 0 or negatives == 0:
        return 0.5

    sorted_rows = sorted(rows, key=lambda item: item[0])
    rank = 1
    pos_rank_sum = 0.0
    index = 0
    while index < len(sorted_rows):
        next_index = index + 1
        while next_index < len(sorted_rows) and sorted_rows[next_index][0] == sorted_rows[index][0]:
            next_index += 1
        avg_rank = (rank + (rank + (next_index - index) - 1)) / 2.0
        positives_in_tie = sum(1 for _, label in sorted_rows[index:next_index] if label)
        pos_rank_sum += positives_in_tie * avg_rank
        rank += next_index - index
        index = next_index

    return (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def question_local_auc_summary(
    question_ids: list[str],
    scores: list[float],
    labels: list[bool],
) -> dict[str, float | int]:
    grouped: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for qid, score, label in zip(question_ids, scores, labels):
        grouped[qid].append((score, label))

    aucs: list[float] = []
    for rows in grouped.values():
        positives = sum(1 for _, label in rows if label)
        negatives = len(rows) - positives
        if positives == 0 or negatives == 0:
            continue
        aucs.append(roc_auc(rows))

    return {
        "mixed_question_count": len(aucs),
        "question_local_auc_mean": mean(aucs) if aucs else 0.5,
        "question_local_auc_ge_0_5_rate": (
            sum(1 for auc in aucs if auc >= 0.5) / len(aucs) if aucs else 0.0
        ),
        "question_local_auc_ge_0_6_rate": (
            sum(1 for auc in aucs if auc >= 0.6) / len(aucs) if aucs else 0.0
        ),
    }


def summarize_predictions(scores: list[float], labels: list[bool]) -> dict[str, float | int]:
    if not scores or len(scores) != len(labels):
        raise ValueError("Scores and labels must be non-empty and aligned.")

    predictions = [score >= 0.5 for score in scores]
    tp = sum(1 for pred, label in zip(predictions, labels) if pred and label)
    tn = sum(1 for pred, label in zip(predictions, labels) if (not pred) and (not label))
    fp = sum(1 for pred, label in zip(predictions, labels) if pred and (not label))
    fn = sum(1 for pred, label in zip(predictions, labels) if (not pred) and label)

    total = len(labels)
    accuracy = (tp + tn) / total if total else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (tpr + tnr) / 2.0
    auc = roc_auc(list(zip(scores, labels)))

    return {
        "n": total,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "auc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_length_threshold(lengths: list[int], labels: list[bool]) -> float:
    candidates = sorted(set(lengths))
    if not candidates:
        return 0.0

    best_threshold = float(candidates[0])
    best_bal_acc = -1.0
    thresholds = [candidates[0] - 0.5]
    thresholds.extend((a + b) / 2.0 for a, b in zip(candidates, candidates[1:]))
    thresholds.append(candidates[-1] + 0.5)

    for threshold in thresholds:
        scores = [1.0 if length >= threshold else 0.0 for length in lengths]
        metrics = summarize_predictions(scores, labels)
        bal_acc = float(metrics["balanced_accuracy"])
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_threshold = threshold
    return best_threshold


def length_baseline_scores(
    train_rows: list[TraceRow],
    test_rows: list[TraceRow],
    *,
    prefix_fraction: float,
) -> tuple[list[float], list[float]]:
    train_lengths = [len(truncate_tokens(row.tokens, prefix_fraction)) for row in train_rows]
    train_labels = [row.is_correct for row in train_rows]
    threshold = train_length_threshold(train_lengths, train_labels)
    raw_scores = [float(len(truncate_tokens(row.tokens, prefix_fraction))) for row in test_rows]
    thresholded_scores = [
        1.0 if len(truncate_tokens(row.tokens, prefix_fraction)) >= threshold else 0.0
        for row in test_rows
    ]
    return raw_scores, thresholded_scores


def summarize_length_baseline(
    train_rows: list[TraceRow],
    test_rows: list[TraceRow],
    *,
    prefix_fraction: float,
) -> dict[str, float | int]:
    raw_scores, thresholded_scores = length_baseline_scores(
        train_rows,
        test_rows,
        prefix_fraction=prefix_fraction,
    )
    labels = [row.is_correct for row in test_rows]
    question_ids = [row.question_id for row in test_rows]

    threshold_metrics = summarize_predictions(thresholded_scores, labels)
    threshold_metrics["auc"] = roc_auc(list(zip(raw_scores, labels)))
    threshold_metrics.update(question_local_auc_summary(question_ids, raw_scores, labels))
    return threshold_metrics


def train_bernoulli_nb(
    rows: list[TraceRow],
    feature_sets: list[set[str]],
    *,
    min_support: int,
    alpha: float = 1.0,
) -> tuple[float, dict[str, float], dict[str, int]]:
    if len(rows) != len(feature_sets):
        raise ValueError("Rows and feature_sets must be aligned.")

    pos_total = sum(1 for row in rows if row.is_correct)
    neg_total = len(rows) - pos_total
    if pos_total == 0 or neg_total == 0:
        raise ValueError("Bernoulli NB requires both positive and negative examples.")

    pos_counts: Counter[str] = Counter()
    neg_counts: Counter[str] = Counter()
    total_counts: Counter[str] = Counter()

    for row, features in zip(rows, feature_sets):
        if row.is_correct:
            pos_counts.update(features)
        else:
            neg_counts.update(features)
        total_counts.update(features)

    selected = {feature for feature, count in total_counts.items() if count >= min_support}
    prior_logit = math.log((pos_total + alpha) / (neg_total + alpha))
    constant = prior_logit
    deltas: dict[str, float] = {}

    for feature in sorted(selected):
        p_pos = (pos_counts[feature] + alpha) / (pos_total + 2.0 * alpha)
        p_neg = (neg_counts[feature] + alpha) / (neg_total + 2.0 * alpha)

        constant += math.log(1.0 - p_pos + EPSILON) - math.log(1.0 - p_neg + EPSILON)
        deltas[feature] = (
            math.log(p_pos + EPSILON)
            - math.log(1.0 - p_pos + EPSILON)
            - math.log(p_neg + EPSILON)
            + math.log(1.0 - p_neg + EPSILON)
        )

    metadata = {
        "selected_feature_count": len(selected),
        "train_positive_count": pos_total,
        "train_negative_count": neg_total,
    }
    return constant, deltas, metadata


def score_bernoulli_nb(
    feature_sets: list[set[str]],
    *,
    constant: float,
    deltas: dict[str, float],
) -> list[float]:
    scores: list[float] = []
    for features in feature_sets:
        logit = constant
        for feature in features:
            delta = deltas.get(feature)
            if delta is not None:
                logit += delta
        scores.append(sigmoid(logit))
    return scores


def mean_metric(metrics: list[dict[str, float | int]], key: str) -> float:
    return sum(float(metric[key]) for metric in metrics) / len(metrics) if metrics else 0.0


def run_question_holdout_experiment(
    rows: list[TraceRow],
    *,
    folds: int,
    seed: int,
    min_support: int,
    prefix_fraction: float,
    min_len: int,
    max_len: int,
    experiment_name: str,
) -> dict[str, object]:
    test_folds = build_question_folds(rows, folds, seed)
    fold_results: list[dict[str, object]] = []

    for fold_idx, test_question_ids in enumerate(test_folds):
        train_rows = [row for row in rows if row.question_id not in test_question_ids]
        test_rows = [row for row in rows if row.question_id in test_question_ids]

        train_features = build_feature_sets(
            train_rows,
            prefix_fraction=prefix_fraction,
            min_len=min_len,
            max_len=max_len,
        )
        test_features = build_feature_sets(
            test_rows,
            prefix_fraction=prefix_fraction,
            min_len=min_len,
            max_len=max_len,
        )

        constant, deltas, metadata = train_bernoulli_nb(
            train_rows,
            train_features,
            min_support=min_support,
        )
        test_scores = score_bernoulli_nb(test_features, constant=constant, deltas=deltas)
        labels = [row.is_correct for row in test_rows]
        question_ids = [row.question_id for row in test_rows]

        motif_metrics = summarize_predictions(test_scores, labels)
        motif_metrics.update(question_local_auc_summary(question_ids, test_scores, labels))
        length_metrics = summarize_length_baseline(
            train_rows,
            test_rows,
            prefix_fraction=prefix_fraction,
        )

        fold_results.append(
            {
                "fold_index": fold_idx,
                "test_question_count": len(test_question_ids),
                "test_trace_count": len(test_rows),
                "motif": motif_metrics,
                "length_only": length_metrics,
                "selected_feature_count": metadata["selected_feature_count"],
            }
        )

    return {
        "experiment_name": experiment_name,
        "prefix_fraction": prefix_fraction,
        "motif_len_range": [min_len, max_len],
        "min_support": min_support,
        "folds": fold_results,
        "aggregate": {
            "motif_accuracy": mean_metric([fold["motif"] for fold in fold_results], "accuracy"),
            "motif_balanced_accuracy": mean_metric([fold["motif"] for fold in fold_results], "balanced_accuracy"),
            "motif_auc": mean_metric([fold["motif"] for fold in fold_results], "auc"),
            "motif_question_local_auc_mean": mean_metric(
                [fold["motif"] for fold in fold_results], "question_local_auc_mean"
            ),
            "length_accuracy": mean_metric([fold["length_only"] for fold in fold_results], "accuracy"),
            "length_balanced_accuracy": mean_metric([fold["length_only"] for fold in fold_results], "balanced_accuracy"),
            "length_auc": mean_metric([fold["length_only"] for fold in fold_results], "auc"),
            "length_question_local_auc_mean": mean_metric(
                [fold["length_only"] for fold in fold_results], "question_local_auc_mean"
            ),
            "avg_selected_feature_count": mean_metric(
                [{"selected_feature_count": fold["selected_feature_count"]} for fold in fold_results],
                "selected_feature_count",
            ),
        },
    }


def run_cross_model_transfer(
    source_rows: list[TraceRow],
    target_rows: list[TraceRow],
    *,
    min_support: int,
    prefix_fraction: float,
    min_len: int,
    max_len: int,
    experiment_name: str,
) -> dict[str, object]:
    source_features = build_feature_sets(
        source_rows,
        prefix_fraction=prefix_fraction,
        min_len=min_len,
        max_len=max_len,
    )
    target_features = build_feature_sets(
        target_rows,
        prefix_fraction=prefix_fraction,
        min_len=min_len,
        max_len=max_len,
    )
    constant, deltas, metadata = train_bernoulli_nb(
        source_rows,
        source_features,
        min_support=min_support,
    )
    motif_scores = score_bernoulli_nb(target_features, constant=constant, deltas=deltas)
    labels = [row.is_correct for row in target_rows]
    question_ids = [row.question_id for row in target_rows]
    motif_metrics = summarize_predictions(motif_scores, labels)
    motif_metrics.update(question_local_auc_summary(question_ids, motif_scores, labels))
    length_metrics = summarize_length_baseline(
        source_rows,
        target_rows,
        prefix_fraction=prefix_fraction,
    )

    return {
        "experiment_name": experiment_name,
        "prefix_fraction": prefix_fraction,
        "motif_len_range": [min_len, max_len],
        "min_support": min_support,
        "selected_feature_count": metadata["selected_feature_count"],
        "motif": motif_metrics,
        "length_only": length_metrics,
    }


def top_weighted_features(
    rows: list[TraceRow],
    *,
    prefix_fraction: float,
    min_len: int,
    max_len: int,
    min_support: int,
    top_k: int = 25,
) -> dict[str, list[dict[str, float | int | str]]]:
    feature_sets = build_feature_sets(
        rows,
        prefix_fraction=prefix_fraction,
        min_len=min_len,
        max_len=max_len,
    )
    _, deltas, _ = train_bernoulli_nb(rows, feature_sets, min_support=min_support)

    support_counter: Counter[str] = Counter()
    for features in feature_sets:
        support_counter.update(features)

    ranked = sorted(
        (
            {
                "motif": feature,
                "weight": weight,
                "support": support_counter[feature],
            }
            for feature, weight in deltas.items()
        ),
        key=lambda item: abs(float(item["weight"])),
        reverse=True,
    )
    success = [item for item in ranked if float(item["weight"]) > 0][:top_k]
    failure = [item for item in ranked if float(item["weight"]) < 0][:top_k]
    return {"success": success, "failure": failure}


def full_weight_table(
    rows: list[TraceRow],
    *,
    prefix_fraction: float,
    min_len: int,
    max_len: int,
    min_support: int,
) -> dict[str, float]:
    feature_sets = build_feature_sets(
        rows,
        prefix_fraction=prefix_fraction,
        min_len=min_len,
        max_len=max_len,
    )
    _, deltas, _ = train_bernoulli_nb(rows, feature_sets, min_support=min_support)
    return deltas


def shared_motif_stability(
    left_name: str,
    left_weights: dict[str, float],
    right_name: str,
    right_weights: dict[str, float],
    *,
    top_k: int = 25,
) -> dict[str, object]:
    shared = sorted(set(left_weights) & set(right_weights))
    same_sign_rows: list[dict[str, float | str]] = []
    sign_agree = 0
    success_success = 0
    failure_failure = 0

    for motif in shared:
        left_weight = float(left_weights[motif])
        right_weight = float(right_weights[motif])
        same_sign = (left_weight > 0) == (right_weight > 0)
        if same_sign:
            sign_agree += 1
            if left_weight > 0:
                success_success += 1
            else:
                failure_failure += 1
            same_sign_rows.append(
                {
                    "motif": motif,
                    "left_weight": left_weight,
                    "right_weight": right_weight,
                    "abs_weight_sum": abs(left_weight) + abs(right_weight),
                    "direction": "success" if left_weight > 0 else "failure",
                }
            )

    same_sign_rows.sort(key=lambda row: float(row["abs_weight_sum"]), reverse=True)

    return {
        "left_name": left_name,
        "right_name": right_name,
        "shared_feature_count": len(shared),
        "sign_agreement_rate": (sign_agree / len(shared)) if shared else 0.0,
        "shared_success_success_count": success_success,
        "shared_failure_failure_count": failure_failure,
        "top_same_sign_rows": same_sign_rows[:top_k],
    }


def write_markdown_report(
    output_path: Path,
    *,
    question_holdout_results: dict[str, dict[str, object]],
    prefix_results: dict[str, dict[str, object]],
    cross_model_results: list[dict[str, object]],
    motif_tables: dict[str, dict[str, list[dict[str, float | int | str]]]],
    motif_stability: dict[str, object],
) -> None:
    lines = [
        "# Paper Experiments Report",
        "",
        "## Held-out Question Prediction",
        "",
        "| Dataset | Feature Set | Accuracy | Balanced Accuracy | AUC | Q-Local AUC | Length Q-Local AUC | Avg Selected Features |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for dataset_name, result in question_holdout_results.items():
        aggregate = result["aggregate"]
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset_name,
                    result["experiment_name"],
                    f"{float(aggregate['motif_accuracy']):.4f}",
                    f"{float(aggregate['motif_balanced_accuracy']):.4f}",
                    f"{float(aggregate['motif_auc']):.4f}",
                    f"{float(aggregate['motif_question_local_auc_mean']):.4f}",
                    f"{float(aggregate['length_question_local_auc_mean']):.4f}",
                    f"{float(aggregate['avg_selected_feature_count']):.1f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Early-Prefix Prediction",
            "",
            "| Dataset | Prefix Fraction | Motif AUC | Motif Q-Local AUC | Length Q-Local AUC | Motif Balanced Accuracy |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for dataset_name, result in prefix_results.items():
        aggregate = result["aggregate"]
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset_name,
                    f"{float(result['prefix_fraction']):.2f}",
                    f"{float(aggregate['motif_auc']):.4f}",
                    f"{float(aggregate['motif_question_local_auc_mean']):.4f}",
                    f"{float(aggregate['length_question_local_auc_mean']):.4f}",
                    f"{float(aggregate['motif_balanced_accuracy']):.4f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Cross-Model Transfer",
            "",
            "| Transfer | Prefix Fraction | Motif AUC | Motif Q-Local AUC | Length Q-Local AUC | Motif Balanced Accuracy | Selected Features |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for result in cross_model_results:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(result["experiment_name"]),
                    f"{float(result['prefix_fraction']):.2f}",
                    f"{float(result['motif']['auc']):.4f}",
                    f"{float(result['motif']['question_local_auc_mean']):.4f}",
                    f"{float(result['length_only']['question_local_auc_mean']):.4f}",
                    f"{float(result['motif']['balanced_accuracy']):.4f}",
                    str(result["selected_feature_count"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Shared Motif Stability",
            "",
            f"- Shared features between {motif_stability['left_name']} and {motif_stability['right_name']}: {motif_stability['shared_feature_count']}",
            f"- Sign agreement rate on shared features: {float(motif_stability['sign_agreement_rate']):.4f}",
            f"- Shared success motifs: {motif_stability['shared_success_success_count']}",
            f"- Shared failure motifs: {motif_stability['shared_failure_failure_count']}",
            "",
            "| Motif | Direction | Left Weight | Right Weight | Weight Sum |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in motif_stability["top_same_sign_rows"]:
        lines.append(
            f"| {row['motif']} | {row['direction']} | {float(row['left_weight']):.4f} | {float(row['right_weight']):.4f} | {float(row['abs_weight_sum']):.4f} |"
        )

    for dataset_name, tables in motif_tables.items():
        lines.extend(
            [
                "",
                f"## Top Weighted Motifs: {dataset_name}",
                "",
                "### Success-leaning",
                "",
                "| Motif | Weight | Support |",
                "| --- | ---: | ---: |",
            ]
        )
        for row in tables["success"][:15]:
            lines.append(
                f"| {row['motif']} | {float(row['weight']):.4f} | {int(row['support'])} |"
            )
        lines.extend(
            [
                "",
                "### Failure-leaning",
                "",
                "| Motif | Weight | Support |",
                "| --- | ---: | ---: |",
            ]
        )
        for row in tables["failure"][:15]:
            lines.append(
                f"| {row['motif']} | {float(row['weight']):.4f} | {int(row['support'])} |"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix_fractions = [float(item.strip()) for item in args.prefix_fractions.split(",") if item.strip()]
    if not prefix_fractions:
        raise ValueError("At least one prefix fraction is required.")

    gpt_rows = load_rows(args.gpt_oss_input)
    deepseek_rows = load_rows(args.deepseek_input)

    gpt_mixed_ids = compute_mixed_question_ids(gpt_rows)
    deepseek_mixed_ids = compute_mixed_question_ids(deepseek_rows)
    gpt_mixed_rows = [row for row in gpt_rows if row.question_id in gpt_mixed_ids]
    deepseek_mixed_rows = [row for row in deepseek_rows if row.question_id in deepseek_mixed_ids]

    question_holdout_results = {
        "gpt_oss_full": run_question_holdout_experiment(
            gpt_rows,
            folds=args.folds,
            seed=args.seed,
            min_support=args.min_support,
            prefix_fraction=1.0,
            min_len=1,
            max_len=3,
            experiment_name="motif_1to3_full_trace",
        ),
        "gpt_oss_mixed_questions": run_question_holdout_experiment(
            gpt_mixed_rows,
            folds=args.folds,
            seed=args.seed,
            min_support=args.min_support,
            prefix_fraction=1.0,
            min_len=1,
            max_len=3,
            experiment_name="motif_1to3_full_trace",
        ),
        "deepseek_full": run_question_holdout_experiment(
            deepseek_rows,
            folds=args.folds,
            seed=args.seed,
            min_support=args.min_support,
            prefix_fraction=1.0,
            min_len=1,
            max_len=3,
            experiment_name="motif_1to3_full_trace",
        ),
        "deepseek_mixed_questions": run_question_holdout_experiment(
            deepseek_mixed_rows,
            folds=args.folds,
            seed=args.seed,
            min_support=args.min_support,
            prefix_fraction=1.0,
            min_len=1,
            max_len=3,
            experiment_name="motif_1to3_full_trace",
        ),
    }

    prefix_results: dict[str, dict[str, object]] = {}
    for dataset_name, rows in [
        ("gpt_oss", gpt_rows),
        ("deepseek", deepseek_rows),
    ]:
        for fraction in prefix_fractions:
            key = f"{dataset_name}_prefix_{fraction:.2f}"
            prefix_results[key] = run_question_holdout_experiment(
                rows,
                folds=args.folds,
                seed=args.seed,
                min_support=args.min_support,
                prefix_fraction=fraction,
                min_len=1,
                max_len=3,
                experiment_name="motif_1to3_prefix",
            )

    cross_model_results = [
        run_cross_model_transfer(
            gpt_rows,
            deepseek_rows,
            min_support=args.min_support,
            prefix_fraction=1.0,
            min_len=1,
            max_len=3,
            experiment_name="train_gpt_oss_test_deepseek",
        ),
        run_cross_model_transfer(
            deepseek_rows,
            gpt_rows,
            min_support=args.min_support,
            prefix_fraction=1.0,
            min_len=1,
            max_len=3,
            experiment_name="train_deepseek_test_gpt_oss",
        ),
    ]

    motif_tables = {
        "gpt_oss": top_weighted_features(
            gpt_rows,
            prefix_fraction=1.0,
            min_len=1,
            max_len=3,
            min_support=args.min_support,
        ),
        "deepseek": top_weighted_features(
            deepseek_rows,
            prefix_fraction=1.0,
            min_len=1,
            max_len=3,
            min_support=args.min_support,
        ),
    }
    gpt_full_weights = full_weight_table(
        gpt_rows,
        prefix_fraction=1.0,
        min_len=1,
        max_len=3,
        min_support=args.min_support,
    )
    deepseek_full_weights = full_weight_table(
        deepseek_rows,
        prefix_fraction=1.0,
        min_len=1,
        max_len=3,
        min_support=args.min_support,
    )
    motif_stability = shared_motif_stability(
        "gpt_oss",
        gpt_full_weights,
        "deepseek",
        deepseek_full_weights,
    )

    payload = {
        "config": {
            "folds": args.folds,
            "seed": args.seed,
            "min_support": args.min_support,
            "prefix_fractions": prefix_fractions,
        },
        "datasets": {
            "gpt_oss_full_rows": len(gpt_rows),
            "gpt_oss_mixed_rows": len(gpt_mixed_rows),
            "deepseek_full_rows": len(deepseek_rows),
            "deepseek_mixed_rows": len(deepseek_mixed_rows),
        },
        "question_holdout_results": question_holdout_results,
        "prefix_results": prefix_results,
        "cross_model_results": cross_model_results,
        "motif_tables": motif_tables,
        "motif_stability": motif_stability,
    }

    json_path = output_dir / "paper_experiments_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(
        output_dir / "paper_experiments_report.md",
        question_holdout_results=question_holdout_results,
        prefix_results=prefix_results,
        cross_model_results=cross_model_results,
        motif_tables=motif_tables,
        motif_stability=motif_stability,
    )

    print(f"Wrote summary JSON to {json_path}")
    print(f"Wrote report to {output_dir / 'paper_experiments_report.md'}")


if __name__ == "__main__":
    main()
