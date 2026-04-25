#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_paper_experiments import (  # type: ignore
    build_feature_sets,
    build_question_folds,
    load_rows,
    roc_auc,
    run_question_holdout_experiment,
    score_bernoulli_nb,
    train_bernoulli_nb,
    truncate_tokens,
)


@dataclass(frozen=True)
class TraceRecord:
    question_id: str
    sample_id: str
    gold_answer: str
    predicted_answer: str
    is_correct: bool
    tokens: tuple[str, ...]


@dataclass
class ExampleRef:
    question_id: str
    sample_id: str
    is_correct: bool
    predicted_answer: str
    tokenized_trace: str


@dataclass
class MotifCard:
    motif: str
    motif_tokens: list[str]
    motif_length: int
    support: int
    prevalence: float
    question_coverage: int
    success_present: int
    failure_present: int
    success_rate_present: float
    success_rate_absent: float
    support_difference: float
    lift_overall_success: float
    odds_ratio: float
    odds_ratio_ci_low: float
    odds_ratio_ci_high: float
    fisher_p_value: float
    fdr_q_value: float
    avg_stage_position: float
    stage_bucket: str
    avg_trace_length_present: float
    noise_cooccurrence_rate: float
    representative_examples: list[dict[str, object]]
    top_neighbor_motifs: list[str]


@dataclass
class QuestionMotifBundle:
    question_id: str
    trace_count: int
    success_count: int
    failure_count: int
    avg_token_count: float
    rows_with_noise: int
    local_top: list[MotifCard]
    local_success_enriched: list[MotifCard]
    local_failure_enriched: list[MotifCard]
    visible_global_top: list[MotifCard]
    visible_global_success: list[MotifCard]
    visible_global_failure: list[MotifCard]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build motif-card artifacts and usefulness experiments from tokenized traces."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("tokenizer/gpt_oss_subset_15q_all_traces_tokenized_gpt-oss-chunked_live_1.csv"),
        help="Tokenized trace CSV to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_experiments/motif_cards_gpt_oss_live_1_filtered"),
        help="Directory to write motif-card artifacts.",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=2,
        help="Minimum motif length.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=3,
        help="Maximum motif length.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=8,
        help="Minimum trace support for a motif card.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of cards to surface per family.",
    )
    parser.add_argument(
        "--examples-per-card",
        type=int,
        default=3,
        help="Representative examples stored for each card.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Question holdout folds for experiments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=73,
        help="Random seed for question folds.",
    )
    parser.add_argument(
        "--exclude-substrings",
        default="final,solution,answer",
        help="Comma-separated substrings; motifs containing any of them are filtered out of surfaced cards.",
    )
    parser.add_argument(
        "--per-question-top-k",
        type=int,
        default=8,
        help="Number of per-question cards to surface per family.",
    )
    parser.add_argument(
        "--per-question-min-support",
        type=int,
        default=2,
        help="Minimum trace support for per-question local motif cards.",
    )
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "y", "t"}


def load_trace_records(path: Path) -> list[TraceRecord]:
    records: list[TraceRecord] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tokenized_trace = row["tokenized_trace"].strip()
            records.append(
                TraceRecord(
                    question_id=str(row["question_id"]),
                    sample_id=str(row["sample_id"]),
                    gold_answer=str(row.get("gold_answer", "")),
                    predicted_answer=str(row.get("predicted_answer", "")),
                    is_correct=parse_bool(str(row["is_correct"])),
                    tokens=tuple(token for token in tokenized_trace.split() if token),
                )
            )
    if not records:
        raise ValueError(f"No rows found in {path}")
    return records


def unique_contiguous_motifs(tokens: tuple[str, ...], min_len: int, max_len: int) -> set[tuple[str, ...]]:
    motifs: set[tuple[str, ...]] = set()
    if not tokens:
        return motifs
    local_max = min(max_len, len(tokens))
    for width in range(max(1, min_len), local_max + 1):
        for start in range(0, len(tokens) - width + 1):
            motifs.add(tokens[start : start + width])
    return motifs


def first_motif_position(tokens: tuple[str, ...], motif: tuple[str, ...]) -> float:
    if not tokens or not motif or len(tokens) < len(motif):
        return 0.0
    width = len(motif)
    for idx in range(0, len(tokens) - width + 1):
        if tokens[idx : idx + width] == motif:
            if len(tokens) == width:
                return 0.5
            return idx / max(1, len(tokens) - width)
    return 0.0


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    q_values = [1.0] * n
    prev = 1.0
    for reverse_rank, (index, p_value) in enumerate(reversed(indexed), start=1):
        rank = n - reverse_rank + 1
        value = (p_value * n) / rank
        prev = min(prev, value)
        q_values[index] = max(0.0, min(1.0, prev))
    return q_values


def fisher_exact_pvalue(a: int, b: int, c: int, d: int) -> float:
    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d
    total = row1 + row2

    def hypergeom(x: int) -> float:
        return (
            math.comb(col1, x) * math.comb(col2, row1 - x) / math.comb(total, row1)
        )

    observed = hypergeom(a)
    lower = max(0, col1 - row2)
    upper = min(col1, row1)
    p_value = 0.0
    for x in range(lower, upper + 1):
        prob = hypergeom(x)
        if prob <= observed + 1e-12:
            p_value += prob
    return min(1.0, p_value)


def odds_ratio_and_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    aa = a + 0.5
    bb = b + 0.5
    cc = c + 0.5
    dd = d + 0.5
    odds_ratio = (aa * dd) / (bb * cc)
    log_or = math.log(odds_ratio)
    se = math.sqrt((1.0 / aa) + (1.0 / bb) + (1.0 / cc) + (1.0 / dd))
    z = 1.96
    return (
        odds_ratio,
        math.exp(log_or - z * se),
        math.exp(log_or + z * se),
    )


def stage_bucket(position: float) -> str:
    if position < 0.34:
        return "early"
    if position < 0.67:
        return "middle"
    return "late"


def parse_excluded_substrings(raw: str) -> tuple[str, ...]:
    items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return tuple(items)


def motif_is_allowed(motif: tuple[str, ...], excluded_substrings: tuple[str, ...]) -> bool:
    if not excluded_substrings:
        return True
    lowered = [token.lower() for token in motif]
    return not any(excluded in token for excluded in excluded_substrings for token in lowered)


def build_dataset_summary(records: list[TraceRecord]) -> dict[str, object]:
    question_counts = Counter(record.question_id for record in records)
    lengths = [len(record.tokens) for record in records]
    noise_rows = sum(1 for record in records if "noise:corrupted-span" in record.tokens)
    type_counts: Counter[str] = Counter()
    unique_tokens: set[str] = set()
    for record in records:
        for token in record.tokens:
            unique_tokens.add(token)
            token_type = token.split(":", 1)[0] if ":" in token else "unknown"
            type_counts[token_type] += 1

    return {
        "trace_count": len(records),
        "question_count": len(question_counts),
        "question_trace_counts": dict(sorted(question_counts.items(), key=lambda item: int(item[0]))),
        "success_count": sum(1 for record in records if record.is_correct),
        "failure_count": sum(1 for record in records if not record.is_correct),
        "avg_token_count": statistics.mean(lengths),
        "median_token_count": statistics.median(lengths),
        "max_token_count": max(lengths),
        "rows_with_noise": noise_rows,
        "rows_with_noise_rate": noise_rows / len(records),
        "unique_token_count": len(unique_tokens),
        "token_type_counts": dict(type_counts),
    }


def build_motif_cards(
    records: list[TraceRecord],
    *,
    min_len: int,
    max_len: int,
    min_support: int,
    examples_per_card: int,
    excluded_substrings: tuple[str, ...],
) -> list[MotifCard]:
    total = len(records)
    total_success = sum(1 for record in records if record.is_correct)
    total_failure = total - total_success
    overall_success_rate = total_success / total if total else 0.0

    support_map: dict[tuple[str, ...], list[int]] = defaultdict(list)
    question_map: dict[tuple[str, ...], set[str]] = defaultdict(set)
    stage_positions: dict[tuple[str, ...], list[float]] = defaultdict(list)
    noise_flags: dict[tuple[str, ...], list[int]] = defaultdict(list)
    trace_lengths: dict[tuple[str, ...], list[int]] = defaultdict(list)
    examples: dict[tuple[str, ...], list[ExampleRef]] = defaultdict(list)
    feature_sets_by_trace: list[set[tuple[str, ...]]] = []

    for record in records:
        motifs = unique_contiguous_motifs(record.tokens, min_len=min_len, max_len=max_len)
        feature_sets_by_trace.append(motifs)
        for motif in motifs:
            support_map[motif].append(1 if record.is_correct else 0)
            question_map[motif].add(record.question_id)
            stage_positions[motif].append(first_motif_position(record.tokens, motif))
            noise_flags[motif].append(1 if "noise:corrupted-span" in record.tokens else 0)
            trace_lengths[motif].append(len(record.tokens))
            if len(examples[motif]) < examples_per_card:
                examples[motif].append(
                    ExampleRef(
                        question_id=record.question_id,
                        sample_id=record.sample_id,
                        is_correct=record.is_correct,
                        predicted_answer=record.predicted_answer,
                        tokenized_trace=" ".join(record.tokens),
                    )
                )

    p_values: list[float] = []
    pending_cards: list[dict[str, object]] = []

    for motif, labels in support_map.items():
        if not motif_is_allowed(motif, excluded_substrings):
            continue
        support = len(labels)
        if support < min_support:
            continue
        success_present = sum(labels)
        failure_present = support - success_present
        success_absent = total_success - success_present
        failure_absent = total_failure - failure_present
        present_rate = success_present / support if support else 0.0
        absent_total = total - support
        absent_rate = success_absent / absent_total if absent_total else overall_success_rate
        support_difference = present_rate - absent_rate
        lift = present_rate / overall_success_rate if overall_success_rate else 0.0
        odds_ratio, ci_low, ci_high = odds_ratio_and_ci(
            success_present,
            failure_present,
            success_absent,
            failure_absent,
        )
        p_value = fisher_exact_pvalue(
            success_present,
            failure_present,
            success_absent,
            failure_absent,
        )
        p_values.append(p_value)
        pending_cards.append(
            {
                "motif": motif,
                "support": support,
                "prevalence": support / total,
                "question_coverage": len(question_map[motif]),
                "success_present": success_present,
                "failure_present": failure_present,
                "success_rate_present": present_rate,
                "success_rate_absent": absent_rate,
                "support_difference": support_difference,
                "lift_overall_success": lift,
                "odds_ratio": odds_ratio,
                "odds_ratio_ci_low": ci_low,
                "odds_ratio_ci_high": ci_high,
                "fisher_p_value": p_value,
                "avg_stage_position": statistics.mean(stage_positions[motif]),
                "stage_bucket": stage_bucket(statistics.mean(stage_positions[motif])),
                "avg_trace_length_present": statistics.mean(trace_lengths[motif]),
                "noise_cooccurrence_rate": statistics.mean(noise_flags[motif]),
                "representative_examples": [asdict(example) for example in examples[motif]],
            }
        )

    q_values = benjamini_hochberg(p_values)
    cards: list[MotifCard] = []
    for pending, q_value in zip(pending_cards, q_values):
        motif = pending["motif"]
        cards.append(
            MotifCard(
                motif=" ".join(motif),
                motif_tokens=list(motif),
                motif_length=len(motif),
                support=int(pending["support"]),
                prevalence=float(pending["prevalence"]),
                question_coverage=int(pending["question_coverage"]),
                success_present=int(pending["success_present"]),
                failure_present=int(pending["failure_present"]),
                success_rate_present=float(pending["success_rate_present"]),
                success_rate_absent=float(pending["success_rate_absent"]),
                support_difference=float(pending["support_difference"]),
                lift_overall_success=float(pending["lift_overall_success"]),
                odds_ratio=float(pending["odds_ratio"]),
                odds_ratio_ci_low=float(pending["odds_ratio_ci_low"]),
                odds_ratio_ci_high=float(pending["odds_ratio_ci_high"]),
                fisher_p_value=float(pending["fisher_p_value"]),
                fdr_q_value=float(q_value),
                avg_stage_position=float(pending["avg_stage_position"]),
                stage_bucket=str(pending["stage_bucket"]),
                avg_trace_length_present=float(pending["avg_trace_length_present"]),
                noise_cooccurrence_rate=float(pending["noise_cooccurrence_rate"]),
                representative_examples=list(pending["representative_examples"]),
                top_neighbor_motifs=[],
            )
        )

    motif_feature_sets = {
        tuple(card.motif_tokens): {index for index, features in enumerate(feature_sets_by_trace) if tuple(card.motif_tokens) in features}
        for card in cards
    }
    for card in cards:
        motif_key = tuple(card.motif_tokens)
        present_indices = motif_feature_sets[motif_key]
        neighbors: list[tuple[float, str]] = []
        for other in cards:
            if other.motif == card.motif:
                continue
            other_key = tuple(other.motif_tokens)
            overlap = len(present_indices & motif_feature_sets[other_key])
            if overlap == 0:
                continue
            overlap_rate = overlap / len(present_indices)
            neighbors.append((overlap_rate, other.motif))
        neighbors.sort(reverse=True)
        card.top_neighbor_motifs = [motif for _, motif in neighbors[:5]]

    return cards


def contains_motif(tokens: tuple[str, ...], motif_tokens: tuple[str, ...]) -> bool:
    if not motif_tokens or len(tokens) < len(motif_tokens):
        return False
    width = len(motif_tokens)
    for idx in range(0, len(tokens) - width + 1):
        if tokens[idx : idx + width] == motif_tokens:
            return True
    return False


def top_card_families(
    cards: list[MotifCard],
    top_k: int,
    *,
    min_question_coverage: int = 2,
) -> dict[str, list[MotifCard]]:
    by_support = sorted(
        cards,
        key=lambda card: (card.support, card.question_coverage, abs(card.support_difference)),
        reverse=True,
    )
    success_enriched = [
        card
        for card in cards
        if card.support_difference > 0.0 and card.question_coverage >= min_question_coverage
    ]
    success_enriched.sort(
        key=lambda card: (
            card.support_difference,
            -card.fdr_q_value,
            card.support,
        ),
        reverse=True,
    )
    failure_enriched = [
        card
        for card in cards
        if card.support_difference < 0.0 and card.question_coverage >= min_question_coverage
    ]
    failure_enriched.sort(
        key=lambda card: (
            card.support_difference,
            card.fdr_q_value,
            -card.support,
        )
    )
    return {
        "global_top": by_support[:top_k],
        "success_enriched": success_enriched[:top_k],
        "failure_enriched": failure_enriched[:top_k],
    }


def build_question_bundles(
    records: list[TraceRecord],
    *,
    global_families: dict[str, list[MotifCard]],
    min_len: int,
    max_len: int,
    min_support: int,
    top_k: int,
    examples_per_card: int,
    excluded_substrings: tuple[str, ...],
) -> list[QuestionMotifBundle]:
    grouped: dict[str, list[TraceRecord]] = defaultdict(list)
    for record in records:
        grouped[record.question_id].append(record)

    bundles: list[QuestionMotifBundle] = []
    for question_id, question_records in sorted(grouped.items(), key=lambda item: int(item[0])):
        local_cards = build_motif_cards(
            question_records,
            min_len=min_len,
            max_len=max_len,
            min_support=min_support,
            examples_per_card=examples_per_card,
            excluded_substrings=excluded_substrings,
        )
        local_families = top_card_families(local_cards, top_k=top_k, min_question_coverage=1)
        question_token_sets = [record.tokens for record in question_records]

        def visible(cards: list[MotifCard]) -> list[MotifCard]:
            out: list[MotifCard] = []
            for card in cards:
                motif = tuple(card.motif_tokens)
                if any(contains_motif(tokens, motif) for tokens in question_token_sets):
                    out.append(card)
            return out[:top_k]

        bundles.append(
            QuestionMotifBundle(
                question_id=question_id,
                trace_count=len(question_records),
                success_count=sum(1 for record in question_records if record.is_correct),
                failure_count=sum(1 for record in question_records if not record.is_correct),
                avg_token_count=statistics.mean(len(record.tokens) for record in question_records),
                rows_with_noise=sum(1 for record in question_records if "noise:corrupted-span" in record.tokens),
                local_top=local_families["global_top"],
                local_success_enriched=local_families["success_enriched"],
                local_failure_enriched=local_families["failure_enriched"],
                visible_global_top=visible(global_families["global_top"]),
                visible_global_success=visible(global_families["success_enriched"]),
                visible_global_failure=visible(global_families["failure_enriched"]),
            )
        )
    return bundles


def write_cards_csv(path: Path, cards: Iterable[MotifCard]) -> None:
    fieldnames = [
        "motif",
        "motif_length",
        "support",
        "prevalence",
        "question_coverage",
        "success_present",
        "failure_present",
        "success_rate_present",
        "success_rate_absent",
        "support_difference",
        "lift_overall_success",
        "odds_ratio",
        "odds_ratio_ci_low",
        "odds_ratio_ci_high",
        "fisher_p_value",
        "fdr_q_value",
        "avg_stage_position",
        "stage_bucket",
        "avg_trace_length_present",
        "noise_cooccurrence_rate",
        "top_neighbor_motifs",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for card in cards:
            writer.writerow(
                {
                    "motif": card.motif,
                    "motif_length": card.motif_length,
                    "support": card.support,
                    "prevalence": f"{card.prevalence:.6f}",
                    "question_coverage": card.question_coverage,
                    "success_present": card.success_present,
                    "failure_present": card.failure_present,
                    "success_rate_present": f"{card.success_rate_present:.6f}",
                    "success_rate_absent": f"{card.success_rate_absent:.6f}",
                    "support_difference": f"{card.support_difference:.6f}",
                    "lift_overall_success": f"{card.lift_overall_success:.6f}",
                    "odds_ratio": f"{card.odds_ratio:.6f}",
                    "odds_ratio_ci_low": f"{card.odds_ratio_ci_low:.6f}",
                    "odds_ratio_ci_high": f"{card.odds_ratio_ci_high:.6f}",
                    "fisher_p_value": f"{card.fisher_p_value:.8f}",
                    "fdr_q_value": f"{card.fdr_q_value:.8f}",
                    "avg_stage_position": f"{card.avg_stage_position:.6f}",
                    "stage_bucket": card.stage_bucket,
                    "avg_trace_length_present": f"{card.avg_trace_length_present:.6f}",
                    "noise_cooccurrence_rate": f"{card.noise_cooccurrence_rate:.6f}",
                    "top_neighbor_motifs": " || ".join(card.top_neighbor_motifs),
                }
            )


def coverage_for_cards(records: list[TraceRecord], cards: list[MotifCard]) -> dict[str, object]:
    chosen = {tuple(card.motif_tokens) for card in cards}
    covered_trace_count = 0
    covered_questions: set[str] = set()
    for record in records:
        features = unique_contiguous_motifs(record.tokens, min_len=1, max_len=3)
        if any(feature in chosen for feature in features):
            covered_trace_count += 1
            covered_questions.add(record.question_id)
    return {
        "covered_trace_count": covered_trace_count,
        "covered_trace_rate": covered_trace_count / len(records),
        "covered_question_count": len(covered_questions),
        "covered_question_rate": len(covered_questions) / len({record.question_id for record in records}),
    }


def run_card_compression_experiment(
    rows_path: Path,
    *,
    folds: int,
    seed: int,
    min_support: int,
    top_k_values: list[int],
) -> list[dict[str, object]]:
    rows = load_rows(rows_path)
    test_folds = build_question_folds(rows, folds, seed)
    results: list[dict[str, object]] = []

    for top_k in top_k_values:
        fold_aucs: list[float] = []
        fold_q_local_aucs: list[float] = []
        retained_feature_counts: list[int] = []

        for test_question_ids in test_folds:
            train_rows = [row for row in rows if row.question_id not in test_question_ids]
            test_rows = [row for row in rows if row.question_id in test_question_ids]
            train_features = build_feature_sets(
                train_rows,
                prefix_fraction=1.0,
                min_len=1,
                max_len=3,
            )
            test_features = build_feature_sets(
                test_rows,
                prefix_fraction=1.0,
                min_len=1,
                max_len=3,
            )
            constant, deltas, _ = train_bernoulli_nb(
                train_rows,
                train_features,
                min_support=min_support,
            )
            retained = dict(
                sorted(deltas.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
            )
            retained_feature_counts.append(len(retained))
            scores = score_bernoulli_nb(test_features, constant=constant, deltas=retained)
            labels = [row.is_correct for row in test_rows]
            fold_aucs.append(roc_auc(list(zip(scores, labels))))

            grouped: dict[str, list[tuple[float, bool]]] = defaultdict(list)
            for row, score in zip(test_rows, scores):
                grouped[row.question_id].append((score, row.is_correct))
            q_aucs = [
                roc_auc(group_rows)
                for group_rows in grouped.values()
                if any(label for _, label in group_rows) and any((not label) for _, label in group_rows)
            ]
            fold_q_local_aucs.append(sum(q_aucs) / len(q_aucs) if q_aucs else 0.5)

        results.append(
            {
                "top_k": top_k,
                "motif_auc": sum(fold_aucs) / len(fold_aucs),
                "motif_question_local_auc": sum(fold_q_local_aucs) / len(fold_q_local_aucs),
                "avg_feature_count": sum(retained_feature_counts) / len(retained_feature_counts),
            }
        )

    return results


def run_length_matched_experiment(
    rows_path: Path,
    *,
    folds: int,
    seed: int,
    min_support: int,
    max_length_gap: int,
) -> dict[str, object]:
    rows = load_rows(rows_path)
    test_folds = build_question_folds(rows, folds, seed)

    motif_pair_hits = 0
    length_pair_hits = 0
    total_pairs = 0

    for test_question_ids in test_folds:
        train_rows = [row for row in rows if row.question_id not in test_question_ids]
        test_rows = [row for row in rows if row.question_id in test_question_ids]
        train_features = build_feature_sets(train_rows, prefix_fraction=1.0, min_len=1, max_len=3)
        test_features = build_feature_sets(test_rows, prefix_fraction=1.0, min_len=1, max_len=3)
        constant, deltas, _ = train_bernoulli_nb(train_rows, train_features, min_support=min_support)
        motif_scores = score_bernoulli_nb(test_features, constant=constant, deltas=deltas)
        lengths = [len(row.tokens) for row in test_rows]

        grouped: dict[str, list[tuple[int, float, int, bool]]] = defaultdict(list)
        for index, row in enumerate(test_rows):
            grouped[row.question_id].append((index, motif_scores[index], lengths[index], row.is_correct))

        for group_rows in grouped.values():
            correct = [item for item in group_rows if item[3]]
            failure = [item for item in group_rows if not item[3]]
            for c_index, c_score, c_len, _ in correct:
                best_match: tuple[int, float, int, bool] | None = None
                best_gap: int | None = None
                for candidate in failure:
                    gap = abs(c_len - candidate[2])
                    if gap > max_length_gap:
                        continue
                    if best_gap is None or gap < best_gap:
                        best_gap = gap
                        best_match = candidate
                if best_match is None:
                    continue
                total_pairs += 1
                motif_pair_hits += 1 if c_score > best_match[1] else 0
                if c_len > best_match[2]:
                    length_pair_hits += 1
                elif c_len == best_match[2]:
                    length_pair_hits += 0.5

    return {
        "max_length_gap": max_length_gap,
        "matched_pair_count": total_pairs,
        "motif_pair_accuracy": (motif_pair_hits / total_pairs) if total_pairs else 0.0,
        "length_pair_accuracy": (length_pair_hits / total_pairs) if total_pairs else 0.0,
    }


def render_markdown_report(
    summary: dict[str, object],
    families: dict[str, list[MotifCard]],
    question_bundles: list[QuestionMotifBundle],
    coverage: dict[str, object],
    predictive: dict[str, object],
    compression: list[dict[str, object]],
    matched: dict[str, object],
) -> str:
    lines: list[str] = []
    lines.append("# Motif Card Report")
    lines.append("")
    lines.append("## Dataset snapshot")
    lines.append("")
    lines.append(f"- Traces: {summary['trace_count']}")
    lines.append(f"- Questions: {summary['question_count']}")
    lines.append(f"- Success / failure: {summary['success_count']} / {summary['failure_count']}")
    lines.append(
        f"- Avg token count: {float(summary['avg_token_count']):.2f} (median {float(summary['median_token_count']):.1f})"
    )
    lines.append(
        f"- Rows with noise: {summary['rows_with_noise']} ({100.0 * float(summary['rows_with_noise_rate']):.1f}%)"
    )
    lines.append(f"- Unique tokens: {summary['unique_token_count']}")
    lines.append(f"- Per-question bundles emitted: {len(question_bundles)}")
    lines.append("")
    lines.append("## Why motif cards look useful")
    lines.append("")
    lines.append(
        f"- Top global cards cover {int(coverage['covered_trace_count'])} / {summary['trace_count']} traces "
        f"({100.0 * float(coverage['covered_trace_rate']):.1f}%) and {int(coverage['covered_question_count'])} / "
        f"{summary['question_count']} questions."
    )
    lines.append(
        f"- Full motif model AUC: {float(predictive['motif_auc']):.4f}; question-local AUC: "
        f"{float(predictive['motif_question_local_auc_mean']):.4f}."
    )
    lines.append(
        f"- Length baseline AUC: {float(predictive['length_auc']):.4f}; question-local AUC: "
        f"{float(predictive['length_question_local_auc_mean']):.4f}."
    )
    lines.append(
        f"- In length-matched within-question pairs (gap <= {matched['max_length_gap']}), motif scores rank the correct trace above the incorrect one "
        f"{100.0 * float(matched['motif_pair_accuracy']):.1f}% of the time vs "
        f"{100.0 * float(matched['length_pair_accuracy']):.1f}% for raw length."
    )
    lines.append("")
    lines.append("## Compression experiment")
    lines.append("")
    lines.append("| Top-K cards | Motif AUC | Question-local AUC | Avg retained features |")
    lines.append("|---|---:|---:|---:|")
    for row in compression:
        lines.append(
            f"| {row['top_k']} | {float(row['motif_auc']):.4f} | {float(row['motif_question_local_auc']):.4f} | {float(row['avg_feature_count']):.1f} |"
        )
    lines.append("")

    def add_family(title: str, cards: list[MotifCard]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Motif | Support | Questions | Success when present | Success when absent | Stage | q-value |")
        lines.append("|---|---:|---:|---:|---:|---|---:|")
        for card in cards:
            lines.append(
                f"| `{card.motif}` | {card.support} | {card.question_coverage} | {card.success_rate_present:.3f} | "
                f"{card.success_rate_absent:.3f} | {card.stage_bucket} | {card.fdr_q_value:.4g} |"
            )
        lines.append("")

    add_family("Top Global Cards", families["global_top"])
    add_family("Success-Enriched Cards", families["success_enriched"])
    add_family("Failure-Enriched Cards", families["failure_enriched"])
    lines.append("## Frontend Notes")
    lines.append("")
    lines.append("- Motifs are contiguous subsequences of token length 1 to 3.")
    lines.append("- The JSON bundle includes both global cards and per-question local cards.")
    lines.append("- Per-question bundles expose local top cards plus the subset of global cards visible in that question.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    excluded_substrings = parse_excluded_substrings(args.exclude_substrings)

    records = load_trace_records(args.input_csv)
    summary = build_dataset_summary(records)
    cards = build_motif_cards(
        records,
        min_len=args.min_len,
        max_len=args.max_len,
        min_support=args.min_support,
        examples_per_card=args.examples_per_card,
        excluded_substrings=excluded_substrings,
    )
    families = top_card_families(cards, top_k=args.top_k, min_question_coverage=2)
    question_bundles = build_question_bundles(
        records,
        global_families=families,
        min_len=args.min_len,
        max_len=args.max_len,
        min_support=args.per_question_min_support,
        top_k=args.per_question_top_k,
        examples_per_card=args.examples_per_card,
        excluded_substrings=excluded_substrings,
    )

    predictive = run_question_holdout_experiment(
        load_rows(args.input_csv),
        folds=args.folds,
        seed=args.seed,
        min_support=args.min_support,
        prefix_fraction=1.0,
        min_len=args.min_len,
        max_len=args.max_len,
        experiment_name="motif_cards_full",
    )["aggregate"]
    coverage = coverage_for_cards(
        records,
        families["global_top"],
    )
    compression = run_card_compression_experiment(
        args.input_csv,
        folds=args.folds,
        seed=args.seed,
        min_support=args.min_support,
        top_k_values=[5, 10, 20, 40],
    )
    matched = run_length_matched_experiment(
        args.input_csv,
        folds=args.folds,
        seed=args.seed,
        min_support=args.min_support,
        max_length_gap=1,
    )

    summary_path = args.output_dir / "dataset_summary.json"
    cards_json_path = args.output_dir / "motif_cards.json"
    cards_csv_path = args.output_dir / "motif_cards.csv"
    families_json_path = args.output_dir / "motif_card_families.json"
    question_bundles_json_path = args.output_dir / "question_motif_cards.json"
    experiments_json_path = args.output_dir / "motif_card_experiments.json"
    report_path = args.output_dir / "motif_card_report.md"

    summary["motif_definition"] = {
        "type": "contiguous_subsequence",
        "min_len": args.min_len,
        "max_len": args.max_len,
        "excluded_substrings": list(excluded_substrings),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cards_json_path.write_text(
        json.dumps([asdict(card) for card in cards], indent=2),
        encoding="utf-8",
    )
    write_cards_csv(cards_csv_path, cards)
    families_json_path.write_text(
        json.dumps(
            {name: [asdict(card) for card in family] for name, family in families.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    question_bundles_json_path.write_text(
        json.dumps([asdict(bundle) for bundle in question_bundles], indent=2),
        encoding="utf-8",
    )
    experiments_json_path.write_text(
        json.dumps(
            {
                "predictive": predictive,
                "coverage": coverage,
                "compression": compression,
                "length_matched": matched,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        render_markdown_report(summary, families, question_bundles, coverage, predictive, compression, matched),
        encoding="utf-8",
    )

    print(f"dataset_summary: {summary_path}")
    print(f"motif_cards_json: {cards_json_path}")
    print(f"motif_cards_csv: {cards_csv_path}")
    print(f"motif_families_json: {families_json_path}")
    print(f"question_bundles_json: {question_bundles_json_path}")
    print(f"experiments_json: {experiments_json_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
