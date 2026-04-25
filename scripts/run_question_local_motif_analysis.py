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


DEFAULT_GPT_OSS_INPUT = Path(
    "tokenizer/clean_expanded_pool_s100_seed73_qwen25_7b_hot30_tokenized_gpt-oss-120b_0.csv"
)
DEFAULT_DEEPSEEK_INPUT = Path(
    "tokenizer/clean_expanded_pool_s100_seed73_qwen25_7b_hot30_tokenized_DeepSeek-V3.1_0.csv"
)
DEFAULT_OUTPUT_DIR = Path("paper_experiments/question_local")

TRUTHY_VALUES = {"true", "1", "yes", "y", "t"}
FALSY_VALUES = {"false", "0", "no", "n", "f"}


@dataclass(frozen=True)
class TraceRow:
    question_id: str
    sample_id: str
    is_correct: bool
    tokens: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run question-local motif analysis. Motifs are mined independently within each "
            "question and permutation-tested within that same question."
        )
    )
    parser.add_argument("--gpt-oss-input", type=Path, default=DEFAULT_GPT_OSS_INPUT)
    parser.add_argument("--deepseek-input", type=Path, default=DEFAULT_DEEPSEEK_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-motif-len", type=int, default=1)
    parser.add_argument("--max-motif-len", type=int, default=3)
    parser.add_argument("--min-class-size", type=int, default=2)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--q-value-threshold", type=float, default=0.10)
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
            rows.append(
                TraceRow(
                    question_id=str(row["question_id"]),
                    sample_id=str(row["sample_id"]),
                    is_correct=parse_bool(row["is_correct"]),
                    tokens=tuple(tok for tok in row["tokenized_trace"].split() if tok),
                )
            )
    return rows


def unique_contiguous_motifs(tokens: tuple[str, ...], min_len: int, max_len: int) -> set[str]:
    motifs: set[str] = set()
    if not tokens:
        return motifs
    local_max = min(max_len, len(tokens))
    for motif_len in range(max(1, min_len), local_max + 1):
        for start in range(0, len(tokens) - motif_len + 1):
            motifs.add(" ".join(tokens[start : start + motif_len]))
    return motifs


def smoothed_log_odds(
    correct_count: int,
    incorrect_count: int,
    total_correct: int,
    total_incorrect: int,
    alpha: float = 0.5,
) -> float:
    a = correct_count + alpha
    b = (total_correct - correct_count) + alpha
    c = incorrect_count + alpha
    d = (total_incorrect - incorrect_count) + alpha
    return math.log((a / b) / (c / d))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    n = len(p_values)
    q_values = [1.0] * n
    running_min = 1.0
    for reverse_rank, (original_index, p_value) in enumerate(reversed(indexed), start=1):
        rank = n - reverse_rank + 1
        adjusted = min(1.0, p_value * n / rank)
        running_min = min(running_min, adjusted)
        q_values[original_index] = running_min
    return q_values


def motif_delta(
    motif: str,
    motif_presence: list[set[str]],
    labels: list[bool],
) -> tuple[int, int, float]:
    correct_count = 0
    incorrect_count = 0
    for features, label in zip(motif_presence, labels):
        if motif in features:
            if label:
                correct_count += 1
            else:
                incorrect_count += 1
    total_correct = sum(1 for label in labels if label)
    total_incorrect = len(labels) - total_correct
    correct_support = correct_count / total_correct if total_correct else 0.0
    incorrect_support = incorrect_count / total_incorrect if total_incorrect else 0.0
    return correct_count, incorrect_count, correct_support - incorrect_support


def permutation_p_value(
    motif: str,
    motif_presence: list[set[str]],
    labels: list[bool],
    *,
    permutations: int,
    rng: random.Random,
) -> float:
    _, _, observed = motif_delta(motif, motif_presence, labels)
    if permutations <= 0:
        return 1.0

    n = len(labels)
    n_correct = sum(1 for label in labels if label)
    indices = list(range(n))
    abs_observed = abs(observed)
    extreme = 1

    for _ in range(permutations):
        correct_indices = set(rng.sample(indices, n_correct))
        shuffled_labels = [idx in correct_indices for idx in indices]
        _, _, permuted = motif_delta(motif, motif_presence, shuffled_labels)
        if abs(permuted) >= abs_observed:
            extreme += 1

    return extreme / (permutations + 1)


def analyze_question(
    rows: list[TraceRow],
    *,
    min_len: int,
    max_len: int,
    min_support: int,
    top_k: int,
    permutations: int,
    seed: int,
    q_value_threshold: float,
) -> dict[str, object] | None:
    correct_rows = [row for row in rows if row.is_correct]
    incorrect_rows = [row for row in rows if not row.is_correct]
    if not correct_rows or not incorrect_rows:
        return None

    motif_presence = [unique_contiguous_motifs(row.tokens, min_len, max_len) for row in rows]
    labels = [row.is_correct for row in rows]
    total_correct = len(correct_rows)
    total_incorrect = len(incorrect_rows)

    correct_counts: Counter[str] = Counter()
    incorrect_counts: Counter[str] = Counter()
    for features, label in zip(motif_presence, labels):
        if label:
            correct_counts.update(features)
        else:
            incorrect_counts.update(features)

    motif_rows: list[dict[str, object]] = []
    motifs = set(correct_counts) | set(incorrect_counts)
    rng = random.Random(seed + int(rows[0].question_id))

    for motif in motifs:
        support = correct_counts[motif] + incorrect_counts[motif]
        if support < min_support:
            continue

        correct_count, incorrect_count, support_delta = motif_delta(motif, motif_presence, labels)
        p_correct = correct_count / support
        p_incorrect = incorrect_count / support
        lor = smoothed_log_odds(correct_count, incorrect_count, total_correct, total_incorrect)
        p_value = permutation_p_value(
            motif,
            motif_presence,
            labels,
            permutations=permutations,
            rng=rng,
        )
        motif_rows.append(
            {
                "motif": motif,
                "support_total": support,
                "support_correct": correct_count,
                "support_incorrect": incorrect_count,
                "p_correct_given_motif": p_correct,
                "p_incorrect_given_motif": p_incorrect,
                "support_delta": support_delta,
                "log_odds_ratio": lor,
                "p_value": p_value,
            }
        )

    if not motif_rows:
        return None

    q_values = benjamini_hochberg([float(row["p_value"]) for row in motif_rows])
    for row, q_value in zip(motif_rows, q_values):
        row["q_value"] = q_value

    success_rows = sorted(
        [row for row in motif_rows if float(row["log_odds_ratio"]) > 0],
        key=lambda row: (float(row["q_value"]), -float(row["log_odds_ratio"]), -int(row["support_total"])),
    )[:top_k]
    failure_rows = sorted(
        [row for row in motif_rows if float(row["log_odds_ratio"]) < 0],
        key=lambda row: (float(row["q_value"]), -abs(float(row["log_odds_ratio"])), -int(row["support_total"])),
    )[:top_k]
    significant_success = sorted(
        [
            row
            for row in motif_rows
            if float(row["log_odds_ratio"]) > 0 and float(row["q_value"]) <= q_value_threshold
        ],
        key=lambda row: (float(row["q_value"]), -float(row["log_odds_ratio"])),
    )[:top_k]
    significant_failure = sorted(
        [
            row
            for row in motif_rows
            if float(row["log_odds_ratio"]) < 0 and float(row["q_value"]) <= q_value_threshold
        ],
        key=lambda row: (float(row["q_value"]), -abs(float(row["log_odds_ratio"]))),
    )[:top_k]

    return {
        "question_id": rows[0].question_id,
        "total_traces": len(rows),
        "correct_traces": total_correct,
        "incorrect_traces": total_incorrect,
        "num_candidate_motifs": len(motif_rows),
        "num_significant_success_motifs": len(
            [
                row
                for row in motif_rows
                if float(row["log_odds_ratio"]) > 0 and float(row["q_value"]) <= q_value_threshold
            ]
        ),
        "num_significant_failure_motifs": len(
            [
                row
                for row in motif_rows
                if float(row["log_odds_ratio"]) < 0 and float(row["q_value"]) <= q_value_threshold
            ]
        ),
        "top_success_motifs": success_rows,
        "top_failure_motifs": failure_rows,
        "significant_success_motifs": significant_success,
        "significant_failure_motifs": significant_failure,
    }


def analyze_dataset(
    rows: list[TraceRow],
    *,
    min_len: int,
    max_len: int,
    min_class_size: int,
    min_support: int,
    top_k: int,
    permutations: int,
    seed: int,
    q_value_threshold: float,
) -> dict[str, object]:
    grouped: dict[str, list[TraceRow]] = defaultdict(list)
    for row in rows:
        grouped[row.question_id].append(row)

    question_results: list[dict[str, object]] = []
    for question_id, question_rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        correct_n = sum(1 for row in question_rows if row.is_correct)
        incorrect_n = len(question_rows) - correct_n
        if correct_n < min_class_size or incorrect_n < min_class_size:
            continue
        result = analyze_question(
            question_rows,
            min_len=min_len,
            max_len=max_len,
            min_support=min_support,
            top_k=top_k,
            permutations=permutations,
            seed=seed,
            q_value_threshold=q_value_threshold,
        )
        if result is not None:
            question_results.append(result)

    return {
        "question_count": len(question_results),
        "questions_with_any_significant_motif": sum(
            1
            for row in question_results
            if row["num_significant_success_motifs"] > 0 or row["num_significant_failure_motifs"] > 0
        ),
        "questions": question_results,
    }


def build_local_overlap(
    left: dict[str, object],
    right: dict[str, object],
) -> dict[str, object]:
    left_by_q = {str(item["question_id"]): item for item in left["questions"]}
    right_by_q = {str(item["question_id"]): item for item in right["questions"]}
    shared_questions = sorted(set(left_by_q) & set(right_by_q), key=int)

    overlap_rows: list[dict[str, object]] = []
    for qid in shared_questions:
        left_q = left_by_q[qid]
        right_q = right_by_q[qid]

        left_success = {row["motif"] for row in left_q["top_success_motifs"]}
        right_success = {row["motif"] for row in right_q["top_success_motifs"]}
        left_failure = {row["motif"] for row in left_q["top_failure_motifs"]}
        right_failure = {row["motif"] for row in right_q["top_failure_motifs"]}

        left_sig_success = {row["motif"] for row in left_q["significant_success_motifs"]}
        right_sig_success = {row["motif"] for row in right_q["significant_success_motifs"]}
        left_sig_failure = {row["motif"] for row in left_q["significant_failure_motifs"]}
        right_sig_failure = {row["motif"] for row in right_q["significant_failure_motifs"]}

        overlap_rows.append(
            {
                "question_id": qid,
                "success_overlap_count": len(left_success & right_success),
                "failure_overlap_count": len(left_failure & right_failure),
                "significant_success_overlap_count": len(left_sig_success & right_sig_success),
                "significant_failure_overlap_count": len(left_sig_failure & right_sig_failure),
                "shared_success_motifs": sorted(left_success & right_success),
                "shared_failure_motifs": sorted(left_failure & right_failure),
                "shared_significant_success_motifs": sorted(left_sig_success & right_sig_success),
                "shared_significant_failure_motifs": sorted(left_sig_failure & right_sig_failure),
            }
        )

    return {
        "shared_question_count": len(shared_questions),
        "questions": overlap_rows,
        "mean_success_overlap": (
            sum(row["success_overlap_count"] for row in overlap_rows) / len(overlap_rows)
            if overlap_rows
            else 0.0
        ),
        "mean_failure_overlap": (
            sum(row["failure_overlap_count"] for row in overlap_rows) / len(overlap_rows)
            if overlap_rows
            else 0.0
        ),
        "mean_significant_success_overlap": (
            sum(row["significant_success_overlap_count"] for row in overlap_rows) / len(overlap_rows)
            if overlap_rows
            else 0.0
        ),
        "mean_significant_failure_overlap": (
            sum(row["significant_failure_overlap_count"] for row in overlap_rows) / len(overlap_rows)
            if overlap_rows
            else 0.0
        ),
    }


def write_question_csv(path: Path, dataset: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "total_traces",
                "correct_traces",
                "incorrect_traces",
                "num_candidate_motifs",
                "num_significant_success_motifs",
                "num_significant_failure_motifs",
                "top_success_motifs",
                "top_failure_motifs",
                "significant_success_motifs",
                "significant_failure_motifs",
            ],
        )
        writer.writeheader()
        for row in dataset["questions"]:
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "total_traces": row["total_traces"],
                    "correct_traces": row["correct_traces"],
                    "incorrect_traces": row["incorrect_traces"],
                    "num_candidate_motifs": row["num_candidate_motifs"],
                    "num_significant_success_motifs": row["num_significant_success_motifs"],
                    "num_significant_failure_motifs": row["num_significant_failure_motifs"],
                    "top_success_motifs": json.dumps(row["top_success_motifs"], ensure_ascii=True),
                    "top_failure_motifs": json.dumps(row["top_failure_motifs"], ensure_ascii=True),
                    "significant_success_motifs": json.dumps(row["significant_success_motifs"], ensure_ascii=True),
                    "significant_failure_motifs": json.dumps(row["significant_failure_motifs"], ensure_ascii=True),
                }
            )


def write_report(
    path: Path,
    *,
    gpt: dict[str, object],
    deepseek: dict[str, object],
    overlap: dict[str, object],
) -> None:
    lines = [
        "# Question-Local Motif Analysis",
        "",
        "Motifs in this report are mined independently within each question.",
        "There is no global motif pool across questions.",
        "",
        f"- gpt_oss analyzable mixed questions: {gpt['question_count']}",
        f"- gpt_oss questions with any significant local motif: {gpt['questions_with_any_significant_motif']}",
        f"- deepseek analyzable mixed questions: {deepseek['question_count']}",
        f"- deepseek questions with any significant local motif: {deepseek['questions_with_any_significant_motif']}",
        f"- shared analyzable questions: {overlap['shared_question_count']}",
        f"- mean success overlap across tokenizers: {float(overlap['mean_success_overlap']):.3f}",
        f"- mean failure overlap across tokenizers: {float(overlap['mean_failure_overlap']):.3f}",
        f"- mean significant success overlap across tokenizers: {float(overlap['mean_significant_success_overlap']):.3f}",
        f"- mean significant failure overlap across tokenizers: {float(overlap['mean_significant_failure_overlap']):.3f}",
        "",
        "## Example Shared Local Overlap",
        "",
        "| Question | Success Overlap | Failure Overlap | Sig Success Overlap | Sig Failure Overlap | Shared Significant Success | Shared Significant Failure |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    rows = sorted(
        overlap["questions"],
        key=lambda row: (
            row["significant_success_overlap_count"] + row["significant_failure_overlap_count"],
            row["success_overlap_count"] + row["failure_overlap_count"],
        ),
        reverse=True,
    )
    for row in rows[:20]:
        lines.append(
            f"| {row['question_id']} | {row['success_overlap_count']} | {row['failure_overlap_count']} | "
            f"{row['significant_success_overlap_count']} | {row['significant_failure_overlap_count']} | "
            f"{', '.join(row['shared_significant_success_motifs']) if row['shared_significant_success_motifs'] else '-'} | "
            f"{', '.join(row['shared_significant_failure_motifs']) if row['shared_significant_failure_motifs'] else '-'} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gpt_rows = load_rows(args.gpt_oss_input)
    deepseek_rows = load_rows(args.deepseek_input)

    gpt = analyze_dataset(
        gpt_rows,
        min_len=args.min_motif_len,
        max_len=args.max_motif_len,
        min_class_size=args.min_class_size,
        min_support=args.min_support,
        top_k=args.top_k,
        permutations=args.permutations,
        seed=args.seed,
        q_value_threshold=args.q_value_threshold,
    )
    deepseek = analyze_dataset(
        deepseek_rows,
        min_len=args.min_motif_len,
        max_len=args.max_motif_len,
        min_class_size=args.min_class_size,
        min_support=args.min_support,
        top_k=args.top_k,
        permutations=args.permutations,
        seed=args.seed,
        q_value_threshold=args.q_value_threshold,
    )
    overlap = build_local_overlap(gpt, deepseek)

    summary = {
        "config": {
            "min_motif_len": args.min_motif_len,
            "max_motif_len": args.max_motif_len,
            "min_class_size": args.min_class_size,
            "min_support": args.min_support,
            "top_k": args.top_k,
            "permutations": args.permutations,
            "seed": args.seed,
            "q_value_threshold": args.q_value_threshold,
        },
        "gpt_oss": gpt,
        "deepseek": deepseek,
        "cross_tokenizer_local_overlap": overlap,
    }

    (output_dir / "question_local_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_question_csv(output_dir / "gpt_oss_question_local.csv", gpt)
    write_question_csv(output_dir / "deepseek_question_local.csv", deepseek)
    write_report(output_dir / "question_local_report.md", gpt=gpt, deepseek=deepseek, overlap=overlap)

    print(f"Wrote summary JSON to {output_dir / 'question_local_summary.json'}")
    print(f"Wrote report to {output_dir / 'question_local_report.md'}")


if __name__ == "__main__":
    main()
