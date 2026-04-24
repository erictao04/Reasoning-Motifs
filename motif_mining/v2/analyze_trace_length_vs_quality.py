from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import median

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PER_QUESTION_CSV = BASE_DIR / "trace_length_per_question_stats.csv"
DEFAULT_AGG_CSV = BASE_DIR / "trace_length_aggregate_stats.csv"
DEFAULT_REPORT_MD = BASE_DIR / "trace_length_analysis_report.md"

TRUTHY_VALUES = {"true", "1", "yes", "y", "t"}
FALSY_VALUES = {"false", "0", "no", "n", "f"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether trace length indicates trace correctness while controlling "
            "for question difficulty by analyzing each question independently."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to tokenized traces CSV.")
    parser.add_argument(
        "--per-question-csv",
        type=Path,
        default=DEFAULT_PER_QUESTION_CSV,
        help=f"Output CSV path for per-question stats (default: {DEFAULT_PER_QUESTION_CSV}).",
    )
    parser.add_argument(
        "--aggregate-csv",
        type=Path,
        default=DEFAULT_AGG_CSV,
        help=f"Output CSV path for aggregate stats (default: {DEFAULT_AGG_CSV}).",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=DEFAULT_REPORT_MD,
        help=f"Output markdown report path (default: {DEFAULT_REPORT_MD}).",
    )
    parser.add_argument(
        "--question-column",
        default="question_id",
        help="Column containing question IDs (default: question_id).",
    )
    parser.add_argument(
        "--label-column",
        default="is_correct",
        help="Column containing correctness labels (default: is_correct).",
    )
    parser.add_argument(
        "--trace-column",
        default="tokenized_trace",
        help="Column containing tokenized traces (default: tokenized_trace).",
    )
    parser.add_argument(
        "--min-class-size",
        type=int,
        default=2,
        help="Minimum examples required in each class per question (default: 2).",
    )
    parser.add_argument(
        "--per-question-permutations",
        type=int,
        default=2000,
        help="Permutation count for per-question p-values (default: 2000).",
    )
    parser.add_argument(
        "--aggregate-permutations",
        type=int,
        default=5000,
        help="Permutation count for aggregate stratified test (default: 5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=73,
        help="Random seed for reproducibility (default: 73).",
    )
    parser.add_argument(
        "--include-missing-as-zero",
        action="store_true",
        help="Treat missing traces (empty or MISSING) as zero length instead of skipping.",
    )
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in TRUTHY_VALUES:
        return True
    if normalized in FALSY_VALUES:
        return False
    raise ValueError(f"Unrecognized boolean value: {value!r}")


def trace_length(raw_trace: str, include_missing_as_zero: bool) -> int | None:
    cleaned = raw_trace.strip()
    if not cleaned or cleaned.upper() == "MISSING":
        return 0 if include_missing_as_zero else None
    return len([token for token in cleaned.split() if token])


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def sample_variance(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / (n - 1)


def cohen_d(correct_lengths: list[int], incorrect_lengths: list[int]) -> float:
    n1 = len(correct_lengths)
    n0 = len(incorrect_lengths)
    if n1 < 2 or n0 < 2:
        return 0.0

    var1 = sample_variance([float(x) for x in correct_lengths])
    var0 = sample_variance([float(x) for x in incorrect_lengths])
    pooled_var_num = (n1 - 1) * var1 + (n0 - 1) * var0
    pooled_var_den = (n1 + n0 - 2)
    if pooled_var_den <= 0:
        return 0.0

    pooled_sd = math.sqrt(max(pooled_var_num / pooled_var_den, 0.0))
    if pooled_sd == 0.0:
        return 0.0

    return (mean([float(x) for x in correct_lengths]) - mean([float(x) for x in incorrect_lengths])) / pooled_sd


def common_language_effect(correct_lengths: list[int], incorrect_lengths: list[int]) -> float:
    """
    Probability that a random correct trace is longer than a random incorrect trace,
    with ties counting as 0.5.
    """
    n1 = len(correct_lengths)
    n0 = len(incorrect_lengths)
    if n1 == 0 or n0 == 0:
        return 0.5

    wins = 0.0
    for c in correct_lengths:
        for i in incorrect_lengths:
            if c > i:
                wins += 1.0
            elif c == i:
                wins += 0.5
    return wins / (n1 * n0)


def delta_mean_for_labels(lengths: list[int], labels: list[bool]) -> float:
    correct = [length for length, label in zip(lengths, labels) if label]
    incorrect = [length for length, label in zip(lengths, labels) if not label]
    if not correct or not incorrect:
        return 0.0
    return mean([float(x) for x in correct]) - mean([float(x) for x in incorrect])


def permutation_p_value_delta(
    lengths: list[int],
    labels: list[bool],
    permutations: int,
    rng: random.Random,
) -> tuple[float, float]:
    observed = delta_mean_for_labels(lengths, labels)
    if permutations <= 0:
        return observed, 1.0

    n_correct = sum(1 for x in labels if x)
    n_total = len(labels)
    if n_correct == 0 or n_correct == n_total:
        return observed, 1.0

    abs_observed = abs(observed)
    extreme = 1  # +1 smoothing

    indices = list(range(n_total))
    for _ in range(permutations):
        correct_idx = set(rng.sample(indices, n_correct))
        shuffled_labels = [idx in correct_idx for idx in indices]
        perm_delta = delta_mean_for_labels(lengths, shuffled_labels)
        if abs(perm_delta) >= abs_observed:
            extreme += 1

    p_value = extreme / (permutations + 1)
    return observed, p_value


def load_grouped_data(
    input_path: Path,
    question_col: str,
    label_col: str,
    trace_col: str,
    include_missing_as_zero: bool,
) -> tuple[dict[str, list[tuple[int, bool]]], int, int]:
    grouped: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    skipped_rows = 0
    total_rows = 0

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Input file has no header: {input_path}")

        required = [question_col, label_col, trace_col]
        missing = [col for col in required if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns {missing} in input header {reader.fieldnames}.")

        for row in reader:
            total_rows += 1
            qid_raw = row.get(question_col)
            label_raw = row.get(label_col)
            trace_raw = row.get(trace_col)

            if qid_raw is None or label_raw is None or trace_raw is None:
                skipped_rows += 1
                continue

            try:
                is_correct = parse_bool(label_raw)
            except ValueError:
                skipped_rows += 1
                continue

            length = trace_length(trace_raw, include_missing_as_zero)
            if length is None:
                skipped_rows += 1
                continue

            grouped[str(qid_raw)].append((length, is_correct))

    return grouped, total_rows, skipped_rows


def analyze_per_question(
    grouped: dict[str, list[tuple[int, bool]]],
    min_class_size: int,
    permutations: int,
    rng: random.Random,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, object]]]:
    per_question_rows: list[dict[str, float | int | str]] = []
    valid_groups: list[dict[str, object]] = []

    for question_id, items in grouped.items():
        lengths = [length for length, _ in items]
        labels = [label for _, label in items]

        correct_lengths = [length for length, label in items if label]
        incorrect_lengths = [length for length, label in items if not label]

        n_correct = len(correct_lengths)
        n_incorrect = len(incorrect_lengths)
        n_total = len(items)

        if n_correct < min_class_size or n_incorrect < min_class_size:
            per_question_rows.append(
                {
                    "question_id": question_id,
                    "n_total": n_total,
                    "n_correct": n_correct,
                    "n_incorrect": n_incorrect,
                    "mean_len_correct": "",
                    "mean_len_incorrect": "",
                    "median_len_correct": "",
                    "median_len_incorrect": "",
                    "delta_mean_correct_minus_incorrect": "",
                    "cohen_d": "",
                    "common_language_effect": "",
                    "permutation_p_two_sided": "",
                    "direction": "insufficient_data",
                }
            )
            continue

        mean_correct = mean([float(x) for x in correct_lengths])
        mean_incorrect = mean([float(x) for x in incorrect_lengths])
        delta = mean_correct - mean_incorrect
        d_value = cohen_d(correct_lengths, incorrect_lengths)
        cle = common_language_effect(correct_lengths, incorrect_lengths)
        _, p_value = permutation_p_value_delta(lengths, labels, permutations, rng)

        direction = "correct_longer" if delta > 0 else "incorrect_longer" if delta < 0 else "equal"

        row = {
            "question_id": question_id,
            "n_total": n_total,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "mean_len_correct": mean_correct,
            "mean_len_incorrect": mean_incorrect,
            "median_len_correct": median(correct_lengths),
            "median_len_incorrect": median(incorrect_lengths),
            "delta_mean_correct_minus_incorrect": delta,
            "cohen_d": d_value,
            "common_language_effect": cle,
            "permutation_p_two_sided": p_value,
            "direction": direction,
        }
        per_question_rows.append(row)

        valid_groups.append(
            {
                "question_id": question_id,
                "lengths": lengths,
                "labels": labels,
                "n_total": n_total,
                "delta": delta,
                "p_value": p_value,
                "direction": direction,
            }
        )

    per_question_rows.sort(key=lambda r: (r["question_id"]))
    return per_question_rows, valid_groups


def binomial_two_sided_p_value(k: int, n: int, p: float = 0.5) -> float:
    if n <= 0:
        return 1.0

    def pmf(x: int) -> float:
        return math.comb(n, x) * (p**x) * ((1 - p) ** (n - x))

    observed = pmf(k)
    total = 0.0
    for x in range(n + 1):
        px = pmf(x)
        if px <= observed + 1e-15:
            total += px
    return min(1.0, total)


def aggregate_analysis(
    valid_groups: list[dict[str, object]],
    aggregate_permutations: int,
    rng: random.Random,
) -> dict[str, float | int]:
    if not valid_groups:
        return {
            "n_valid_questions": 0,
            "weighted_mean_delta": 0.0,
            "unweighted_mean_delta": 0.0,
            "median_delta": 0.0,
            "questions_correct_longer": 0,
            "questions_incorrect_longer": 0,
            "questions_equal": 0,
            "fraction_correct_longer": 0.0,
            "sign_test_two_sided_p": 1.0,
            "aggregate_permutation_two_sided_p": 1.0,
        }

    deltas = [float(group["delta"]) for group in valid_groups]
    weights = [int(group["n_total"]) for group in valid_groups]

    weight_sum = sum(weights)
    weighted_mean_delta = sum(w * d for w, d in zip(weights, deltas)) / weight_sum
    unweighted_mean_delta = mean(deltas)
    median_delta = median(deltas)

    correct_longer = sum(1 for d in deltas if d > 0)
    incorrect_longer = sum(1 for d in deltas if d < 0)
    equal = sum(1 for d in deltas if d == 0)

    n_nonzero = correct_longer + incorrect_longer
    fraction_correct_longer = correct_longer / n_nonzero if n_nonzero else 0.0
    sign_test_p = binomial_two_sided_p_value(correct_longer, n_nonzero, 0.5) if n_nonzero else 1.0

    # Stratified permutation test: shuffle labels within each question, then aggregate.
    abs_observed = abs(weighted_mean_delta)
    extreme = 1

    if aggregate_permutations > 0:
        for _ in range(aggregate_permutations):
            perm_weighted_num = 0.0
            for group in valid_groups:
                lengths = list(group["lengths"])  # type: ignore[arg-type]
                labels = list(group["labels"])  # type: ignore[arg-type]
                n_correct = sum(1 for x in labels if x)
                indices = list(range(len(labels)))
                correct_idx = set(rng.sample(indices, n_correct))
                shuffled = [idx in correct_idx for idx in indices]
                delta = delta_mean_for_labels(lengths, shuffled)
                perm_weighted_num += int(group["n_total"]) * delta

            perm_weighted_mean = perm_weighted_num / weight_sum
            if abs(perm_weighted_mean) >= abs_observed:
                extreme += 1

        agg_perm_p = extreme / (aggregate_permutations + 1)
    else:
        agg_perm_p = 1.0

    return {
        "n_valid_questions": len(valid_groups),
        "weighted_mean_delta": weighted_mean_delta,
        "unweighted_mean_delta": unweighted_mean_delta,
        "median_delta": median_delta,
        "questions_correct_longer": correct_longer,
        "questions_incorrect_longer": incorrect_longer,
        "questions_equal": equal,
        "fraction_correct_longer": fraction_correct_longer,
        "sign_test_two_sided_p": sign_test_p,
        "aggregate_permutation_two_sided_p": agg_perm_p,
    }


def write_per_question_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "n_total",
        "n_correct",
        "n_incorrect",
        "mean_len_correct",
        "mean_len_incorrect",
        "median_len_correct",
        "median_len_incorrect",
        "delta_mean_correct_minus_incorrect",
        "cohen_d",
        "common_language_effect",
        "permutation_p_two_sided",
        "direction",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_csv(path: Path, aggregate: dict[str, float | int], meta: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in meta.items():
            writer.writerow([k, v])
        for k, v in aggregate.items():
            writer.writerow([k, v])


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def write_report(
    path: Path,
    input_path: Path,
    meta: dict[str, float | int | str],
    aggregate: dict[str, float | int],
    per_question_rows: list[dict[str, float | int | str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    valid_rows = [r for r in per_question_rows if r["direction"] != "insufficient_data"]
    strongest_positive = sorted(
        [r for r in valid_rows if float(r["delta_mean_correct_minus_incorrect"]) > 0],
        key=lambda r: float(r["delta_mean_correct_minus_incorrect"]),
        reverse=True,
    )[:10]
    strongest_negative = sorted(
        [r for r in valid_rows if float(r["delta_mean_correct_minus_incorrect"]) < 0],
        key=lambda r: float(r["delta_mean_correct_minus_incorrect"]),
    )[:10]

    lines = [
        "# Trace Length vs Quality Analysis",
        "",
        f"- Input: `{input_path}`",
        f"- Parsed rows: {meta['parsed_rows']}",
        f"- Skipped rows: {meta['skipped_rows']}",
        f"- Unique questions seen: {meta['unique_questions_seen']}",
        f"- Questions with analyzable class balance: {aggregate['n_valid_questions']}",
        "",
        "## Aggregate Conclusion",
        "",
        f"- Weighted mean length delta (correct - incorrect): {float(aggregate['weighted_mean_delta']):.4f} tokens",
        f"- Unweighted mean length delta: {float(aggregate['unweighted_mean_delta']):.4f} tokens",
        f"- Median per-question length delta: {float(aggregate['median_delta']):.4f} tokens",
        f"- Questions where correct traces are longer: {aggregate['questions_correct_longer']}",
        f"- Questions where incorrect traces are longer: {aggregate['questions_incorrect_longer']}",
        f"- Questions tied: {aggregate['questions_equal']}",
        f"- Fraction (non-tied) with correct longer: {fmt_pct(float(aggregate['fraction_correct_longer']))}",
        f"- Sign test (two-sided) p-value: {float(aggregate['sign_test_two_sided_p']):.6f}",
        (
            "- Stratified permutation test p-value (within-question label shuffle, weighted mean delta): "
            f"{float(aggregate['aggregate_permutation_two_sided_p']):.6f}"
        ),
        "",
        "## Questions With Largest Positive Delta (Correct Longer)",
        "",
        "| question_id | delta_mean (tokens) | n_correct | n_incorrect | p-value |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in strongest_positive:
        lines.append(
            f"| {row['question_id']} | {float(row['delta_mean_correct_minus_incorrect']):.4f} "
            f"| {row['n_correct']} | {row['n_incorrect']} | {float(row['permutation_p_two_sided']):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Questions With Largest Negative Delta (Incorrect Longer)",
            "",
            "| question_id | delta_mean (tokens) | n_correct | n_incorrect | p-value |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for row in strongest_negative:
        lines.append(
            f"| {row['question_id']} | {float(row['delta_mean_correct_minus_incorrect']):.4f} "
            f"| {row['n_correct']} | {row['n_incorrect']} | {float(row['permutation_p_two_sided']):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Each question is analyzed independently to control for difficulty differences.",
            "- Primary effect is `mean_length(correct) - mean_length(incorrect)` per question.",
            "- Aggregate permutation p-value tests whether the overall length effect is stronger than expected by chance when correctness labels are shuffled only within each question.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.min_class_size <= 0:
        raise ValueError("--min-class-size must be positive.")
    if args.per_question_permutations < 0 or args.aggregate_permutations < 0:
        raise ValueError("Permutation counts must be non-negative.")

    rng = random.Random(args.seed)

    grouped, total_rows, skipped_rows = load_grouped_data(
        input_path=args.input,
        question_col=args.question_column,
        label_col=args.label_column,
        trace_col=args.trace_column,
        include_missing_as_zero=args.include_missing_as_zero,
    )

    per_question_rows, valid_groups = analyze_per_question(
        grouped=grouped,
        min_class_size=args.min_class_size,
        permutations=args.per_question_permutations,
        rng=rng,
    )

    aggregate = aggregate_analysis(
        valid_groups=valid_groups,
        aggregate_permutations=args.aggregate_permutations,
        rng=rng,
    )

    meta = {
        "input_path": str(args.input),
        "parsed_rows": total_rows - skipped_rows,
        "skipped_rows": skipped_rows,
        "unique_questions_seen": len(grouped),
        "min_class_size": args.min_class_size,
        "per_question_permutations": args.per_question_permutations,
        "aggregate_permutations": args.aggregate_permutations,
        "seed": args.seed,
        "include_missing_as_zero": args.include_missing_as_zero,
    }

    write_per_question_csv(args.per_question_csv, per_question_rows)
    write_aggregate_csv(args.aggregate_csv, aggregate, meta)
    write_report(args.report_md, args.input, meta, aggregate, per_question_rows)

    print(f"Wrote per-question stats CSV: {args.per_question_csv}")
    print(f"Wrote aggregate stats CSV: {args.aggregate_csv}")
    print(f"Wrote markdown report: {args.report_md}")


if __name__ == "__main__":
    main()
