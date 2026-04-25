#!/usr/bin/env python3
"""Stage 5 / Experiment 4: cross-tokenizer robustness analysis."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 5 cross-tokenizer robustness analysis from Stage 2 outputs."
    )
    parser.add_argument(
        "--stage2-root",
        type=Path,
        default=Path("research/stage2_exp1_single_token_delta"),
        help="Root directory containing Stage 2 results_* folders.",
    )
    parser.add_argument(
        "--result-dirs",
        nargs="*",
        type=Path,
        default=[],
        help="Explicit Stage 2 result directories (overrides auto-discovery if provided).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("experiments/exp4_cross_tokenizer"),
        help="Output directory for Stage 5 artifacts.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Top-K per tokenizer to compute stable diagnostic set.",
    )
    parser.add_argument(
        "--min-tokenizers",
        type=int,
        default=3,
        help="Minimum tokenizer runs required.",
    )
    return parser.parse_args()


def canonical_word(word: str) -> str:
    synonyms = {
        "modular": "mod",
        "modulo": "mod",
        "equation": "eq",
        "equations": "eq",
        "equalities": "eq",
        "equality": "eq",
        "equals": "eq",
        "backtracking": "backtrack",
        "backtracked": "backtrack",
        "backtracks": "backtrack",
        "factorization": "factor",
        "factoring": "factor",
        "simplify": "simplification",
        "simplified": "simplification",
        "inequality": "ineq",
        "inequalities": "ineq",
        "parities": "parity",
    }
    return synonyms.get(word, word)


def canonicalize_token(token: str) -> str:
    token = (token or "").strip().lower()
    if not token:
        return ""
    if ":" not in token:
        return token
    basetype, qualifier = token.split(":", 1)
    basetype = basetype.strip()
    words = [w for w in re.split(r"[^a-z0-9]+", qualifier) if w]
    words = [canonical_word(w) for w in words]
    if not words:
        return basetype
    return f"{basetype}:{'-'.join(words)}"


def discover_result_dirs(stage2_root: Path) -> list[Path]:
    if not stage2_root.exists():
        return []
    result_dirs = []
    for path in stage2_root.iterdir():
        if not path.is_dir():
            continue
        if not path.name.startswith("results_"):
            continue
        if (path / "token_deltas_all.csv").exists():
            result_dirs.append(path)
    return sorted(result_dirs)


def model_name_from_dir(path: Path) -> str:
    name = path.name
    if name.startswith("results_"):
        return name[len("results_") :]
    return name


def read_token_deltas(path: Path) -> dict[str, float]:
    token_scores: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            token = (row.get("token") or "").strip()
            if not token:
                continue
            try:
                mean_delta = float(row.get("mean_delta", ""))
            except ValueError:
                continue
            token_scores[token] = mean_delta
    return token_scores


def aggregate_scores(
    raw_scores: dict[str, float], key_fn
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for token, score in raw_scores.items():
        key = key_fn(token)
        if not key:
            continue
        grouped[key].append(score)
    return {k: (sum(v) / len(v)) for k, v in grouped.items()}


def average_ranks_desc(scores: dict[str, float]) -> dict[str, float]:
    sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ranks: dict[str, float] = {}
    i = 0
    while i < len(sorted_items):
        j = i + 1
        while j < len(sorted_items) and sorted_items[j][1] == sorted_items[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[sorted_items[k][0]] = avg_rank
        i = j
    return ranks


def spearman_rho(scores_a: dict[str, float], scores_b: dict[str, float]) -> tuple[float, int]:
    common = sorted(set(scores_a) & set(scores_b))
    n = len(common)
    if n < 3:
        return float("nan"), n
    rank_a = average_ranks_desc({k: scores_a[k] for k in common})
    rank_b = average_ranks_desc({k: scores_b[k] for k in common})
    xa = [rank_a[k] for k in common]
    xb = [rank_b[k] for k in common]
    mean_a = sum(xa) / n
    mean_b = sum(xb) / n
    num = sum((a - mean_a) * (b - mean_b) for a, b in zip(xa, xb))
    den_a = math.sqrt(sum((a - mean_a) ** 2 for a in xa))
    den_b = math.sqrt(sum((b - mean_b) ** 2 for b in xb))
    if den_a == 0 or den_b == 0:
        return float("nan"), n
    return num / (den_a * den_b), n


def top_k_by_abs(scores: dict[str, float], k: int) -> dict[str, float]:
    ranked = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
    return dict(ranked)


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    result_dirs = args.result_dirs or discover_result_dirs(args.stage2_root)
    result_dirs = [p for p in result_dirs if (p / "token_deltas_all.csv").exists()]
    if len(result_dirs) < args.min_tokenizers:
        raise SystemExit(
            f"Need at least {args.min_tokenizers} Stage 2 result dirs with token_deltas_all.csv; found {len(result_dirs)}."
        )

    model_scores_raw: dict[str, dict[str, float]] = {}
    skipped_empty: list[str] = []
    for result_dir in result_dirs:
        model = model_name_from_dir(result_dir)
        raw = read_token_deltas(result_dir / "token_deltas_all.csv")
        if not raw:
            skipped_empty.append(model)
            continue
        model_scores_raw[model] = raw

    if len(model_scores_raw) < args.min_tokenizers:
        raise SystemExit(
            f"Need at least {args.min_tokenizers} non-empty tokenizer runs; found {len(model_scores_raw)}."
        )

    model_scores_basetype: dict[str, dict[str, float]] = {}
    model_scores_full: dict[str, dict[str, float]] = {}
    for model, raw in model_scores_raw.items():
        model_scores_basetype[model] = aggregate_scores(raw, lambda t: t.split(":", 1)[0].strip().lower())
        model_scores_full[model] = aggregate_scores(raw, canonicalize_token)

    models = sorted(model_scores_raw.keys())
    pairwise_rows: list[dict[str, object]] = []
    for i, model_a in enumerate(models):
        for model_b in models[i + 1 :]:
            rho_b, n_b = spearman_rho(model_scores_basetype[model_a], model_scores_basetype[model_b])
            rho_f, n_f = spearman_rho(model_scores_full[model_a], model_scores_full[model_b])
            pairwise_rows.append(
                {
                    "level": "basetype",
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_common_items": n_b,
                    "spearman_rho": rho_b,
                }
            )
            pairwise_rows.append(
                {
                    "level": "full_token",
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_common_items": n_f,
                    "spearman_rho": rho_f,
                }
            )

    stable_rows: list[dict[str, object]] = []
    for level, model_scores in (("basetype", model_scores_basetype), ("full_token", model_scores_full)):
        top_per_model = {m: top_k_by_abs(scores, args.top_k) for m, scores in model_scores.items()}
        votes: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for model, top_scores in top_per_model.items():
            for token, score in top_scores.items():
                votes[token].append((model, score))
        stable_threshold = math.ceil(len(models) / 2.0)
        for token, model_and_score in votes.items():
            present_models = sorted(m for m, _ in model_and_score)
            score_values = [s for _, s in model_and_score]
            vote_count = len(present_models)
            mean_score = sum(score_values) / vote_count
            std_score = statistics.pstdev(score_values) if vote_count > 1 else 0.0
            stable_rows.append(
                {
                    "level": level,
                    "token": token,
                    "vote_count": vote_count,
                    "n_tokenizers": len(models),
                    "vote_fraction": vote_count / len(models),
                    "stable_threshold": stable_threshold,
                    "is_stable": vote_count >= stable_threshold,
                    "models": ";".join(present_models),
                    "mean_delta_when_topk": mean_score,
                    "std_delta_when_topk": std_score,
                }
            )

    pairwise_rows = sorted(pairwise_rows, key=lambda r: (str(r["level"]), str(r["model_a"]), str(r["model_b"])))
    stable_rows = sorted(
        stable_rows,
        key=lambda r: (str(r["level"]), -int(r["vote_count"]), -abs(float(r["mean_delta_when_topk"])), str(r["token"])),
    )

    outdir = args.outdir
    write_csv(
        outdir / "pairwise_spearman.csv",
        pairwise_rows,
        ["level", "model_a", "model_b", "n_common_items", "spearman_rho"],
    )
    write_csv(
        outdir / "stable_diagnostic_set.csv",
        stable_rows,
        [
            "level",
            "token",
            "vote_count",
            "n_tokenizers",
            "vote_fraction",
            "stable_threshold",
            "is_stable",
            "models",
            "mean_delta_when_topk",
            "std_delta_when_topk",
        ],
    )

    mean_pairwise = [
        float(r["spearman_rho"])
        for r in pairwise_rows
        if isinstance(r["spearman_rho"], float) and not math.isnan(float(r["spearman_rho"]))
    ]
    print(f"Tokenizer models analyzed: {len(models)}")
    if skipped_empty:
        print(f"Skipped empty tokenizer runs: {', '.join(sorted(skipped_empty))}")
    print(f"Models: {', '.join(models)}")
    print(f"Mean pairwise Spearman rho (all rows): {(sum(mean_pairwise) / len(mean_pairwise)) if mean_pairwise else float('nan'):.4f}")
    print(f"Wrote Stage 5 outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
