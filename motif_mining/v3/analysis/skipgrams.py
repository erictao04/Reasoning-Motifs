from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Iterable

import pandas as pd

from .scoring import finalize_scores, rank_failure_enriched, rank_success_enriched, score_item


def _generate_skipgrams_exact_length(
    tokens: list[str],
    n: int,
    max_gap: int,
) -> set[tuple[str, ...]]:
    """
    Generate bounded-gap skip-grams of exact length n.

    Two neighboring selected positions i<j must satisfy (j - i - 1) <= max_gap.
    """
    if n <= 0 or len(tokens) < n:
        return set()

    results: set[tuple[str, ...]] = set()

    def backtrack(start_idx: int, last_idx: int, current: list[str]) -> None:
        if len(current) == n:
            results.add(tuple(current))
            return

        remaining = n - len(current)
        max_start = len(tokens) - remaining
        for idx in range(start_idx, max_start + 1):
            if last_idx >= 0 and (idx - last_idx - 1) > max_gap:
                if idx > last_idx:
                    break
            current.append(tokens[idx])
            backtrack(idx + 1, idx, current)
            current.pop()

    backtrack(0, -1, [])
    return results


@lru_cache(maxsize=2048)
def _cached_skipgrams(tokens_key: tuple[str, ...], min_len: int, max_len: int, max_gap: int) -> tuple[tuple[str, ...], ...]:
    tokens = list(tokens_key)
    motifs: set[tuple[str, ...]] = set()
    for n in range(min_len, max_len + 1):
        motifs.update(_generate_skipgrams_exact_length(tokens, n, max_gap))
    return tuple(sorted(motifs))


def skipgrams_for_trace(tokens: list[str], min_len: int, max_len: int, max_gap: int) -> set[tuple[str, ...]]:
    """All bounded-gap skip-grams in one trace (document-level unique motifs)."""
    if not tokens:
        return set()
    return set(_cached_skipgrams(tuple(tokens), min_len, max_len, max_gap))


def _count_doc_support(
    traces: Iterable[list[str]],
    min_len: int,
    max_len: int,
    max_gap: int,
) -> Counter[tuple[str, ...]]:
    counter: Counter[tuple[str, ...]] = Counter()
    for tokens in traces:
        motifs = skipgrams_for_trace(tokens, min_len, max_len, max_gap)
        counter.update(motifs)
    return counter


def mine_skipgrams(
    success_traces: list[list[str]],
    failure_traces: list[list[str]],
    *,
    min_len: int = 2,
    max_len: int = 4,
    max_gap: int = 2,
    min_support_count: int = 1,
) -> pd.DataFrame:
    """Mine discriminative bounded-gap skip-grams."""
    success_counts = _count_doc_support(success_traces, min_len, max_len, max_gap)
    failure_counts = _count_doc_support(failure_traces, min_len, max_len, max_gap)

    motifs = set(success_counts) | set(failure_counts)
    rows = []

    for motif in motifs:
        s_count = success_counts.get(motif, 0)
        f_count = failure_counts.get(motif, 0)
        if (s_count + f_count) < min_support_count:
            continue
        rows.append(
            score_item(
                motif,
                s_count,
                f_count,
                len(success_traces),
                len(failure_traces),
            )
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return finalize_scores(df)


def top_skipgram_tables(df: pd.DataFrame, top_k: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return top success-enriched and failure-enriched motif tables."""
    if df.empty:
        return df.copy(), df.copy()
    return rank_success_enriched(df, top_k=top_k), rank_failure_enriched(df, top_k=top_k)
