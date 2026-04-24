from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable, Literal

import pandas as pd

from .scoring import finalize_scores, rank_failure_enriched, rank_success_enriched, score_item

Pattern = tuple[str, ...]
Backend = Literal["auto", "python", "prefixspan"]


def is_subsequence(pattern: Pattern, sequence: list[str]) -> bool:
    """Return True if pattern appears in sequence with gaps allowed."""
    it = iter(sequence)
    return all(token in it for token in pattern)


def enumerate_subsequences(
    tokens: list[str],
    min_len: int,
    max_len: int,
) -> set[Pattern]:
    """Enumerate unique subsequences (with arbitrary gaps) up to max_len."""
    n = len(tokens)
    out: set[Pattern] = set()
    if n == 0:
        return out

    upper = min(max_len, n)
    for k in range(max(1, min_len), upper + 1):
        for idxs in combinations(range(n), k):
            out.add(tuple(tokens[i] for i in idxs))
    return out


def mine_frequent_python(
    traces: Iterable[list[str]],
    *,
    min_support_count: int,
    min_len: int,
    max_len: int,
) -> Counter[Pattern]:
    """Pure Python frequent subsequence miner for small/medium settings.

    Complexity is O(N * sum_{k=min_len..max_len} C(L, k)) where N is number of traces and L is average trace length.
    """
    counts: Counter[Pattern] = Counter()
    for tokens in traces:
        motifs = enumerate_subsequences(tokens, min_len=min_len, max_len=max_len)
        counts.update(motifs)

    return Counter({k: v for k, v in counts.items() if v >= min_support_count})


def mine_frequent_prefixspan(
    traces: list[list[str]],
    *,
    min_support_count: int,
    min_len: int,
    max_len: int,
) -> Counter[Pattern]:
    """Optional PrefixSpan adapter (if package is installed)."""
    try:
        from prefixspan import PrefixSpan
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("prefixspan backend requested but package is unavailable") from exc

    token_to_id: dict[str, int] = {}
    id_to_token: dict[int, str] = {}

    encoded: list[list[int]] = []
    for seq in traces:
        row: list[int] = []
        for token in seq:
            if token not in token_to_id:
                next_id = len(token_to_id)
                token_to_id[token] = next_id
                id_to_token[next_id] = token
            row.append(token_to_id[token])
        encoded.append(row)

    ps = PrefixSpan(encoded)
    results = ps.frequent(min_support_count)

    out: Counter[Pattern] = Counter()
    for support, pattern in results:
        if len(pattern) < min_len or len(pattern) > max_len:
            continue
        out[tuple(id_to_token[idx] for idx in pattern)] = int(support)
    return out


def mine_frequent_patterns(
    traces: list[list[str]],
    *,
    backend: Backend = "auto",
    min_support_count: int = 5,
    min_len: int = 1,
    max_len: int = 4,
) -> Counter[Pattern]:
    """Mine frequent subsequences with backend fallback."""
    if backend == "python":
        return mine_frequent_python(
            traces,
            min_support_count=min_support_count,
            min_len=min_len,
            max_len=max_len,
        )

    if backend == "prefixspan":
        return mine_frequent_prefixspan(
            traces,
            min_support_count=min_support_count,
            min_len=min_len,
            max_len=max_len,
        )

    # auto
    try:
        return mine_frequent_prefixspan(
            traces,
            min_support_count=min_support_count,
            min_len=min_len,
            max_len=max_len,
        )
    except Exception:
        return mine_frequent_python(
            traces,
            min_support_count=min_support_count,
            min_len=min_len,
            max_len=max_len,
        )


def discriminative_pattern_table(
    success_traces: list[list[str]],
    failure_traces: list[list[str]],
    *,
    backend: Backend = "auto",
    min_support_count: int = 5,
    min_len: int = 1,
    max_len: int = 4,
) -> pd.DataFrame:
    """Mine sequential patterns and score success vs failure enrichment."""
    success_freq = mine_frequent_patterns(
        success_traces,
        backend=backend,
        min_support_count=min_support_count,
        min_len=min_len,
        max_len=max_len,
    )
    failure_freq = mine_frequent_patterns(
        failure_traces,
        backend=backend,
        min_support_count=min_support_count,
        min_len=min_len,
        max_len=max_len,
    )

    candidates = set(success_freq) | set(failure_freq)
    rows = []
    for pattern in candidates:
        s_count = success_freq.get(pattern, 0)
        f_count = failure_freq.get(pattern, 0)
        rows.append(
            score_item(
                pattern,
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


def _is_subsequence_tuple(shorter: tuple[str, ...], longer: tuple[str, ...]) -> bool:
    it = iter(longer)
    return all(tok in it for tok in shorter)


def reduce_redundant_patterns(
    df: pd.DataFrame,
    *,
    support_tolerance: float = 0.005,
    score_column: str = "abs_log_odds_ratio",
) -> pd.DataFrame:
    """
    Pragmatic redundancy reduction:

    Drop pattern P if there exists a longer supersequence Q such that:
    - P is a subsequence of Q
    - |support_success(P)-support_success(Q)| <= tolerance
    - |support_failure(P)-support_failure(Q)| <= tolerance
    - score(Q) >= score(P)
    """
    if df.empty:
        return df.copy()

    work = df.copy()
    work["pattern_tuple"] = work["motif"].map(lambda s: tuple(str(s).split("|")))
    work = work.sort_values(by=["length", score_column], ascending=[False, False]).reset_index(drop=True)

    keep = [True] * len(work)
    for i, row_i in work.iterrows():
        if not keep[i]:
            continue
        p = row_i["pattern_tuple"]
        s_sup = float(row_i["success_support"])
        f_sup = float(row_i["failure_support"])
        score = float(row_i[score_column])

        for j, row_j in work.iloc[:i].iterrows():
            if not keep[j]:
                continue
            q = row_j["pattern_tuple"]
            if len(q) <= len(p):
                continue
            if not _is_subsequence_tuple(p, q):
                continue

            s_close = abs(s_sup - float(row_j["success_support"])) <= support_tolerance
            f_close = abs(f_sup - float(row_j["failure_support"])) <= support_tolerance
            better_score = float(row_j[score_column]) >= score
            if s_close and f_close and better_score:
                keep[i] = False
                break

    reduced = work[pd.Series(keep)].drop(columns=["pattern_tuple"]).reset_index(drop=True)
    return reduced


def top_pattern_tables(df: pd.DataFrame, top_k: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top success/failure enriched tables."""
    if df.empty:
        return df.copy(), df.copy()
    return rank_success_enriched(df, top_k=top_k), rank_failure_enriched(df, top_k=top_k)
