from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .stats import benjamini_hochberg, fisher_exact_pvalue

EPSILON = 1e-9


def support(count: int, total_docs: int) -> float:
    """Document-level support."""
    if total_docs <= 0:
        return 0.0
    return count / total_docs


def smoothed_log_odds_ratio(
    success_count: int,
    failure_count: int,
    success_total: int,
    failure_total: int,
    alpha: float = 0.5,
) -> float:
    """Smoothed log-odds ratio with Haldane-Anscombe style smoothing."""
    a = success_count + alpha
    b = (success_total - success_count) + alpha
    c = failure_count + alpha
    d = (failure_total - failure_count) + alpha
    return float(np.log((a / b) / (c / d)))


def score_item(
    item: tuple[str, ...],
    success_count: int,
    failure_count: int,
    success_total: int,
    failure_total: int,
    *,
    alpha: float = 0.5,
    compute_p_value: bool = True,
) -> dict[str, Any]:
    """Compute shared discriminative metrics for one motif/rule."""
    success_sup = support(success_count, success_total)
    failure_sup = support(failure_count, failure_total)
    diff = success_sup - failure_sup
    lift = success_sup / (failure_sup + EPSILON)
    lor = smoothed_log_odds_ratio(success_count, failure_count, success_total, failure_total, alpha)

    p_value = None
    if compute_p_value:
        p_value = fisher_exact_pvalue(
            success_count,
            success_total - success_count,
            failure_count,
            failure_total - failure_count,
        )

    return {
        "motif": "|".join(item),
        "length": len(item),
        "success_count": int(success_count),
        "failure_count": int(failure_count),
        "success_support": success_sup,
        "failure_support": failure_sup,
        "support_difference": diff,
        "lift": lift,
        "log_odds_ratio": lor,
        "p_value": p_value,
    }


def finalize_scores(df: pd.DataFrame, *, q_value: bool = True) -> pd.DataFrame:
    """Add q-values and helper ranking columns."""
    out = df.copy()

    if q_value and "p_value" in out.columns and out["p_value"].notna().any():
        pvals = out["p_value"].fillna(1.0).astype(float).to_numpy()
        out["q_value"] = benjamini_hochberg(pvals)
    else:
        out["q_value"] = np.nan

    out["abs_support_difference"] = out["support_difference"].abs()
    out["abs_log_odds_ratio"] = out["log_odds_ratio"].abs()
    return out


def rank_success_enriched(df: pd.DataFrame, top_k: int = 100) -> pd.DataFrame:
    """Top motifs enriched in successful traces."""
    ranked = df.sort_values(
        by=["support_difference", "success_support", "abs_log_odds_ratio"],
        ascending=[False, False, False],
    )
    return ranked.head(top_k).reset_index(drop=True)


def rank_failure_enriched(df: pd.DataFrame, top_k: int = 100) -> pd.DataFrame:
    """Top motifs enriched in unsuccessful traces."""
    ranked = df.sort_values(
        by=["support_difference", "failure_support", "abs_log_odds_ratio"],
        ascending=[True, False, False],
    )
    return ranked.head(top_k).reset_index(drop=True)
