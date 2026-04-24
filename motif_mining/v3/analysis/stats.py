from __future__ import annotations

import numpy as np


try:
    from scipy.stats import fisher_exact
except Exception:  # pragma: no cover - optional dependency behavior
    fisher_exact = None


def fisher_exact_pvalue(a: int, b: int, c: int, d: int) -> float | None:
    """Two-sided Fisher exact p-value if scipy is available."""
    if fisher_exact is None:
        return None
    _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return float(p)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p

    order = np.argsort(p)
    ranked = p[order]

    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = (ranked[i] * n) / rank
        prev = min(prev, value)
        q[i] = prev

    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out
