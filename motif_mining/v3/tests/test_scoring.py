from __future__ import annotations

import pandas as pd

from analysis.scoring import finalize_scores, smoothed_log_odds_ratio, support


def test_support_basic() -> None:
    assert support(5, 10) == 0.5
    assert support(0, 0) == 0.0


def test_log_odds_sanity() -> None:
    pos = smoothed_log_odds_ratio(10, 2, 20, 20)
    neg = smoothed_log_odds_ratio(2, 10, 20, 20)
    assert pos > 0
    assert neg < 0


def test_finalize_scores_adds_qvalue() -> None:
    df = pd.DataFrame(
        [
            {"motif": "a", "support_difference": 0.2, "success_support": 0.3, "failure_support": 0.1, "log_odds_ratio": 1.2, "p_value": 0.01},
            {"motif": "b", "support_difference": -0.1, "success_support": 0.1, "failure_support": 0.2, "log_odds_ratio": -0.7, "p_value": 0.2},
        ]
    )
    out = finalize_scores(df)
    assert "q_value" in out.columns
