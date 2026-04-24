from __future__ import annotations

from analysis.skipgrams import skipgrams_for_trace


def test_skipgrams_bounded_gap() -> None:
    tokens = ["a", "b", "c", "d"]
    motifs = skipgrams_for_trace(tokens, min_len=2, max_len=2, max_gap=1)

    assert ("a", "b") in motifs
    assert ("a", "c") in motifs  # one skipped token allowed
    assert ("a", "d") not in motifs  # gap too large


def test_skipgrams_document_level_uniqueness() -> None:
    tokens = ["a", "b", "a", "b"]
    motifs = skipgrams_for_trace(tokens, min_len=2, max_len=2, max_gap=0)
    assert ("a", "b") in motifs
    assert len([m for m in motifs if m == ("a", "b")]) == 1
