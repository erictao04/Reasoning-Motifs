from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import pandas as pd

from .scoring import finalize_scores, rank_failure_enriched, rank_success_enriched, score_item
from .sequential_patterns import Pattern, is_subsequence, mine_frequent_patterns


@dataclass(frozen=True)
class Rule:
    antecedent: Pattern
    consequent: Pattern

    def label(self) -> str:
        return f"{'|'.join(self.antecedent)} => {'|'.join(self.consequent)}"


def _pattern_end_positions(sequence: list[str], pattern: Pattern) -> list[int]:
    """End indices for all occurrences of a pattern as a subsequence."""
    ends: list[int] = []

    def backtrack(seq_idx: int, pat_idx: int, last_idx: int) -> None:
        if pat_idx == len(pattern):
            ends.append(last_idx)
            return
        token = pattern[pat_idx]
        for i in range(seq_idx, len(sequence)):
            if sequence[i] == token:
                backtrack(i + 1, pat_idx + 1, i)

    backtrack(0, 0, -1)
    return ends


def _rule_holds(sequence: list[str], antecedent: Pattern, consequent: Pattern) -> bool:
    """True if antecedent occurs before consequent in sequence (gaps allowed)."""
    ends = _pattern_end_positions(sequence, antecedent)
    if not ends:
        return False

    for end_idx in ends:
        if is_subsequence(consequent, sequence[end_idx + 1 :]):
            return True
    return False


def _rule_metrics_for_group(traces: list[list[str]], rule: Rule) -> tuple[int, int, int, int]:
    """Return (rule_count, antecedent_count, consequent_count, total_docs)."""
    rule_count = 0
    ant_count = 0
    cons_count = 0

    for tokens in traces:
        ant_present = is_subsequence(rule.antecedent, tokens)
        cons_present = is_subsequence(rule.consequent, tokens)
        if ant_present:
            ant_count += 1
        if cons_present:
            cons_count += 1
        if ant_present and cons_present and _rule_holds(tokens, rule.antecedent, rule.consequent):
            rule_count += 1

    return rule_count, ant_count, cons_count, len(traces)


def mine_sequential_rules(
    success_traces: list[list[str]],
    failure_traces: list[list[str]],
    *,
    min_support_count: int = 5,
    max_len: int = 4,
    backend: str = "auto",
    max_candidates: int = 80,
) -> pd.DataFrame:
    """
    Mine sequential rules antecedent => consequent.

    Strategy:
    1) Mine frequent subsequences from all traces.
    2) Keep top candidate patterns by global support.
    3) Form rule candidates and score discriminativeness by rule support in success vs failure.
    """
    all_traces = success_traces + failure_traces
    frequent = mine_frequent_patterns(
        all_traces,
        backend=backend,  # type: ignore[arg-type]
        min_support_count=min_support_count,
        min_len=1,
        max_len=max_len,
    )
    if not frequent:
        return pd.DataFrame()

    patterns_sorted = sorted(frequent.items(), key=lambda kv: kv[1], reverse=True)
    patterns = [pat for pat, _ in patterns_sorted[:max_candidates]]

    rules: list[Rule] = []
    for ant, cons in product(patterns, patterns):
        if ant == cons:
            continue
        if len(ant) + len(cons) > max_len:
            continue
        rules.append(Rule(ant, cons))

    rows = []
    for rule in rules:
        s_rule, s_ant, s_cons, s_total = _rule_metrics_for_group(success_traces, rule)
        f_rule, f_ant, f_cons, f_total = _rule_metrics_for_group(failure_traces, rule)

        if (s_rule + f_rule) < min_support_count:
            continue

        scored = score_item(rule.antecedent + rule.consequent, s_rule, f_rule, s_total, f_total)
        scored["motif"] = rule.label()
        scored["antecedent"] = "|".join(rule.antecedent)
        scored["consequent"] = "|".join(rule.consequent)
        scored["antecedent_len"] = len(rule.antecedent)
        scored["consequent_len"] = len(rule.consequent)

        s_conf = s_rule / s_ant if s_ant else 0.0
        f_conf = f_rule / f_ant if f_ant else 0.0
        s_conseq_sup = s_cons / s_total if s_total else 0.0
        f_conseq_sup = f_cons / f_total if f_total else 0.0

        scored["success_confidence"] = s_conf
        scored["failure_confidence"] = f_conf
        scored["success_rule_lift"] = s_conf / (s_conseq_sup + 1e-9)
        scored["failure_rule_lift"] = f_conf / (f_conseq_sup + 1e-9)
        scored["confidence_difference"] = s_conf - f_conf

        rows.append(scored)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return finalize_scores(df)


def top_rule_tables(df: pd.DataFrame, top_k: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top success/failure enriched rule tables."""
    if df.empty:
        return df.copy(), df.copy()
    return rank_success_enriched(df, top_k=top_k), rank_failure_enriched(df, top_k=top_k)
