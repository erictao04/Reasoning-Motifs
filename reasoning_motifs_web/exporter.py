from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .fixtures import (
    DEFAULT_GLOBAL_MOTIFS_CSV,
    DEFAULT_RAW_TRACE_CSV,
    DEFAULT_TOKENIZED_TRACE_CSV,
)
from .models import (
    AnswerDistributionRow,
    CorpusOverview,
    MotifBucket,
    MotifRow,
    QuestionDetail,
    QuestionSummary,
    StorySection,
    TraceSummary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = REPO_ROOT / "motif_mining" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from analysis.skipgrams import mine_skipgrams  # noqa: E402


@dataclass(frozen=True)
class JoinedTrace:
    question_id: str
    sample_id: str
    attempt_index: str
    question_text: str
    gold_answer: str
    predicted_answer: str
    is_correct: bool
    benchmark_name: str
    pilot_question_uid: str
    tokenized_trace: str
    tokens: tuple[str, ...]
    reasoning_trace: str

    @property
    def trace_id(self) -> str:
        return f"{self.question_id}:{self.sample_id}:{self.attempt_index}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "t"}


def _parse_tokens(value: str) -> tuple[str, ...]:
    if "|" in value:
        parts = value.split("|")
    elif " > " in value:
        parts = value.split(" > ")
    else:
        parts = value.split()
    return tuple(token.strip() for token in parts if token.strip())


def load_joined_traces(
    *,
    tokenized_csv: Path = DEFAULT_TOKENIZED_TRACE_CSV,
    raw_csv: Path = DEFAULT_RAW_TRACE_CSV,
) -> list[JoinedTrace]:
    tokenized_rows = _read_csv(tokenized_csv)
    raw_rows = _read_csv(raw_csv)

    raw_lookup = {
        (row["question_id"], row["sample_id"], row["attempt_index"]): row
        for row in raw_rows
    }

    joined: list[JoinedTrace] = []
    for row in tokenized_rows:
        attempt_index = row.get("attempt_index")
        if attempt_index is None and raw_lookup:
            matching_keys = [
                key
                for key in raw_lookup
                if key[0] == row["question_id"] and key[1] == row["sample_id"]
            ]
            if len(matching_keys) == 1:
                attempt_index = matching_keys[0][2]
        attempt_index = attempt_index or "0"

        key = (row["question_id"], row["sample_id"], attempt_index)
        raw = raw_lookup.get(key)
        if raw is None:
            continue

        tokenized_trace = row.get("reasoning_trace") or row.get("tokenized_trace") or ""
        joined.append(
            JoinedTrace(
                question_id=row["question_id"],
                sample_id=row["sample_id"],
                attempt_index=attempt_index,
                question_text=row.get("question") or raw.get("question") or "",
                gold_answer=row.get("gold_answer") or raw.get("gold_answer") or "",
                predicted_answer=row.get("predicted_answer") or raw.get("predicted_answer") or "",
                is_correct=_parse_bool(row["is_correct"]),
                benchmark_name=row.get("benchmark_name") or raw.get("benchmark_name") or "expanded_pool",
                pilot_question_uid=(
                    row.get("pilot_question_uid")
                    or raw.get("pilot_question_uid")
                    or f"{row['question_id']}:{row['sample_id']}"
                ),
                tokenized_trace=tokenized_trace,
                tokens=_parse_tokens(tokenized_trace),
                reasoning_trace=raw["reasoning_trace"],
            )
        )

    return joined


def _contains_motif(tokens: tuple[str, ...], motif_tokens: tuple[str, ...]) -> bool:
    if not motif_tokens or len(tokens) < len(motif_tokens):
        return False
    width = len(motif_tokens)
    for idx in range(0, len(tokens) - width + 1):
        if tokens[idx : idx + width] == motif_tokens:
            return True
    return False


def _mine_motif_rows(
    success_traces: list[tuple[str, ...]],
    failure_traces: list[tuple[str, ...]],
    *,
    min_len: int,
    max_len: int,
    min_support_count: int,
    scope: str,
) -> list[MotifRow]:
    if not success_traces or not failure_traces:
        return []

    frame = mine_skipgrams(
        [list(tokens) for tokens in success_traces],
        [list(tokens) for tokens in failure_traces],
        min_len=min_len,
        max_len=max_len,
        max_gap=0,
        min_support_count=min_support_count,
    )
    if frame.empty:
        return []

    rows: list[MotifRow] = []
    for entry in frame.to_dict(orient="records"):
        rows.append(
            MotifRow(
                motif=entry["motif"],
                tokens=entry["motif"].split("|"),
                length=int(entry["length"]),
                scope=scope,
                direction="success" if float(entry["support_difference"]) >= 0 else "failure",
                success_count=int(entry["success_count"]),
                failure_count=int(entry["failure_count"]),
                success_support=float(entry["success_support"]),
                failure_support=float(entry["failure_support"]),
                support_difference=float(entry["support_difference"]),
                lift=float(entry["lift"]),
                log_odds_ratio=float(entry["log_odds_ratio"]),
                q_value=(
                    None
                    if entry.get("q_value") is None or str(entry.get("q_value")) == "nan"
                    else float(entry["q_value"])
                ),
            )
        )
    return rows


def _split_motifs(rows: list[MotifRow], *, top_k: int) -> tuple[list[MotifRow], list[MotifRow]]:
    success = [row for row in rows if row.direction == "success"]
    failure = [row for row in rows if row.direction == "failure"]
    success.sort(
        key=lambda row: (row.support_difference, abs(row.log_odds_ratio), row.success_support),
        reverse=True,
    )
    failure.sort(
        key=lambda row: (row.support_difference, -abs(row.log_odds_ratio), -row.failure_support),
    )
    return success[:top_k], failure[:top_k]


def _load_precomputed_motif_rows(path: Path, *, scope: str) -> list[MotifRow]:
    rows = _read_csv(path)
    motifs: list[MotifRow] = []
    for entry in rows:
        support_difference = float(entry["support_difference"])
        motifs.append(
            MotifRow(
                motif=entry["motif"],
                tokens=entry["motif"].split("|"),
                length=int(entry["length"]),
                scope=scope,
                direction="success" if support_difference >= 0 else "failure",
                success_count=int(entry["success_count"]),
                failure_count=int(entry["failure_count"]),
                success_support=float(entry["success_support"]),
                failure_support=float(entry["failure_support"]),
                support_difference=support_difference,
                lift=float(entry["lift"]),
                log_odds_ratio=float(entry["log_odds_ratio"]),
                q_value=(
                    None
                    if entry.get("q_value") in {None, "", "nan"}
                    else float(entry["q_value"])
                ),
            )
        )
    return motifs


def _build_trace_summary(
    trace: JoinedTrace,
    *,
    success_motifs: list[MotifRow],
    failure_motifs: list[MotifRow],
) -> TraceSummary:
    matched_success = [
        motif.motif
        for motif in success_motifs
        if _contains_motif(trace.tokens, tuple(motif.tokens))
    ]
    matched_failure = [
        motif.motif
        for motif in failure_motifs
        if _contains_motif(trace.tokens, tuple(motif.tokens))
    ]
    return TraceSummary(
        trace_id=trace.trace_id,
        question_id=trace.question_id,
        sample_id=trace.sample_id,
        attempt_index=trace.attempt_index,
        predicted_answer=trace.predicted_answer,
        is_correct=trace.is_correct,
        tokenized_trace=trace.tokenized_trace,
        tokens=list(trace.tokens),
        token_count=len(trace.tokens),
        reasoning_trace=trace.reasoning_trace,
        matched_success_motifs=matched_success,
        matched_failure_motifs=matched_failure,
    )


def _answer_distribution(traces: list[JoinedTrace]) -> list[AnswerDistributionRow]:
    counts = Counter(trace.predicted_answer for trace in traces)
    total = sum(counts.values()) or 1
    rows = [
        AnswerDistributionRow(answer=answer, count=count, share=count / total)
        for answer, count in counts.most_common()
    ]
    return rows


def _question_tags(
    *,
    success_count: int,
    failure_count: int,
    distinct_predicted_answers: int,
    local_motif_available: bool,
) -> list[str]:
    tags: list[str] = []
    if success_count and failure_count:
        tags.append("mixed_outcomes")
    if failure_count and success_count / (success_count + failure_count) < 0.5:
        tags.append("hard_question")
    if distinct_predicted_answers >= 3:
        tags.append("high_diversity")
    if not local_motif_available:
        tags.append("low_evidence")
    return tags


def _insight_bullets(
    *,
    traces: list[JoinedTrace],
    success_motifs: list[MotifRow],
    failure_motifs: list[MotifRow],
    local_available: bool,
) -> list[str]:
    success = [trace for trace in traces if trace.is_correct]
    failure = [trace for trace in traces if not trace.is_correct]
    bullets: list[str] = []

    if success and failure:
        bullets.append(
            f"This question shows mixed outcomes: {len(success)} successful traces versus {len(failure)} failures."
        )
    elif success:
        bullets.append("All available traces are successful, so this page shows consistency more than separation.")
    else:
        bullets.append("All available traces are unsuccessful, so this page highlights a concentrated failure mode.")

    success_lengths = [len(trace.tokens) for trace in success]
    failure_lengths = [len(trace.tokens) for trace in failure]
    if success_lengths and failure_lengths:
        success_median = statistics.median(success_lengths)
        failure_median = statistics.median(failure_lengths)
        if success_median < failure_median:
            bullets.append("Successful traces tend to be shorter than failed traces on this question.")
        elif success_median > failure_median:
            bullets.append("Successful traces tend to be longer than failed traces on this question.")
        else:
            bullets.append("Trace length alone does not separate success and failure much on this question.")

    if local_available and success_motifs:
        motif = success_motifs[0]
        bullets.append(
            f'The strongest local success motif is "{motif.motif}", appearing more often in correct traces.'
        )
    if local_available and failure_motifs:
        motif = failure_motifs[0]
        bullets.append(
            f'The strongest local failure motif is "{motif.motif}", which clusters with incorrect attempts.'
        )

    distinct_answers = len({trace.predicted_answer for trace in traces})
    if distinct_answers >= 3:
        bullets.append(
            f"Answer diversity is high here: the traces produce {distinct_answers} distinct final answers."
        )
    return bullets[:4]


def _representative_traces(
    traces: list[JoinedTrace],
    *,
    success_motifs: list[MotifRow],
    failure_motifs: list[MotifRow],
    limit: int = 3,
) -> dict[str, list[TraceSummary]]:
    success_candidates = [trace for trace in traces if trace.is_correct]
    failure_candidates = [trace for trace in traces if not trace.is_correct]

    def _score_trace(trace: JoinedTrace, motifs: list[MotifRow], median_len: float) -> tuple[int, float, int]:
        matches = sum(
            1 for motif in motifs if _contains_motif(trace.tokens, tuple(motif.tokens))
        )
        length_delta = abs(len(trace.tokens) - median_len)
        return (-matches, length_delta, int(trace.attempt_index))

    def _select(group: list[JoinedTrace], motifs: list[MotifRow]) -> list[TraceSummary]:
        if not group:
            return []
        median_len = statistics.median(len(trace.tokens) for trace in group)
        ordered = sorted(group, key=lambda trace: _score_trace(trace, motifs, median_len))
        return [
            _build_trace_summary(trace, success_motifs=success_motifs, failure_motifs=failure_motifs)
            for trace in ordered[:limit]
        ]

    return {
        "success": _select(success_candidates, success_motifs),
        "failure": _select(failure_candidates, failure_motifs),
    }


def _question_payloads(
    traces: list[JoinedTrace],
    *,
    corpus_success_motifs: list[MotifRow],
    corpus_failure_motifs: list[MotifRow],
) -> tuple[QuestionSummary, QuestionDetail]:
    success_traces = [trace.tokens for trace in traces if trace.is_correct]
    failure_traces = [trace.tokens for trace in traces if not trace.is_correct]

    local_rows = _mine_motif_rows(
        success_traces,
        failure_traces,
        min_len=1,
        max_len=3,
        min_support_count=2,
        scope="question_local",
    )
    local_success, local_failure = _split_motifs(local_rows, top_k=8)
    local_available = bool(local_rows) and len(success_traces) >= 2 and len(failure_traces) >= 2

    question_token_sets = [trace.tokens for trace in traces]
    visible_global_success = [
        motif
        for motif in corpus_success_motifs
        if any(_contains_motif(tokens, tuple(motif.tokens)) for tokens in question_token_sets)
    ][:8]
    visible_global_failure = [
        motif
        for motif in corpus_failure_motifs
        if any(_contains_motif(tokens, tuple(motif.tokens)) for tokens in question_token_sets)
    ][:8]

    local_bucket = (
        MotifBucket(available=True, success=local_success, failure=local_failure)
        if local_available
        else MotifBucket(
            available=False,
            reason="Need at least two successful and two failed traces with repeated motifs.",
            success=[],
            failure=[],
        )
    )
    global_bucket = MotifBucket(
        available=bool(visible_global_success or visible_global_failure),
        reason=None if (visible_global_success or visible_global_failure) else "No corpus-global motifs matched this question.",
        success=visible_global_success,
        failure=visible_global_failure,
    )

    tags = _question_tags(
        success_count=len(success_traces),
        failure_count=len(failure_traces),
        distinct_predicted_answers=len({trace.predicted_answer for trace in traces}),
        local_motif_available=local_available,
    )
    summary = QuestionSummary(
        question_id=traces[0].question_id,
        question_text=traces[0].question_text,
        gold_answer=traces[0].gold_answer,
        benchmark_name=traces[0].benchmark_name,
        total_traces=len(traces),
        success_count=len(success_traces),
        failure_count=len(failure_traces),
        success_rate=len(success_traces) / len(traces),
        avg_token_count=sum(len(trace.tokens) for trace in traces) / len(traces),
        median_token_count=statistics.median(len(trace.tokens) for trace in traces),
        distinct_predicted_answers=len({trace.predicted_answer for trace in traces}),
        local_motif_separation=max(
            [abs(motif.support_difference) for motif in local_rows],
            default=0.0,
        ),
        top_success_motif=local_success[0].motif if local_success else None,
        top_failure_motif=local_failure[0].motif if local_failure else None,
        tags=tags,
    )

    detail = QuestionDetail(
        question_id=summary.question_id,
        question_text=summary.question_text,
        gold_answer=summary.gold_answer,
        benchmark_name=summary.benchmark_name,
        pilot_question_uid=traces[0].pilot_question_uid,
        total_traces=summary.total_traces,
        success_count=summary.success_count,
        failure_count=summary.failure_count,
        success_rate=summary.success_rate,
        avg_token_count=summary.avg_token_count,
        median_token_count=summary.median_token_count,
        distinct_predicted_answers=summary.distinct_predicted_answers,
        tags=tags,
        insights=_insight_bullets(
            traces=traces,
            success_motifs=local_success,
            failure_motifs=local_failure,
            local_available=local_available,
        ),
        answer_distribution=_answer_distribution(traces),
        local_motifs=local_bucket,
        global_motifs=global_bucket,
        representative_traces=_representative_traces(
            traces,
            success_motifs=local_success,
            failure_motifs=local_failure,
        ),
        all_traces=[
            _build_trace_summary(
                trace,
                success_motifs=local_success,
                failure_motifs=local_failure,
            )
            for trace in sorted(traces, key=lambda item: (int(item.sample_id), int(item.attempt_index)))
        ],
    )
    return summary, detail


def export_webapp_data(
    output_dir: Path,
    *,
    tokenized_csv: Path = DEFAULT_TOKENIZED_TRACE_CSV,
    raw_csv: Path = DEFAULT_RAW_TRACE_CSV,
    global_motifs_csv: Path | None = DEFAULT_GLOBAL_MOTIFS_CSV,
) -> dict[str, Path]:
    traces = load_joined_traces(tokenized_csv=tokenized_csv, raw_csv=raw_csv)
    by_question: dict[str, list[JoinedTrace]] = defaultdict(list)
    for trace in traces:
        by_question[trace.question_id].append(trace)

    if global_motifs_csv is not None and Path(global_motifs_csv).exists():
        corpus_rows = _load_precomputed_motif_rows(
            Path(global_motifs_csv),
            scope="corpus_global",
        )
    else:
        corpus_rows = _mine_motif_rows(
            [trace.tokens for trace in traces if trace.is_correct],
            [trace.tokens for trace in traces if not trace.is_correct],
            min_len=1,
            max_len=3,
            min_support_count=3,
            scope="corpus_global",
        )
    corpus_success, corpus_failure = _split_motifs(corpus_rows, top_k=12)

    summaries: list[QuestionSummary] = []
    details: dict[str, QuestionDetail] = {}
    for question_id, question_traces in sorted(by_question.items(), key=lambda item: int(item[0])):
        summary, detail = _question_payloads(
            question_traces,
            corpus_success_motifs=corpus_success,
            corpus_failure_motifs=corpus_failure,
        )
        summaries.append(summary)
        details[question_id] = detail

    summaries.sort(
        key=lambda row: (
            "mixed_outcomes" not in row.tags,
            -row.local_motif_separation,
            row.question_id,
        )
    )
    featured_question_ids = [summary.question_id for summary in summaries[:3]]
    token_counts = [len(trace.tokens) for trace in traces]
    overview = CorpusOverview(
        corpus_label="Curated pilot reasoning traces",
        num_questions=len(by_question),
        num_traces=len(traces),
        num_success=sum(trace.is_correct for trace in traces),
        num_failure=sum(not trace.is_correct for trace in traces),
        success_rate=sum(trace.is_correct for trace in traces) / len(traces),
        avg_token_count=sum(token_counts) / len(token_counts),
        median_token_count=statistics.median(token_counts),
        featured_question_ids=featured_question_ids,
        story_sections=[
            StorySection(
                id="hypothesis",
                title="Reasoning fingerprints",
                body=(
                    "Successful and failed solutions do not just differ in final answers. "
                    "They often reuse different local reasoning motifs that can be surfaced and compared."
                ),
            ),
            StorySection(
                id="method",
                title="Method shape",
                body=(
                    "We join raw traces with tokenized traces, mine motifs at the corpus and question level, "
                    "and then present representative traces alongside the motifs they instantiate."
                ),
            ),
            StorySection(
                id="significance",
                title="Why it matters",
                body=(
                    "If reasoning traces have structural fingerprints, motifs can become a practical lens for "
                    "debugging model failures, diagnosing benchmarks, and designing targeted interventions."
                ),
            ),
        ],
        top_success_motifs=corpus_success,
        top_failure_motifs=corpus_failure,
    )

    question_dir = output_dir / "question"
    question_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overview.json").write_text(
        overview.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (output_dir / "questions.json").write_text(
        json.dumps([summary.model_dump(mode="json") for summary in summaries], indent=2),
        encoding="utf-8",
    )
    for question_id, detail in details.items():
        (question_dir / f"{question_id}.json").write_text(
            detail.model_dump_json(indent=2),
            encoding="utf-8",
        )

    return {
        "overview": output_dir / "overview.json",
        "questions": output_dir / "questions.json",
        "question_dir": question_dir,
    }
