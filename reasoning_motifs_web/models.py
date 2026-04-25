from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


MotifDirection = Literal["success", "failure"]
MotifScope = Literal["question_local", "corpus_global"]


class StorySection(BaseModel):
    id: str
    title: str
    body: str


class MotifRow(BaseModel):
    motif: str
    tokens: list[str]
    length: int
    scope: MotifScope
    direction: MotifDirection
    success_count: int
    failure_count: int
    success_support: float
    failure_support: float
    support_difference: float
    lift: float
    log_odds_ratio: float
    q_value: float | None = None


class TraceSummary(BaseModel):
    trace_id: str
    question_id: str
    sample_id: str
    attempt_index: str
    predicted_answer: str
    is_correct: bool
    tokenized_trace: str
    tokens: list[str]
    token_count: int
    reasoning_trace: str
    matched_success_motifs: list[str] = Field(default_factory=list)
    matched_failure_motifs: list[str] = Field(default_factory=list)


class AnswerDistributionRow(BaseModel):
    answer: str
    count: int
    share: float


class MotifBucket(BaseModel):
    available: bool
    reason: str | None = None
    success: list[MotifRow] = Field(default_factory=list)
    failure: list[MotifRow] = Field(default_factory=list)


class QuestionSummary(BaseModel):
    question_id: str
    question_text: str
    gold_answer: str
    benchmark_name: str
    total_traces: int
    success_count: int
    failure_count: int
    success_rate: float
    avg_token_count: float
    median_token_count: float
    distinct_predicted_answers: int
    local_motif_separation: float
    top_success_motif: str | None = None
    top_failure_motif: str | None = None
    tags: list[str] = Field(default_factory=list)


class QuestionDetail(BaseModel):
    question_id: str
    question_text: str
    gold_answer: str
    benchmark_name: str
    pilot_question_uid: str
    total_traces: int
    success_count: int
    failure_count: int
    success_rate: float
    avg_token_count: float
    median_token_count: float
    distinct_predicted_answers: int
    tags: list[str] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    answer_distribution: list[AnswerDistributionRow] = Field(default_factory=list)
    local_motifs: MotifBucket
    global_motifs: MotifBucket
    representative_traces: dict[str, list[TraceSummary]]
    all_traces: list[TraceSummary]


class CorpusOverview(BaseModel):
    corpus_label: str
    num_questions: int
    num_traces: int
    num_success: int
    num_failure: int
    success_rate: float
    avg_token_count: float
    median_token_count: float
    featured_question_ids: list[str] = Field(default_factory=list)
    story_sections: list[StorySection] = Field(default_factory=list)
    top_success_motifs: list[MotifRow] = Field(default_factory=list)
    top_failure_motifs: list[MotifRow] = Field(default_factory=list)
