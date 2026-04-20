from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RequestConfig:
    model: str
    api_base: str
    api_key_env: str
    system_prompt: str
    temperature: float
    top_p: float
    max_tokens: int
    enable_thinking: bool
    timeout_seconds: float


@dataclass(frozen=True)
class BenchmarkPreset:
    name: str
    path: Path
    description: str
    answer_format: str


@dataclass(frozen=True)
class BenchmarkItem:
    question_id: int
    question: str
    gold_answer: str
    answer_format: str = "final"
    level: int | None = None
    subject: str | None = None
    source_id: str | None = None


@dataclass
class ReasoningResult:
    question_id: int
    question: str
    gold_answer: str
    predicted_answer: str
    has_clear_answer: bool
    is_correct: bool
    final_response_text: str
    reasoning_trace: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    finish_reason: str
    request_id: str | None
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TrajectoryRecord:
    question_id: int
    sample_id: int
    attempt_index: int
    question: str
    gold_answer: str
    predicted_answer: str
    has_clear_answer: bool
    is_correct: bool
    final_response_text: str
    reasoning_trace: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    finish_reason: str
    request_id: str | None
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TokenDomainParameters:
    domain_size: str
    abstraction_level: str
    flexibility: str


@dataclass(frozen=True)
class TraceExample:
    question_id: int
    sample_id: int
    attempt_index: int
    question: str
    gold_answer: str
    predicted_answer: str
    is_correct: bool
    reasoning_trace: str


@dataclass
class QuestionTraceBundle:
    question_id: int
    question: str
    gold_answer: str
    domain_parameters: TokenDomainParameters
    traces: list[TraceExample]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TokenDomainDefinition:
    token: str
    definition: str
    when_to_use: str
    when_not_to_use: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TokenizedStep:
    step_index: int
    source_text: str
    token: str
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TokenizedTrace:
    sample_id: int
    attempt_index: int
    predicted_answer: str
    is_correct: bool
    token_sequence: list[str]
    steps: list[TokenizedStep]
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class QuestionTokenizationArtifact:
    question_id: int
    question: str
    gold_answer: str
    domain_parameters: TokenDomainParameters
    token_domain: list[TokenDomainDefinition]
    trace_mappings: list[TokenizedTrace]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class AdaptiveQuestionSummary:
    question_id: int
    question: str
    gold_answer: str
    scout_samples_target: int
    final_samples_target: int
    accepted_scout: int
    accepted_final: int
    right_scout: int
    wrong_scout: int
    right_final: int
    wrong_final: int
    attempts_used: int
    was_densified: bool
    decision: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DataClasses:
    RequestConfig = RequestConfig
    BenchmarkPreset = BenchmarkPreset
    BenchmarkItem = BenchmarkItem
    ReasoningResult = ReasoningResult
    TrajectoryRecord = TrajectoryRecord
    TokenDomainParameters = TokenDomainParameters
    TraceExample = TraceExample
    QuestionTraceBundle = QuestionTraceBundle
    TokenDomainDefinition = TokenDomainDefinition
    TokenizedStep = TokenizedStep
    TokenizedTrace = TokenizedTrace
    QuestionTokenizationArtifact = QuestionTokenizationArtifact
    AdaptiveQuestionSummary = AdaptiveQuestionSummary
