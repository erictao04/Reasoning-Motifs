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


class DataClasses:
    RequestConfig = RequestConfig
    BenchmarkPreset = BenchmarkPreset
    BenchmarkItem = BenchmarkItem
    ReasoningResult = ReasoningResult
    TrajectoryRecord = TrajectoryRecord
