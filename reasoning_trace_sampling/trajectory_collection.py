"""Trajectory collection utilities for fixed-budget and adaptive sampling.

This module owns the concurrency and bookkeeping around repeated calls to
``ReasoningTraceSampling.ask_one``. It intentionally separates single-attempt
sampling from higher-level collection policies:

- ``TraceCollector``: fixed accepted-sample target per question
- ``AdaptiveTraceCollector``: scout each question, then densify only mixed ones
"""

from __future__ import annotations

import csv
import json
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from .data_classes import AdaptiveQuestionSummary, BenchmarkItem, ReasoningResult, TrajectoryRecord
from .sampling import ReasoningTraceSampling


TRAJECTORY_FIELDNAMES = [
    "question_id",
    "sample_id",
    "attempt_index",
    "question",
    "gold_answer",
    "predicted_answer",
    "has_clear_answer",
    "is_correct",
    "answer_source",
    "answer_validation",
    "final_response_text",
    "reasoning_trace",
    "temperature",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "finish_reason",
    "request_id",
    "error",
]

ADAPTIVE_SUMMARY_FIELDNAMES = [
    "question_id",
    "question",
    "gold_answer",
    "scout_samples_target",
    "final_samples_target",
    "accepted_scout",
    "accepted_final",
    "right_scout",
    "wrong_scout",
    "right_final",
    "wrong_final",
    "attempts_used",
    "was_densified",
    "decision",
]


class CsvRow(Protocol):
    def to_dict(self) -> dict[str, object]:
        ...


class CollectionLogger:
    def __init__(self, path: Path, *, append: bool = False) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not append:
            self.path.write_text("", encoding="utf-8")
        self._lock = threading.Lock()

    def log(self, event_type: str, **payload: Any) -> None:
        record = {"event": event_type, **payload}
        line = json.dumps(record, ensure_ascii=True) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)


class StreamingCsvWriter:
    def __init__(self, path: Path, *, fieldnames: Sequence[str]) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def write_row(self, row: CsvRow) -> None:
        with self._lock:
            with self.path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(row.to_dict())


@dataclass
class QuestionCollectionState:
    item: BenchmarkItem
    accepted: list[TrajectoryRecord] = field(default_factory=list)
    next_attempt_index: int = 0
    completed_attempts: int = 0
    in_flight: int = 0
    is_done: bool = False


class ProgressReporter:
    """Render lightweight stderr progress for long-running collection jobs."""

    def __init__(self, *, total_questions: int, total_target_samples: int) -> None:
        self.total_questions = total_questions
        self.total_target_samples = total_target_samples
        self.total_attempts = 0
        self.started_attempts = 0
        self.in_flight_attempts = 0
        self.accepted_samples = 0
        self.completed_questions = 0
        self.mixed_questions = 0
        self.all_correct_questions = 0
        self.all_wrong_questions = 0
        self.no_answer_questions = 0
        self.llm_validated_right = 0
        self.llm_validated_wrong = 0
        self.llm_verifier_failed = 0
        self.verifier_active = 0
        self._last_heartbeat_at = 0.0
        self._lock = threading.Lock()

    def record_attempt_started(
        self,
        *,
        question_id: int,
        attempt_index: int,
        temperature: float | None,
        status_prefix: str = "",
    ) -> None:
        with self._lock:
            self.started_attempts += 1
            self.in_flight_attempts += 1
            temp_note = f" temp={temperature:g}" if temperature is not None else ""
            self._render(f"{status_prefix}started q={question_id} attempt={attempt_index}{temp_note}")

    def record_verifier_event(
        self,
        *,
        event_type: str,
        question_id: int,
        attempt_index: int | None,
        answer_validation: str = "",
        status_prefix: str = "",
    ) -> None:
        with self._lock:
            if event_type == "verifier_started":
                self.verifier_active += 1
                status = "LLM verifier started"
            elif event_type in {"verifier_finished", "verifier_failed"}:
                self.verifier_active = max(0, self.verifier_active - 1)
                status = answer_validation or event_type.replace("_", " ")
            else:
                status = event_type.replace("_", " ")
            attempt_note = f" attempt={attempt_index}" if attempt_index is not None else ""
            self._render(f"{status_prefix}q={question_id}{attempt_note} {status}")

    def record_heartbeat(self, *, pending_count: int) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat_at < 1.0:
            return
        with self._lock:
            self._last_heartbeat_at = now
            self._render(f"waiting pending={pending_count}")

    def record_attempt(
        self,
        *,
        question_id: int,
        accepted_for_question: int,
        samples_per_question: int,
        attempts_used: int,
        max_attempts_per_question: int,
        accepted_this_attempt: bool,
        answer_validation: str = "",
        status_prefix: str = "",
    ) -> None:
        with self._lock:
            self.total_attempts += 1
            self.in_flight_attempts = max(0, self.in_flight_attempts - 1)
            if accepted_this_attempt:
                self.accepted_samples += 1
                if answer_validation == "LLM validated RIGHT":
                    self.llm_validated_right += 1
                elif answer_validation == "LLM validated WRONG":
                    self.llm_validated_wrong += 1
            if answer_validation.startswith("LLM verifier failed"):
                self.llm_verifier_failed += 1
            validation_note = f" {answer_validation}" if answer_validation else ""
            self._render(
                f"{status_prefix}q={question_id} accepted={accepted_for_question}/{samples_per_question} "
                f"attempts={attempts_used}/{max_attempts_per_question}{validation_note}"
            )

    def record_question_done(
        self,
        *,
        question_id: int,
        accepted_for_question: int,
        samples_per_question: int,
        attempts_used: int,
        right_count: int | None = None,
        wrong_count: int | None = None,
        status_suffix: str = "",
    ) -> None:
        with self._lock:
            self.completed_questions += 1
            self._record_question_bucket(
                accepted_for_question=accepted_for_question,
                right_count=right_count,
                wrong_count=wrong_count,
            )
            self._render(
                f"finished q={question_id} accepted={accepted_for_question}/{samples_per_question} "
                f"attempts={attempts_used}{status_suffix}"
            )

    def finish(self) -> None:
        with self._lock:
            self._render("done")
            print(file=sys.stderr)

    def _render(self, current_status: str) -> None:
        message = (
            f"\r[{self.completed_questions}/{self.total_questions} questions] "
            f"[{self.accepted_samples}/{self.total_target_samples} accepted] "
            f"[mixed={self.mixed_questions} correct={self.all_correct_questions} "
            f"wrong={self.all_wrong_questions} no_answer={self.no_answer_questions}] "
            f"[llm right={self.llm_validated_right} wrong={self.llm_validated_wrong} "
            f"failed={self.llm_verifier_failed} active={self.verifier_active}] "
            f"[attempts started={self.started_attempts} done={self.total_attempts} "
            f"in_flight={self.in_flight_attempts}] {current_status}"
        )
        print(message, end="", file=sys.stderr, flush=True)

    def _record_question_bucket(
        self,
        *,
        accepted_for_question: int,
        right_count: int | None,
        wrong_count: int | None,
    ) -> None:
        if accepted_for_question == 0:
            self.no_answer_questions += 1
            return
        if right_count is None or wrong_count is None:
            return
        if right_count > 0 and wrong_count > 0:
            self.mixed_questions += 1
            return
        if right_count > 0:
            self.all_correct_questions += 1
            return
        if wrong_count > 0:
            self.all_wrong_questions += 1


class TraceCollector:
    """Collect a fixed number of accepted trajectories per question."""

    def __init__(self, sampler: ReasoningTraceSampling) -> None:
        self.sampler = sampler

    def collect_for_item(
        self,
        item: BenchmarkItem,
        *,
        samples_per_question: int,
        max_attempts_per_question: int,
        temperature_schedule: Sequence[float] | None = None,
        progress_reporter: ProgressReporter | None = None,
        logger: CollectionLogger | None = None,
        accepted_sink: Callable[[TrajectoryRecord], None] | None = None,
    ) -> list[TrajectoryRecord]:
        self._validate_fixed_collection_args(
            sample_goal=samples_per_question,
            max_attempts_per_question=max_attempts_per_question,
        )
        accepted, _attempt_index = self._collect_until(
            item,
            sample_goal=samples_per_question,
            max_attempts_per_question=max_attempts_per_question,
            accepted=[],
            attempt_index=0,
            temperature_schedule=temperature_schedule,
            progress_reporter=progress_reporter,
            logger=logger,
            accepted_sink=accepted_sink,
        )
        return accepted

    def collect_many(
        self,
        items: Sequence[BenchmarkItem],
        *,
        samples_per_question: int,
        max_attempts_per_question: int,
        workers: int,
        show_progress: bool = False,
        log_path: Path | None = None,
        temperature_schedule: Sequence[float] | None = None,
        stream_output_path: Path | None = None,
    ) -> list[TrajectoryRecord]:
        if not items:
            return []
        self._validate_fixed_collection_args(
            sample_goal=samples_per_question,
            max_attempts_per_question=max_attempts_per_question,
        )

        rows: list[TrajectoryRecord] = []
        max_workers = max(1, min(workers, len(items)))
        progress_reporter = None
        logger = CollectionLogger(log_path) if log_path is not None else None
        stream_writer = (
            StreamingCsvWriter(stream_output_path, fieldnames=TRAJECTORY_FIELDNAMES)
            if stream_output_path is not None
            else None
        )
        if show_progress:
            progress_reporter = ProgressReporter(
                total_questions=len(items),
                total_target_samples=len(items) * samples_per_question,
            )
        self._log_run_started(
            logger,
            mode="fixed",
            total_questions=len(items),
            samples_per_question=samples_per_question,
            max_attempts_per_question=max_attempts_per_question,
            temperature_schedule=temperature_schedule,
            stream_output_path=stream_output_path,
            parallelism="attempt",
            workers=max_workers,
        )
        states = [QuestionCollectionState(item=item) for item in items]
        next_state_index = 0

        def can_submit(state: QuestionCollectionState) -> bool:
            if state.is_done:
                return False
            if state.next_attempt_index >= max_attempts_per_question:
                return False
            return len(state.accepted) + state.in_flight < samples_per_question

        def next_schedulable_state() -> QuestionCollectionState | None:
            nonlocal next_state_index
            for offset in range(len(states)):
                state_index = (next_state_index + offset) % len(states)
                state = states[state_index]
                if can_submit(state):
                    next_state_index = (state_index + 1) % len(states)
                    return state
            return None

        def submit_attempt(
            executor: ThreadPoolExecutor,
            state: QuestionCollectionState,
        ):
            attempt_index = state.next_attempt_index
            state.next_attempt_index += 1
            state.in_flight += 1
            temperature = self._temperature_for_attempt(
                temperature_schedule=temperature_schedule,
                attempt_index=attempt_index,
            )
            if progress_reporter is not None:
                progress_reporter.record_attempt_started(
                    question_id=state.item.question_id,
                    attempt_index=attempt_index,
                    temperature=temperature,
                )
            self._log_attempt_started(
                logger,
                stage="fixed",
                question_id=state.item.question_id,
                attempt_index=attempt_index,
                temperature=temperature,
            )

            progress_callback = self._build_progress_callback(
                progress_reporter=progress_reporter,
                logger=logger,
                question_id=state.item.question_id,
                attempt_index=attempt_index,
                progress_status_prefix="",
                log_stage="fixed",
            )

            return (
                executor.submit(
                    self.sampler.ask_one,
                    state.item,
                    temperature=temperature,
                    progress_callback=progress_callback,
                    attempt_index=attempt_index,
                ),
                state,
                attempt_index,
            )

        def fill_pending(executor: ThreadPoolExecutor, pending: dict[object, tuple[QuestionCollectionState, int]]) -> None:
            while len(pending) < max_workers:
                state = next_schedulable_state()
                if state is None:
                    return
                future, future_state, attempt_index = submit_attempt(executor, state)
                pending[future] = (future_state, attempt_index)

        def finish_question_if_ready(state: QuestionCollectionState) -> None:
            if state.is_done:
                return
            if len(state.accepted) < samples_per_question and (
                state.next_attempt_index < max_attempts_per_question or state.in_flight > 0
            ):
                return
            state.is_done = True
            right_count = sum(1 for row in state.accepted if row.is_correct)
            wrong_count = sum(1 for row in state.accepted if not row.is_correct)
            if progress_reporter is not None:
                progress_reporter.record_question_done(
                    question_id=state.item.question_id,
                    accepted_for_question=len(state.accepted),
                    samples_per_question=samples_per_question,
                    attempts_used=state.completed_attempts,
                    right_count=right_count,
                    wrong_count=wrong_count,
                )
            if logger is not None:
                logger.log(
                    "question_completed",
                    mode="fixed",
                    question_id=state.item.question_id,
                    accepted=len(state.accepted),
                    target=samples_per_question,
                    attempts_used=state.completed_attempts,
                    right=right_count,
                    wrong=wrong_count,
                )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending: dict[object, tuple[QuestionCollectionState, int]] = {}
            fill_pending(executor, pending)
            while pending:
                done, _not_done = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                if not done:
                    if progress_reporter is not None:
                        progress_reporter.record_heartbeat(pending_count=len(pending))
                    continue
                for future in done:
                    state, attempt_index = pending.pop(future)
                    state.in_flight -= 1
                    state.completed_attempts += 1
                    result = future.result()
                    accepted_this_attempt = result.has_clear_answer
                    if result.has_clear_answer:
                        row = self._to_trajectory_record(
                            result=result,
                            sample_id=len(state.accepted),
                            attempt_index=attempt_index,
                        )
                        state.accepted.append(row)
                        rows.append(row)
                        if stream_writer is not None:
                            stream_writer.write_row(row)
                    if progress_reporter is not None:
                        progress_reporter.record_attempt(
                            question_id=state.item.question_id,
                            accepted_for_question=len(state.accepted),
                            samples_per_question=samples_per_question,
                            attempts_used=state.completed_attempts,
                            max_attempts_per_question=max_attempts_per_question,
                            accepted_this_attempt=accepted_this_attempt,
                            answer_validation=result.answer_validation,
                        )
                    self._log_attempt_result(
                        logger,
                        stage="fixed",
                        question_id=state.item.question_id,
                        attempt_index=attempt_index,
                        accepted=accepted_this_attempt,
                        accepted_count=len(state.accepted),
                        sample_goal=samples_per_question,
                        result=result,
                    )
                    finish_question_if_ready(state)
                fill_pending(executor, pending)
        if progress_reporter is not None:
            progress_reporter.finish()
        if logger is not None:
            logger.log(
                "run_completed",
                mode="fixed",
                total_rows=len(rows),
                total_correct=sum(1 for row in rows if row.is_correct),
                total_wrong=sum(1 for row in rows if not row.is_correct),
            )

        rows.sort(key=lambda row: (row.question_id, row.sample_id))
        return rows

    @staticmethod
    def save_csv(path: Path, rows: Sequence[TrajectoryRecord]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRAJECTORY_FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow(row.to_dict())

    @staticmethod
    def _to_trajectory_record(
        *,
        result: ReasoningResult,
        sample_id: int,
        attempt_index: int,
    ) -> TrajectoryRecord:
        return TrajectoryRecord(
            question_id=result.question_id,
            sample_id=sample_id,
            attempt_index=attempt_index,
            question=result.question,
            gold_answer=result.gold_answer,
            predicted_answer=result.predicted_answer,
            has_clear_answer=result.has_clear_answer,
            is_correct=result.is_correct,
            answer_source=result.answer_source,
            answer_validation=result.answer_validation,
            final_response_text=result.final_response_text,
            reasoning_trace=result.reasoning_trace,
            temperature=result.temperature,
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            reasoning_tokens=result.reasoning_tokens,
            finish_reason=result.finish_reason,
            request_id=result.request_id,
            error=result.error,
        )

    @staticmethod
    def _validate_fixed_collection_args(
        *,
        sample_goal: int,
        max_attempts_per_question: int,
    ) -> None:
        if sample_goal < 1:
            raise ValueError("sample_goal must be at least 1")
        if max_attempts_per_question < 1:
            raise ValueError("max_attempts_per_question must be at least 1")

    @staticmethod
    def _log_run_started(
        logger: CollectionLogger | None,
        *,
        mode: str,
        temperature_schedule: Sequence[float] | None,
        stream_output_path: Path | None = None,
        summary_stream_output_path: Path | None = None,
        **payload: object,
    ) -> None:
        if logger is None:
            return
        logger.log(
            "run_started",
            mode=mode,
            temperature_schedule=list(temperature_schedule) if temperature_schedule else None,
            stream_output_path=str(stream_output_path) if stream_output_path is not None else None,
            summary_stream_output_path=(
                str(summary_stream_output_path) if summary_stream_output_path is not None else None
            ),
            **payload,
        )

    @staticmethod
    def _log_attempt_started(
        logger: CollectionLogger | None,
        *,
        stage: str,
        question_id: int,
        attempt_index: int,
        temperature: float | None,
    ) -> None:
        if logger is None:
            return
        logger.log(
            "attempt_started",
            mode="collection",
            stage=stage,
            question_id=question_id,
            attempt_index=attempt_index,
            temperature=temperature,
        )

    @staticmethod
    def _log_attempt_result(
        logger: CollectionLogger | None,
        *,
        stage: str,
        question_id: int,
        attempt_index: int,
        accepted: bool,
        accepted_count: int,
        sample_goal: int,
        result: ReasoningResult,
    ) -> None:
        if logger is None:
            return
        logger.log(
            "attempt_result",
            mode="collection",
            stage=stage,
            question_id=question_id,
            attempt_index=attempt_index,
            accepted=accepted,
            accepted_count=accepted_count,
            sample_goal=sample_goal,
            predicted_answer=result.predicted_answer,
            answer_source=result.answer_source,
            answer_validation=result.answer_validation,
            is_correct=result.is_correct,
            finish_reason=result.finish_reason,
            request_id=result.request_id,
            temperature=result.temperature,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            reasoning_tokens=result.reasoning_tokens,
            error=result.error,
        )

    @staticmethod
    def _build_progress_callback(
        *,
        progress_reporter: ProgressReporter | None,
        logger: CollectionLogger | None,
        question_id: int,
        attempt_index: int,
        progress_status_prefix: str,
        log_stage: str,
    ) -> Callable[[str, dict[str, object]], None]:
        def progress_callback(event_type: str, payload: dict[str, object]) -> None:
            answer_validation = str(payload.get("answer_validation", ""))
            if progress_reporter is not None:
                progress_reporter.record_verifier_event(
                    event_type=event_type,
                    question_id=question_id,
                    attempt_index=attempt_index,
                    answer_validation=answer_validation,
                    status_prefix=progress_status_prefix,
                )
            if logger is not None:
                logger.log(
                    event_type,
                    mode="collection",
                    stage=log_stage,
                    question_id=question_id,
                    attempt_index=attempt_index,
                    answer_validation=answer_validation,
                )

        return progress_callback

    def _collect_until(
        self,
        item: BenchmarkItem,
        *,
        sample_goal: int,
        max_attempts_per_question: int,
        accepted: list[TrajectoryRecord],
        attempt_index: int,
        temperature_schedule: Sequence[float] | None = None,
        progress_reporter: ProgressReporter | None = None,
        progress_status_prefix: str = "",
        logger: CollectionLogger | None = None,
        log_stage: str = "fixed",
        accepted_sink: Callable[[TrajectoryRecord], None] | None = None,
    ) -> tuple[list[TrajectoryRecord], int]:
        while len(accepted) < sample_goal and attempt_index < max_attempts_per_question:
            temperature = self._temperature_for_attempt(
                temperature_schedule=temperature_schedule,
                attempt_index=attempt_index,
            )
            if progress_reporter is not None:
                progress_reporter.record_attempt_started(
                    question_id=item.question_id,
                    attempt_index=attempt_index,
                    temperature=temperature,
                    status_prefix=progress_status_prefix,
                )
            self._log_attempt_started(
                logger,
                stage=log_stage,
                question_id=item.question_id,
                attempt_index=attempt_index,
                temperature=temperature,
            )

            progress_callback = self._build_progress_callback(
                progress_reporter=progress_reporter,
                logger=logger,
                question_id=item.question_id,
                attempt_index=attempt_index,
                progress_status_prefix=progress_status_prefix,
                log_stage=log_stage,
            )

            result = self.sampler.ask_one(
                item,
                temperature=temperature,
                progress_callback=progress_callback,
                attempt_index=attempt_index,
            )
            accepted_this_attempt = result.has_clear_answer
            if result.has_clear_answer:
                row = self._to_trajectory_record(
                    result=result,
                    sample_id=len(accepted),
                    attempt_index=attempt_index,
                )
                accepted.append(row)
                if accepted_sink is not None:
                    accepted_sink(row)
            if progress_reporter is not None:
                progress_reporter.record_attempt(
                    question_id=item.question_id,
                    accepted_for_question=len(accepted),
                    samples_per_question=sample_goal,
                    attempts_used=attempt_index + 1,
                    max_attempts_per_question=max_attempts_per_question,
                    accepted_this_attempt=accepted_this_attempt,
                    answer_validation=result.answer_validation,
                    status_prefix=progress_status_prefix,
                )
            self._log_attempt_result(
                logger,
                stage=log_stage,
                question_id=item.question_id,
                attempt_index=attempt_index,
                accepted=accepted_this_attempt,
                accepted_count=len(accepted),
                sample_goal=sample_goal,
                result=result,
            )
            attempt_index += 1
        return accepted, attempt_index

    @staticmethod
    def _temperature_for_attempt(
        *,
        temperature_schedule: Sequence[float] | None,
        attempt_index: int,
    ) -> float | None:
        if not temperature_schedule:
            return None
        return temperature_schedule[attempt_index % len(temperature_schedule)]


class AdaptiveTraceCollector:
    """Collect scout traces first, then spend more budget on mixed questions."""

    def __init__(self, trace_collector: TraceCollector) -> None:
        self.trace_collector = trace_collector

    def collect_for_item(
        self,
        item: BenchmarkItem,
        *,
        scout_samples: int,
        target_samples: int,
        scout_max_attempts_per_question: int,
        max_attempts_per_question: int,
        temperature_schedule: Sequence[float] | None = None,
        progress_reporter: ProgressReporter | None = None,
        logger: CollectionLogger | None = None,
        accepted_sink: Callable[[TrajectoryRecord], None] | None = None,
    ) -> tuple[list[TrajectoryRecord], AdaptiveQuestionSummary]:
        self._validate_adaptive_collection_args(
            scout_samples=scout_samples,
            target_samples=target_samples,
            scout_max_attempts_per_question=scout_max_attempts_per_question,
            max_attempts_per_question=max_attempts_per_question,
        )

        accepted, attempt_index = self.trace_collector._collect_until(
            item,
            sample_goal=scout_samples,
            max_attempts_per_question=scout_max_attempts_per_question,
            accepted=[],
            attempt_index=0,
            temperature_schedule=temperature_schedule,
            progress_reporter=progress_reporter,
            progress_status_prefix="scout ",
            logger=logger,
            log_stage="scout",
            accepted_sink=accepted_sink,
        )
        accepted_scout = len(accepted)
        right_scout = sum(1 for row in accepted if row.is_correct)
        wrong_scout = accepted_scout - right_scout

        should_densify = right_scout > 0 and wrong_scout > 0 and accepted_scout < target_samples
        decision = self._decision_for_scout(
            accepted_scout=accepted_scout,
            right_scout=right_scout,
            wrong_scout=wrong_scout,
            scout_samples=scout_samples,
            should_densify=should_densify,
        )

        if should_densify:
            accepted, attempt_index = self.trace_collector._collect_until(
                item,
                sample_goal=target_samples,
                max_attempts_per_question=max_attempts_per_question,
                accepted=accepted,
                attempt_index=attempt_index,
                temperature_schedule=temperature_schedule,
                progress_reporter=progress_reporter,
                progress_status_prefix="densify ",
                logger=logger,
                log_stage="densify",
                accepted_sink=accepted_sink,
            )

        accepted_final = len(accepted)
        right_final = sum(1 for row in accepted if row.is_correct)
        wrong_final = accepted_final - right_final

        summary = AdaptiveQuestionSummary(
            question_id=item.question_id,
            question=item.question,
            gold_answer=item.gold_answer,
            scout_samples_target=scout_samples,
            final_samples_target=target_samples if should_densify else scout_samples,
            accepted_scout=accepted_scout,
            accepted_final=accepted_final,
            right_scout=right_scout,
            wrong_scout=wrong_scout,
            right_final=right_final,
            wrong_final=wrong_final,
            attempts_used=attempt_index,
            was_densified=should_densify,
            decision=decision,
        )
        return accepted, summary

    def collect_many(
        self,
        items: Sequence[BenchmarkItem],
        *,
        scout_samples: int,
        target_samples: int,
        scout_max_attempts_per_question: int,
        max_attempts_per_question: int,
        workers: int,
        show_progress: bool = False,
        log_path: Path | None = None,
        temperature_schedule: Sequence[float] | None = None,
        stream_output_path: Path | None = None,
        summary_stream_output_path: Path | None = None,
    ) -> tuple[list[TrajectoryRecord], list[AdaptiveQuestionSummary]]:
        if not items:
            return [], []
        self._validate_adaptive_collection_args(
            scout_samples=scout_samples,
            target_samples=target_samples,
            scout_max_attempts_per_question=scout_max_attempts_per_question,
            max_attempts_per_question=max_attempts_per_question,
        )

        rows: list[TrajectoryRecord] = []
        summaries: list[AdaptiveQuestionSummary] = []
        max_workers = max(1, min(workers, len(items)))
        progress_reporter = None
        logger = CollectionLogger(log_path) if log_path is not None else None
        stream_writer = (
            StreamingCsvWriter(stream_output_path, fieldnames=TRAJECTORY_FIELDNAMES)
            if stream_output_path is not None
            else None
        )
        summary_stream_writer = (
            StreamingCsvWriter(summary_stream_output_path, fieldnames=ADAPTIVE_SUMMARY_FIELDNAMES)
            if summary_stream_output_path is not None
            else None
        )
        if show_progress:
            progress_reporter = ProgressReporter(
                total_questions=len(items),
                total_target_samples=len(items) * target_samples,
            )
        self.trace_collector._log_run_started(
            logger,
            mode="adaptive",
            total_questions=len(items),
            scout_samples=scout_samples,
            target_samples=target_samples,
            scout_max_attempts_per_question=scout_max_attempts_per_question,
            max_attempts_per_question=max_attempts_per_question,
            temperature_schedule=temperature_schedule,
            stream_output_path=stream_output_path,
            summary_stream_output_path=summary_stream_output_path,
            workers=max_workers,
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.collect_for_item,
                    item,
                    scout_samples=scout_samples,
                    target_samples=target_samples,
                    scout_max_attempts_per_question=scout_max_attempts_per_question,
                    max_attempts_per_question=max_attempts_per_question,
                    temperature_schedule=temperature_schedule,
                    progress_reporter=progress_reporter,
                    logger=logger,
                    accepted_sink=stream_writer.write_row if stream_writer is not None else None,
                ): item.question_id
                for item in items
            }
            for future in as_completed(futures):
                question_rows, summary = future.result()
                rows.extend(question_rows)
                summaries.append(summary)
                if summary_stream_writer is not None:
                    summary_stream_writer.write_row(summary)
                if progress_reporter is not None:
                    progress_reporter.record_question_done(
                        question_id=summary.question_id,
                        accepted_for_question=summary.accepted_final,
                        samples_per_question=target_samples if summary.was_densified else scout_samples,
                        attempts_used=summary.attempts_used,
                        right_count=summary.right_final,
                        wrong_count=summary.wrong_final,
                        status_suffix=f" decision={summary.decision}",
                    )
                if logger is not None:
                    logger.log(
                        "question_completed",
                        mode="adaptive",
                        question_id=summary.question_id,
                        decision=summary.decision,
                        was_densified=summary.was_densified,
                        accepted_scout=summary.accepted_scout,
                        accepted_final=summary.accepted_final,
                        right_scout=summary.right_scout,
                        wrong_scout=summary.wrong_scout,
                        right_final=summary.right_final,
                        wrong_final=summary.wrong_final,
                        attempts_used=summary.attempts_used,
                    )

        if progress_reporter is not None:
            progress_reporter.finish()
        if logger is not None:
            logger.log(
                "run_completed",
                mode="adaptive",
                total_rows=len(rows),
                total_questions=len(summaries),
                mixed_final=sum(1 for row in summaries if row.right_final > 0 and row.wrong_final > 0),
                densified=sum(1 for row in summaries if row.was_densified),
            )

        rows.sort(key=lambda row: (row.question_id, row.sample_id))
        summaries.sort(key=lambda row: row.question_id)
        return rows, summaries

    @staticmethod
    def save_summary_csv(path: Path, rows: Sequence[AdaptiveQuestionSummary]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ADAPTIVE_SUMMARY_FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow(row.to_dict())

    @staticmethod
    def _decision_for_scout(
        *,
        accepted_scout: int,
        right_scout: int,
        wrong_scout: int,
        scout_samples: int,
        should_densify: bool,
    ) -> str:
        if should_densify:
            return "mixed_densified"
        if accepted_scout == 0:
            return "no_accepted_traces"
        if right_scout > 0 and wrong_scout > 0:
            return "mixed_target_reached"
        if accepted_scout < scout_samples:
            if right_scout > 0:
                return "partial_all_correct_stop"
            if wrong_scout > 0:
                return "partial_all_wrong_stop"
            return "partial_no_signal_stop"
        if wrong_scout == 0:
            return "all_correct_stop"
        if right_scout == 0:
            return "all_wrong_stop"
        return "scout_stop"

    @staticmethod
    def _validate_adaptive_collection_args(
        *,
        scout_samples: int,
        target_samples: int,
        scout_max_attempts_per_question: int,
        max_attempts_per_question: int,
    ) -> None:
        if scout_samples < 1:
            raise ValueError("scout_samples must be at least 1")
        if target_samples < scout_samples:
            raise ValueError("target_samples must be greater than or equal to scout_samples")
        if scout_max_attempts_per_question < 1:
            raise ValueError("scout_max_attempts_per_question must be at least 1")
        if max_attempts_per_question < 1:
            raise ValueError("max_attempts_per_question must be at least 1")
        if scout_max_attempts_per_question > max_attempts_per_question:
            raise ValueError("scout_max_attempts_per_question must be <= max_attempts_per_question")
