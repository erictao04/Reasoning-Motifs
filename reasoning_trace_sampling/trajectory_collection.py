from __future__ import annotations

import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from .data_classes import AdaptiveQuestionSummary, BenchmarkItem, ReasoningResult, TrajectoryRecord
from .sampling import ReasoningTraceSampling


class CollectionLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event_type: str, **payload: Any) -> None:
        record = {"event": event_type, **payload}
        line = json.dumps(record, ensure_ascii=True) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)


class ProgressReporter:
    def __init__(self, *, total_questions: int, total_target_samples: int) -> None:
        self.total_questions = total_questions
        self.total_target_samples = total_target_samples
        self.total_attempts = 0
        self.accepted_samples = 0
        self.completed_questions = 0
        self._lock = threading.Lock()

    def record_attempt(
        self,
        *,
        question_id: int,
        accepted_for_question: int,
        samples_per_question: int,
        attempts_used: int,
        max_attempts_per_question: int,
        accepted_this_attempt: bool,
        status_prefix: str = "",
    ) -> None:
        with self._lock:
            self.total_attempts += 1
            if accepted_this_attempt:
                self.accepted_samples += 1
            self._render(
                f"{status_prefix}q={question_id} accepted={accepted_for_question}/{samples_per_question} "
                f"attempts={attempts_used}/{max_attempts_per_question}"
            )

    def record_question_done(
        self,
        *,
        question_id: int,
        accepted_for_question: int,
        samples_per_question: int,
        attempts_used: int,
        status_suffix: str = "",
    ) -> None:
        with self._lock:
            self.completed_questions += 1
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
            f"[{self.total_attempts} attempts] {current_status}"
        )
        print(message, end="", file=sys.stderr, flush=True)


class TraceCollector:
    def __init__(self, sampler: ReasoningTraceSampling) -> None:
        self.sampler = sampler

    def collect_for_item(
        self,
        item: BenchmarkItem,
        *,
        samples_per_question: int,
        max_attempts_per_question: int,
        progress_reporter: ProgressReporter | None = None,
        logger: CollectionLogger | None = None,
    ) -> list[TrajectoryRecord]:
        accepted, _attempt_index = self._collect_until(
            item,
            sample_goal=samples_per_question,
            max_attempts_per_question=max_attempts_per_question,
            accepted=[],
            attempt_index=0,
            progress_reporter=progress_reporter,
            logger=logger,
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
    ) -> list[TrajectoryRecord]:
        if not items:
            return []

        rows: list[TrajectoryRecord] = []
        max_workers = max(1, min(workers, len(items)))
        progress_reporter = None
        logger = CollectionLogger(log_path) if log_path is not None else None
        if show_progress:
            progress_reporter = ProgressReporter(
                total_questions=len(items),
                total_target_samples=len(items) * samples_per_question,
            )
        if logger is not None:
            logger.log(
                "run_started",
                mode="fixed",
                total_questions=len(items),
                samples_per_question=samples_per_question,
                max_attempts_per_question=max_attempts_per_question,
                workers=max_workers,
            )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.collect_for_item,
                    item,
                    samples_per_question=samples_per_question,
                    max_attempts_per_question=max_attempts_per_question,
                    progress_reporter=progress_reporter,
                    logger=logger,
                ): item.question_id
                for item in items
            }
            for future in as_completed(futures):
                question_id = futures[future]
                result_rows = future.result()
                rows.extend(result_rows)
                if progress_reporter is not None:
                    attempts_used = (
                        result_rows[-1].attempt_index + 1
                        if len(result_rows) >= samples_per_question and result_rows
                        else max_attempts_per_question
                    )
                    progress_reporter.record_question_done(
                        question_id=question_id,
                        accepted_for_question=len(result_rows),
                        samples_per_question=samples_per_question,
                        attempts_used=attempts_used,
                    )
                if logger is not None:
                    logger.log(
                        "question_completed",
                        mode="fixed",
                        question_id=question_id,
                        accepted=len(result_rows),
                        target=samples_per_question,
                        attempts_used=attempts_used,
                        right=sum(1 for row in result_rows if row.is_correct),
                        wrong=sum(1 for row in result_rows if not row.is_correct),
                    )
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
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "question_id",
                    "sample_id",
                    "attempt_index",
                    "question",
                    "gold_answer",
                    "predicted_answer",
                    "has_clear_answer",
                    "is_correct",
                    "final_response_text",
                    "reasoning_trace",
                    "model",
                    "prompt_tokens",
                    "completion_tokens",
                    "reasoning_tokens",
                    "finish_reason",
                    "request_id",
                    "error",
                ],
            )
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
            final_response_text=result.final_response_text,
            reasoning_trace=result.reasoning_trace,
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            reasoning_tokens=result.reasoning_tokens,
            finish_reason=result.finish_reason,
            request_id=result.request_id,
            error=result.error,
        )

    def _collect_until(
        self,
        item: BenchmarkItem,
        *,
        sample_goal: int,
        max_attempts_per_question: int,
        accepted: list[TrajectoryRecord],
        attempt_index: int,
        progress_reporter: ProgressReporter | None = None,
        progress_status_prefix: str = "",
        logger: CollectionLogger | None = None,
        log_stage: str = "fixed",
    ) -> tuple[list[TrajectoryRecord], int]:
        while len(accepted) < sample_goal and attempt_index < max_attempts_per_question:
            result = self.sampler.ask_one(item)
            accepted_this_attempt = result.has_clear_answer
            if result.has_clear_answer:
                accepted.append(
                    self._to_trajectory_record(
                        result=result,
                        sample_id=len(accepted),
                        attempt_index=attempt_index,
                    )
                )
            if progress_reporter is not None:
                progress_reporter.record_attempt(
                    question_id=item.question_id,
                    accepted_for_question=len(accepted),
                    samples_per_question=sample_goal,
                    attempts_used=attempt_index + 1,
                    max_attempts_per_question=max_attempts_per_question,
                    accepted_this_attempt=accepted_this_attempt,
                    status_prefix=progress_status_prefix,
                )
            if logger is not None:
                logger.log(
                    "attempt_result",
                    mode="collection",
                    stage=log_stage,
                    question_id=item.question_id,
                    attempt_index=attempt_index,
                    accepted=accepted_this_attempt,
                    accepted_count=len(accepted),
                    sample_goal=sample_goal,
                    predicted_answer=result.predicted_answer,
                    is_correct=result.is_correct,
                    finish_reason=result.finish_reason,
                    request_id=result.request_id,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    reasoning_tokens=result.reasoning_tokens,
                    error=result.error,
                )
            attempt_index += 1
        return accepted, attempt_index


class AdaptiveTraceCollector:
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
        progress_reporter: ProgressReporter | None = None,
        logger: CollectionLogger | None = None,
    ) -> tuple[list[TrajectoryRecord], AdaptiveQuestionSummary]:
        if scout_samples < 1:
            raise ValueError("scout_samples must be at least 1")
        if target_samples < scout_samples:
            raise ValueError("target_samples must be greater than or equal to scout_samples")
        if scout_max_attempts_per_question > max_attempts_per_question:
            raise ValueError("scout_max_attempts_per_question must be <= max_attempts_per_question")

        accepted, attempt_index = self.trace_collector._collect_until(
            item,
            sample_goal=scout_samples,
            max_attempts_per_question=scout_max_attempts_per_question,
            accepted=[],
            attempt_index=0,
            progress_reporter=progress_reporter,
            progress_status_prefix="scout ",
            logger=logger,
            log_stage="scout",
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
                progress_reporter=progress_reporter,
                progress_status_prefix="densify ",
                logger=logger,
                log_stage="densify",
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
    ) -> tuple[list[TrajectoryRecord], list[AdaptiveQuestionSummary]]:
        if not items:
            return [], []

        rows: list[TrajectoryRecord] = []
        summaries: list[AdaptiveQuestionSummary] = []
        max_workers = max(1, min(workers, len(items)))
        progress_reporter = None
        logger = CollectionLogger(log_path) if log_path is not None else None
        if show_progress:
            progress_reporter = ProgressReporter(
                total_questions=len(items),
                total_target_samples=len(items) * target_samples,
            )
        if logger is not None:
            logger.log(
                "run_started",
                mode="adaptive",
                total_questions=len(items),
                scout_samples=scout_samples,
                target_samples=target_samples,
                scout_max_attempts_per_question=scout_max_attempts_per_question,
                max_attempts_per_question=max_attempts_per_question,
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
                    progress_reporter=progress_reporter,
                    logger=logger,
                ): item.question_id
                for item in items
            }
            for future in as_completed(futures):
                question_rows, summary = future.result()
                rows.extend(question_rows)
                summaries.append(summary)
                if progress_reporter is not None:
                    progress_reporter.record_question_done(
                        question_id=summary.question_id,
                        accepted_for_question=summary.accepted_final,
                        samples_per_question=target_samples if summary.was_densified else scout_samples,
                        attempts_used=summary.attempts_used,
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
            writer = csv.DictWriter(
                f,
                fieldnames=[
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
                ],
            )
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
