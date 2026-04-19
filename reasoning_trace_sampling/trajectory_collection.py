from __future__ import annotations

import csv
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

from .data_classes import BenchmarkItem, ReasoningResult, TrajectoryRecord
from .sampling import ReasoningTraceSampling


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
    ) -> None:
        with self._lock:
            self.total_attempts += 1
            if accepted_this_attempt:
                self.accepted_samples += 1
            self._render(
                f"q={question_id} accepted={accepted_for_question}/{samples_per_question} "
                f"attempts={attempts_used}/{max_attempts_per_question}"
            )

    def record_question_done(
        self,
        *,
        question_id: int,
        accepted_for_question: int,
        samples_per_question: int,
        attempts_used: int,
    ) -> None:
        with self._lock:
            self.completed_questions += 1
            self._render(
                f"finished q={question_id} accepted={accepted_for_question}/{samples_per_question} "
                f"attempts={attempts_used}"
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
    ) -> list[TrajectoryRecord]:
        accepted: list[TrajectoryRecord] = []
        attempt_index = 0

        while len(accepted) < samples_per_question and attempt_index < max_attempts_per_question:
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
                    samples_per_question=samples_per_question,
                    attempts_used=attempt_index + 1,
                    max_attempts_per_question=max_attempts_per_question,
                    accepted_this_attempt=accepted_this_attempt,
                )
            attempt_index += 1

        return accepted

    def collect_many(
        self,
        items: Sequence[BenchmarkItem],
        *,
        samples_per_question: int,
        max_attempts_per_question: int,
        workers: int,
        show_progress: bool = False,
    ) -> list[TrajectoryRecord]:
        if not items:
            return []

        rows: list[TrajectoryRecord] = []
        max_workers = max(1, min(workers, len(items)))
        progress_reporter = None
        if show_progress:
            progress_reporter = ProgressReporter(
                total_questions=len(items),
                total_target_samples=len(items) * samples_per_question,
            )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.collect_for_item,
                    item,
                    samples_per_question=samples_per_question,
                    max_attempts_per_question=max_attempts_per_question,
                    progress_reporter=progress_reporter,
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
        if progress_reporter is not None:
            progress_reporter.finish()

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
