from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from reasoning_trace_sampling.data_classes import BenchmarkItem, ReasoningResult
from reasoning_trace_sampling.question_stats import QuestionTraceAnalyzer
from reasoning_trace_sampling.sampling import ReasoningTraceSampling
from reasoning_trace_sampling.trajectory_collection import AdaptiveTraceCollector, TraceCollector


def make_result(
    *,
    item: BenchmarkItem,
    predicted_answer: str,
    has_clear_answer: bool,
    is_correct: bool,
    attempt_index: int,
) -> ReasoningResult:
    return ReasoningResult(
        question_id=item.question_id,
        question=item.question,
        gold_answer=item.gold_answer,
        predicted_answer=predicted_answer,
        has_clear_answer=has_clear_answer,
        is_correct=is_correct,
        answer_source="parser" if has_clear_answer else "",
        answer_validation="",
        final_response_text=f"FINAL: {predicted_answer}" if has_clear_answer else "",
        reasoning_trace=f"attempt {attempt_index}",
        temperature=0.2,
        model="fake-model",
        prompt_tokens=10,
        completion_tokens=20,
        reasoning_tokens=0,
        finish_reason="stop",
        request_id=f"req-{item.question_id}-{attempt_index}",
        error=None,
    )


class FakeSampler(ReasoningTraceSampling):
    def __init__(self, scripted_results: dict[int, list[ReasoningResult]]) -> None:
        self.scripted_results = {key: list(value) for key, value in scripted_results.items()}
        self._lock = threading.Lock()

    def ask_one(
        self,
        item: BenchmarkItem,
        *,
        temperature: float | None = None,
        progress_callback=None,
        attempt_index: int | None = None,
    ) -> ReasoningResult:
        del temperature, progress_callback, attempt_index
        with self._lock:
            queue = self.scripted_results[item.question_id]
            if not queue:
                raise AssertionError(f"No scripted results left for question {item.question_id}")
            return queue.pop(0)


class TrajectorySamplingSmokeTest(unittest.TestCase):
    def test_answer_extraction_smoke(self) -> None:
        final_answer, final_ok = ReasoningTraceSampling._extract_final_answer(
            "Work\nFINAL: 42",
            answer_format="final",
        )
        hash_answer, hash_ok = ReasoningTraceSampling._extract_final_answer(
            "Work\n#### 17",
            answer_format="hash",
        )
        boxed_answer, boxed_ok = ReasoningTraceSampling._extract_final_answer(
            "Work\n\\boxed{\\frac{3}{4}}",
            answer_format="boxed",
        )

        self.assertEqual((final_answer, final_ok), ("42", True))
        self.assertEqual((hash_answer, hash_ok), ("17", True))
        self.assertEqual((boxed_answer, boxed_ok), ("\\frac{3}{4}", True))

    def test_trace_collector_and_question_stats_smoke(self) -> None:
        item = BenchmarkItem(question_id=7, question="What is 2 + 2?", gold_answer="4")
        scripted = [
            make_result(
                item=item,
                predicted_answer="",
                has_clear_answer=False,
                is_correct=False,
                attempt_index=0,
            ),
            make_result(
                item=item,
                predicted_answer="4",
                has_clear_answer=True,
                is_correct=True,
                attempt_index=1,
            ),
            make_result(
                item=item,
                predicted_answer="5",
                has_clear_answer=True,
                is_correct=False,
                attempt_index=2,
            ),
        ]

        collector = TraceCollector(FakeSampler({item.question_id: scripted}))
        rows = collector.collect_for_item(
            item,
            samples_per_question=2,
            max_attempts_per_question=4,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual([row.sample_id for row in rows], [0, 1])
        self.assertEqual([row.attempt_index for row in rows], [1, 2])
        self.assertEqual([row.is_correct for row in rows], [True, False])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "trajectories.csv"
            TraceCollector.save_csv(output_path, rows)

            analyzer = QuestionTraceAnalyzer()
            stats = analyzer.load_stats(output_path)

        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0].total_traces, 2)
        self.assertEqual(stats[0].right_traces, 1)
        self.assertEqual(stats[0].wrong_traces, 1)
        self.assertTrue(stats[0].has_mixed_outcomes)

    def test_adaptive_trace_collector_densifies_mixed_question(self) -> None:
        item = BenchmarkItem(question_id=9, question="What is 3 + 4?", gold_answer="7")
        scripted = [
            make_result(
                item=item,
                predicted_answer="7",
                has_clear_answer=True,
                is_correct=True,
                attempt_index=0,
            ),
            make_result(
                item=item,
                predicted_answer="8",
                has_clear_answer=True,
                is_correct=False,
                attempt_index=1,
            ),
            make_result(
                item=item,
                predicted_answer="7",
                has_clear_answer=True,
                is_correct=True,
                attempt_index=2,
            ),
        ]

        collector = TraceCollector(FakeSampler({item.question_id: scripted}))
        adaptive = AdaptiveTraceCollector(collector)

        rows, summary = adaptive.collect_for_item(
            item,
            scout_samples=2,
            target_samples=3,
            scout_max_attempts_per_question=3,
            max_attempts_per_question=4,
        )

        self.assertEqual(len(rows), 3)
        self.assertTrue(summary.was_densified)
        self.assertEqual(summary.decision, "mixed_densified")
        self.assertEqual(summary.accepted_scout, 2)
        self.assertEqual(summary.accepted_final, 3)
        self.assertEqual(summary.right_final, 2)
        self.assertEqual(summary.wrong_final, 1)


if __name__ == "__main__":
    unittest.main()
