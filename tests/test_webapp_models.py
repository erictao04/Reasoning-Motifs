from __future__ import annotations

import unittest

from reasoning_motifs_web.models import CorpusOverview, MotifBucket, MotifRow, QuestionDetail, StorySection, TraceSummary


class WebappModelTests(unittest.TestCase):
    def test_question_detail_round_trip(self) -> None:
        motif = MotifRow(
            motif="analyze|compute",
            tokens=["analyze", "compute"],
            length=2,
            scope="question_local",
            direction="success",
            success_count=3,
            failure_count=1,
            success_support=0.75,
            failure_support=0.25,
            support_difference=0.5,
            lift=3.0,
            log_odds_ratio=1.2,
            q_value=0.04,
        )
        trace = TraceSummary(
            trace_id="11:2",
            question_id="11",
            sample_id="2",
            attempt_index="3",
            predicted_answer="294",
            is_correct=True,
            tokenized_trace="analyze | compute",
            tokens=["analyze", "compute"],
            token_count=2,
            reasoning_trace="We count paths.",
            matched_success_motifs=["analyze|compute"],
            matched_failure_motifs=[],
        )
        detail = QuestionDetail(
            question_id="11",
            question_text="Example question",
            gold_answer="294",
            benchmark_name="aime_2024",
            pilot_question_uid="aime_2024:q11",
            total_traces=10,
            success_count=8,
            failure_count=2,
            success_rate=0.8,
            avg_token_count=7.5,
            median_token_count=7.0,
            distinct_predicted_answers=2,
            tags=["mixed_outcomes"],
            insights=["Success traces are shorter."],
            answer_distribution=[],
            local_motifs=MotifBucket(available=True, success=[motif], failure=[]),
            global_motifs=MotifBucket(available=True, success=[motif], failure=[]),
            representative_traces={"success": [trace], "failure": []},
            all_traces=[trace],
        )

        payload = detail.model_dump()
        self.assertEqual(payload["question_id"], "11")
        self.assertEqual(payload["local_motifs"]["success"][0]["motif"], "analyze|compute")
        self.assertEqual(payload["representative_traces"]["success"][0]["trace_id"], "11:2")

    def test_overview_requires_story_sections(self) -> None:
        overview = CorpusOverview(
            corpus_label="Pilot",
            num_questions=22,
            num_traces=251,
            num_success=180,
            num_failure=71,
            success_rate=180 / 251,
            avg_token_count=12.2,
            median_token_count=11.0,
            featured_question_ids=["11"],
            story_sections=[StorySection(id="why", title="Why motifs", body="Motifs expose structure.")],
            top_success_motifs=[],
            top_failure_motifs=[],
        )
        self.assertEqual(overview.story_sections[0].id, "why")


if __name__ == "__main__":
    unittest.main()
