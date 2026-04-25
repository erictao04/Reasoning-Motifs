from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from reasoning_motifs_web.models import CorpusOverview, MotifBucket, QuestionDetail, QuestionSummary, StorySection
from webapp_api.app import create_app


class WebappApiTests(unittest.TestCase):
    def _write_fixture_artifacts(self, root: Path) -> None:
        question_dir = root / "question"
        question_dir.mkdir(parents=True, exist_ok=True)

        overview = CorpusOverview(
            corpus_label="Demo",
            num_questions=1,
            num_traces=2,
            num_success=1,
            num_failure=1,
            success_rate=0.5,
            avg_token_count=4.0,
            median_token_count=4.0,
            featured_question_ids=["7"],
            story_sections=[StorySection(id="why", title="Why", body="Because.")],
            top_success_motifs=[],
            top_failure_motifs=[],
        )
        questions = [
            QuestionSummary(
                question_id="7",
                question_text="Question 7",
                gold_answer="12",
                benchmark_name="demo",
                total_traces=2,
                success_count=1,
                failure_count=1,
                success_rate=0.5,
                avg_token_count=4.0,
                median_token_count=4.0,
                distinct_predicted_answers=2,
                local_motif_separation=0.5,
                top_success_motif="analyze|compute",
                top_failure_motif="backtrack|compute",
                tags=["mixed_outcomes"],
            )
        ]
        detail = QuestionDetail(
            question_id="7",
            question_text="Question 7",
            gold_answer="12",
            benchmark_name="demo",
            pilot_question_uid="demo:q7",
            total_traces=2,
            success_count=1,
            failure_count=1,
            success_rate=0.5,
            avg_token_count=4.0,
            median_token_count=4.0,
            distinct_predicted_answers=2,
            tags=["mixed_outcomes"],
            insights=["Mixed outcomes."],
            answer_distribution=[],
            local_motifs=MotifBucket(available=False, reason="Not enough data."),
            global_motifs=MotifBucket(available=False, reason="None."),
            representative_traces={"success": [], "failure": []},
            all_traces=[],
        )

        (root / "overview.json").write_text(overview.model_dump_json(indent=2), encoding="utf-8")
        (root / "questions.json").write_text(
            json.dumps([item.model_dump(mode="json") for item in questions], indent=2),
            encoding="utf-8",
        )
        (question_dir / "7.json").write_text(detail.model_dump_json(indent=2), encoding="utf-8")

    def test_endpoints_return_expected_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_fixture_artifacts(root)
            client = TestClient(create_app(root))

            overview = client.get("/api/overview")
            self.assertEqual(overview.status_code, 200)
            self.assertEqual(overview.json()["corpus_label"], "Demo")

            questions = client.get("/api/questions")
            self.assertEqual(questions.status_code, 200)
            self.assertEqual(questions.json()[0]["question_id"], "7")

            question = client.get("/api/questions/7")
            self.assertEqual(question.status_code, 200)
            self.assertEqual(question.json()["pilot_question_uid"], "demo:q7")

            missing = client.get("/api/questions/404")
            self.assertEqual(missing.status_code, 404)

    def test_bad_artifact_layout_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(RuntimeError):
                create_app(Path(tmpdir))


if __name__ == "__main__":
    unittest.main()
