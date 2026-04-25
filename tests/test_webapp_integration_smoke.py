from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from reasoning_motifs_web.exporter import export_webapp_data
from webapp_api.app import create_app


class WebappIntegrationSmokeTests(unittest.TestCase):
    def test_export_and_api_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "artifacts"
            export_webapp_data(output_dir)

            client = TestClient(create_app(output_dir))
            overview = client.get("/api/overview")
            questions = client.get("/api/questions")

            self.assertEqual(overview.status_code, 200)
            self.assertEqual(questions.status_code, 200)
            self.assertGreater(len(questions.json()), 0)

            first_question_id = questions.json()[0]["question_id"]
            detail = client.get(f"/api/questions/{first_question_id}")
            self.assertEqual(detail.status_code, 200)

            written_questions = json.loads((output_dir / "questions.json").read_text(encoding="utf-8"))
            self.assertEqual(len(written_questions), len(questions.json()))


if __name__ == "__main__":
    unittest.main()
