from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from reasoning_motifs_web.exporter import export_webapp_data


TOKENIZED_FIELDNAMES = [
    "question_id",
    "sample_id",
    "attempt_index",
    "question",
    "gold_answer",
    "predicted_answer",
    "has_clear_answer",
    "is_correct",
    "reasoning_trace",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "finish_reason",
    "request_id",
    "error",
    "pilot_question_uid",
    "benchmark_name",
    "source_run",
    "source_file",
    "source_question_id",
    "pilot_question_index",
]

RAW_FIELDNAMES = [
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
    "pilot_question_uid",
    "benchmark_name",
    "source_run",
    "source_file",
    "source_question_id",
    "pilot_question_index",
]


class WebappExportTests(unittest.TestCase):
    def _write_rows(self, path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_export_generates_question_payloads_and_low_evidence_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            tokenized_csv = tmp / "tokenized.csv"
            raw_csv = tmp / "raw.csv"
            outdir = tmp / "artifacts"

            tokenized_rows = [
                {
                    "question_id": "1",
                    "sample_id": str(idx),
                    "attempt_index": str(idx),
                    "question": "Question 1",
                    "gold_answer": "42",
                    "predicted_answer": "42" if idx < 3 else "17",
                    "has_clear_answer": "True",
                    "is_correct": "True" if idx < 3 else "False",
                    "reasoning_trace": (
                        "analyze|compute|conclude"
                        if idx < 3
                        else "analyze|backtrack|compute"
                    ),
                    "model": "demo",
                    "prompt_tokens": "0",
                    "completion_tokens": "0",
                    "reasoning_tokens": "0",
                    "finish_reason": "stop",
                    "request_id": f"r-{idx}",
                    "error": "",
                    "pilot_question_uid": "demo:q1",
                    "benchmark_name": "demo",
                    "source_run": "demo",
                    "source_file": "demo.csv",
                    "source_question_id": "1",
                    "pilot_question_index": "0",
                }
                for idx in range(6)
            ] + [
                {
                    "question_id": "2",
                    "sample_id": str(idx),
                    "attempt_index": str(idx),
                    "question": "Question 2",
                    "gold_answer": "9",
                    "predicted_answer": "9",
                    "has_clear_answer": "True",
                    "is_correct": "True",
                    "reasoning_trace": "analyze|compute|simplify|conclude",
                    "model": "demo",
                    "prompt_tokens": "0",
                    "completion_tokens": "0",
                    "reasoning_tokens": "0",
                    "finish_reason": "stop",
                    "request_id": f"s-{idx}",
                    "error": "",
                    "pilot_question_uid": "demo:q2",
                    "benchmark_name": "demo",
                    "source_run": "demo",
                    "source_file": "demo.csv",
                    "source_question_id": "2",
                    "pilot_question_index": "1",
                }
                for idx in range(2)
            ]
            raw_rows = []
            for row in tokenized_rows:
                raw_row = dict(row)
                raw_row["final_response_text"] = "final"
                raw_row["reasoning_trace"] = f"raw trace {row['question_id']}:{row['sample_id']}"
                raw_rows.append(raw_row)

            self._write_rows(tokenized_csv, TOKENIZED_FIELDNAMES, tokenized_rows)
            self._write_rows(raw_csv, RAW_FIELDNAMES, raw_rows)

            export_webapp_data(outdir, tokenized_csv=tokenized_csv, raw_csv=raw_csv)

            questions = json.loads((outdir / "questions.json").read_text(encoding="utf-8"))
            self.assertEqual(len(questions), 2)
            self.assertEqual(questions[0]["question_id"], "1")
            self.assertIn("mixed_outcomes", questions[0]["tags"])

            q1 = json.loads((outdir / "question" / "1.json").read_text(encoding="utf-8"))
            self.assertTrue(q1["local_motifs"]["available"])
            self.assertGreater(len(q1["local_motifs"]["success"]), 0)
            self.assertEqual(len(q1["representative_traces"]["success"]), 3)
            self.assertGreater(
                len(q1["representative_traces"]["success"][0]["matched_success_motifs"]),
                0,
            )

            q2 = json.loads((outdir / "question" / "2.json").read_text(encoding="utf-8"))
            self.assertFalse(q2["local_motifs"]["available"])
            self.assertIn("low_evidence", q2["tags"])

    def test_export_accepts_minimal_clean_expanded_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            tokenized_csv = tmp / "tokenized.csv"
            raw_csv = tmp / "raw.csv"
            outdir = tmp / "artifacts"

            with tokenized_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "question_id",
                        "sample_id",
                        "gold_answer",
                        "predicted_answer",
                        "is_correct",
                        "tokenized_trace",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "question_id": "45",
                        "sample_id": "0",
                        "gold_answer": "2",
                        "predicted_answer": "2",
                        "is_correct": "True",
                        "tokenized_trace": "analyze compute conclude",
                    }
                )

            raw_rows = [
                {
                    "question_id": "45",
                    "sample_id": "0",
                    "attempt_index": "0",
                    "question": "Expanded question",
                    "gold_answer": "2",
                    "predicted_answer": "2",
                    "has_clear_answer": "True",
                    "is_correct": "True",
                    "final_response_text": "final",
                    "reasoning_trace": "raw expanded trace",
                    "model": "demo",
                    "prompt_tokens": "0",
                    "completion_tokens": "0",
                    "reasoning_tokens": "0",
                    "finish_reason": "stop",
                    "request_id": "r-1",
                    "error": "",
                    "pilot_question_uid": "",
                    "benchmark_name": "expanded",
                    "source_run": "demo",
                    "source_file": "demo.csv",
                    "source_question_id": "45",
                    "pilot_question_index": "0",
                }
            ]
            self._write_rows(raw_csv, RAW_FIELDNAMES, raw_rows)

            export_webapp_data(outdir, tokenized_csv=tokenized_csv, raw_csv=raw_csv, global_motifs_csv=None)
            question = json.loads((outdir / "question" / "45.json").read_text(encoding="utf-8"))
            self.assertEqual(question["question_text"], "Expanded question")
            self.assertEqual(question["all_traces"][0]["tokenized_trace"], "analyze compute conclude")
            self.assertEqual(question["all_traces"][0]["attempt_index"], "0")


if __name__ == "__main__":
    unittest.main()
