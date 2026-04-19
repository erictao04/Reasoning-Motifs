from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class QuestionTraceStats:
    question_id: int
    question: str
    gold_answer: str
    total_traces: int
    right_traces: int
    wrong_traces: int
    acceptance_rate: float
    accuracy: float
    has_mixed_outcomes: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class QuestionTraceAnalyzer:
    def load_stats(self, path: Path) -> list[QuestionTraceStats]:
        if not path.exists():
            raise FileNotFoundError(f"Trajectory CSV not found: {path}")

        by_question: dict[int, dict[str, object]] = {}
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question_id = int(row["question_id"])
                entry = by_question.setdefault(
                    question_id,
                    {
                        "question_id": question_id,
                        "question": row["question"],
                        "gold_answer": row["gold_answer"],
                        "total_traces": 0,
                        "right_traces": 0,
                        "wrong_traces": 0,
                    },
                )
                entry["total_traces"] += 1
                if self._is_truthy(row.get("is_correct", "")):
                    entry["right_traces"] += 1
                else:
                    entry["wrong_traces"] += 1

        rows: list[QuestionTraceStats] = []
        for question_id in sorted(by_question):
            entry = by_question[question_id]
            total_traces = int(entry["total_traces"])
            right_traces = int(entry["right_traces"])
            wrong_traces = int(entry["wrong_traces"])
            rows.append(
                QuestionTraceStats(
                    question_id=question_id,
                    question=str(entry["question"]),
                    gold_answer=str(entry["gold_answer"]),
                    total_traces=total_traces,
                    right_traces=right_traces,
                    wrong_traces=wrong_traces,
                    acceptance_rate=1.0,
                    accuracy=(right_traces / total_traces) if total_traces else 0.0,
                    has_mixed_outcomes=right_traces > 0 and wrong_traces > 0,
                )
            )
        return rows

    def filter_stats(
        self,
        rows: Sequence[QuestionTraceStats],
        *,
        only_mixed: bool,
        min_total: int,
        min_right: int,
        min_wrong: int,
    ) -> list[QuestionTraceStats]:
        filtered: list[QuestionTraceStats] = []
        for row in rows:
            if only_mixed and not row.has_mixed_outcomes:
                continue
            if row.total_traces < min_total:
                continue
            if row.right_traces < min_right:
                continue
            if row.wrong_traces < min_wrong:
                continue
            filtered.append(row)
        return filtered

    def save_csv(self, path: Path, rows: Sequence[QuestionTraceStats]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "question_id",
                    "question",
                    "gold_answer",
                    "total_traces",
                    "right_traces",
                    "wrong_traces",
                    "acceptance_rate",
                    "accuracy",
                    "has_mixed_outcomes",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row.to_dict())

    @staticmethod
    def _is_truthy(value: str) -> bool:
        return value.strip().lower() in {"true", "1", "yes"}
