from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from .data_classes import BenchmarkItem, BenchmarkPreset


class BenchmarkRegistry:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self._presets = {
            "sample_math": BenchmarkPreset(
                name="sample_math",
                path=repo_root / "data" / "benchmarks" / "sample_math.csv",
                description="Tiny local toy benchmark for quick smoke tests.",
                answer_format="final",
            ),
            "math_500": BenchmarkPreset(
                name="math_500",
                path=repo_root / "data" / "benchmarks" / "math_500.json",
                description="Recommended real math reasoning benchmark. Place a local copy at this path.",
                answer_format="boxed",
            ),
            "gsm8k": BenchmarkPreset(
                name="gsm8k",
                path=repo_root / "data" / "benchmarks" / "gsm8k_test.json",
                description="Grade-school math word problems. Good first real benchmark.",
                answer_format="hash",
            ),
            "aime_2024": BenchmarkPreset(
                name="aime_2024",
                path=repo_root / "data" / "benchmarks" / "aime_2024.json",
                description="Hard olympiad-style math. Good stress test for long reasoning traces.",
                answer_format="boxed",
            ),
        }

    def list_presets(self) -> list[BenchmarkPreset]:
        return [self._presets[name] for name in sorted(self._presets)]

    def resolve(self, *, benchmark_name: str | None, benchmark_path: Path | None) -> Path:
        if benchmark_path is not None:
            return benchmark_path.resolve()
        if benchmark_name is None:
            benchmark_name = "sample_math"
        preset = self._presets.get(benchmark_name)
        if preset is None:
            known = ", ".join(sorted(self._presets))
            raise ValueError(f"Unknown benchmark '{benchmark_name}'. Known benchmarks: {known}")
        return preset.path.resolve()

    def resolve_answer_format(self, *, benchmark_name: str | None, benchmark_path: Path | None) -> str:
        if benchmark_path is not None:
            return self._infer_answer_format_from_path(benchmark_path)
        if benchmark_name is None:
            benchmark_name = "sample_math"
        preset = self._presets.get(benchmark_name)
        if preset is None:
            return "final"
        return preset.answer_format

    def load_items(self, path: Path, *, answer_format: str) -> list[BenchmarkItem]:
        if not path.exists():
            raise FileNotFoundError(
                f"Benchmark file not found: {path}. "
                "Use --benchmark-path to point at a local dataset file."
            )

        if path.suffix.lower() == ".csv":
            with path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        elif path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("JSON benchmark must be a list of question/answer objects.")
            rows = raw
        else:
            raise ValueError(f"Unsupported benchmark format: {path.suffix}")

        items: list[BenchmarkItem] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"Benchmark row {index} is not an object.")
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not question or not answer:
                raise ValueError(
                    f"Benchmark row {index} must have non-empty 'question' and 'answer' fields."
                )
            items.append(
                BenchmarkItem(
                    question_id=int(row.get("question_id", index)),
                    question=question,
                    gold_answer=answer,
                    answer_format=answer_format,
                    level=self._parse_optional_int(row.get("level")),
                    subject=self._parse_optional_str(row.get("subject")),
                    source_id=self._parse_optional_str(row.get("source_id")),
                )
            )
        return items

    @staticmethod
    def filter_items(
        items: list[BenchmarkItem],
        *,
        min_level: int | None = None,
        max_level: int | None = None,
        sample_size: int | None = None,
        sample_seed: int = 17,
    ) -> list[BenchmarkItem]:
        filtered = items
        if min_level is not None:
            filtered = [item for item in filtered if item.level is not None and item.level >= min_level]
        if max_level is not None:
            filtered = [item for item in filtered if item.level is not None and item.level <= max_level]
        if sample_size is not None:
            if sample_size < 1:
                raise ValueError("sample_size must be at least 1")
            if sample_size > len(filtered):
                raise ValueError(
                    f"Requested sample_size={sample_size}, but only {len(filtered)} questions remain after filtering."
                )
            rng = random.Random(sample_seed)
            filtered = sorted(rng.sample(filtered, sample_size), key=lambda item: item.question_id)
        return filtered

    @staticmethod
    def _infer_answer_format_from_path(path: Path) -> str:
        name = path.name.lower()
        if "gsm8k" in name:
            return "hash"
        if "math" in name or "aime" in name:
            return "boxed"
        return "final"

    @staticmethod
    def _parse_optional_int(value: object) -> int | None:
        if value in {None, ""}:
            return None
        return int(value)

    @staticmethod
    def _parse_optional_str(value: object) -> str | None:
        if value in {None, ""}:
            return None
        return str(value)
