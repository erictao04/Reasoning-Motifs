from __future__ import annotations

import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

from .api_shim import APIShim
from .data_classes import BenchmarkItem, ReasoningResult


class ReasoningTraceSampling:
    def __init__(self, api_shim: APIShim) -> None:
        self.api_shim = api_shim

    def ask_one(self, item: BenchmarkItem) -> ReasoningResult:
        try:
            payload = self.api_shim.ask_question(self._build_prompt(item))
            return self._parse_response(payload, item)
        except Exception as exc:
            return ReasoningResult(
                question_id=item.question_id,
                question=item.question,
                gold_answer=item.gold_answer,
                predicted_answer="",
                has_clear_answer=False,
                is_correct=False,
                final_response_text="",
                reasoning_trace="",
                model=self.api_shim.config.model,
                prompt_tokens=None,
                completion_tokens=None,
                reasoning_tokens=None,
                finish_reason="error",
                request_id=None,
                error=str(exc),
            )

    def ask_many(self, items: Sequence[BenchmarkItem], *, workers: int) -> list[ReasoningResult]:
        if not items:
            return []

        max_workers = max(1, min(workers, len(items)))
        rows: list[ReasoningResult] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.ask_one, item): item.question_id for item in items}
            for future in as_completed(futures):
                rows.append(future.result())

        rows.sort(key=lambda row: row.question_id)
        return rows

    @staticmethod
    def save_csv(path: Path, rows: Sequence[ReasoningResult]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "question_id",
                    "question",
                    "gold_answer",
                    "predicted_answer",
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

    def _parse_response(self, payload: dict[str, object], item: BenchmarkItem) -> ReasoningResult:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Missing choices in Together response: {payload}")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError(f"Malformed choice in Together response: {first_choice}")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError(f"Missing message in Together response: {first_choice}")

        final_response_text = self._normalize_text(message.get("content")).strip()
        reasoning_trace = self._normalize_text(message.get("reasoning")).strip()
        if not reasoning_trace:
            reasoning_trace = self._normalize_text(message.get("reasoning_content")).strip()
        if not reasoning_trace:
            reasoning_trace = self._strip_answer_line(
                final_response_text,
                answer_format=item.answer_format,
            )

        predicted_answer, has_clear_answer = self._extract_final_answer(
            final_response_text,
            answer_format=item.answer_format,
        )
        prompt_tokens, completion_tokens, reasoning_tokens = self._extract_usage(payload.get("usage"))

        return ReasoningResult(
            question_id=item.question_id,
            question=item.question,
            gold_answer=item.gold_answer,
            predicted_answer=predicted_answer,
            has_clear_answer=has_clear_answer,
            is_correct=self._answers_match(
                predicted_answer,
                item.gold_answer,
                answer_format=item.answer_format,
            ),
            final_response_text=final_response_text,
            reasoning_trace=reasoning_trace,
            model=self._maybe_string(payload.get("model")) or self.api_shim.config.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            finish_reason=self._maybe_string(first_choice.get("finish_reason")) or "unknown",
            request_id=self._maybe_string(payload.get("id")),
            error=None,
        )

    @staticmethod
    def _extract_final_answer(text: str, *, answer_format: str) -> tuple[str, bool]:
        if not text.strip():
            return "", False

        extractors = {
            "final": [
                ReasoningTraceSampling._extract_final_marker,
                ReasoningTraceSampling._extract_hash_marker,
                ReasoningTraceSampling._extract_boxed_marker,
            ],
            "hash": [
                ReasoningTraceSampling._extract_hash_marker,
                ReasoningTraceSampling._extract_final_marker,
                ReasoningTraceSampling._extract_boxed_marker,
            ],
            "boxed": [
                ReasoningTraceSampling._extract_boxed_marker,
                ReasoningTraceSampling._extract_final_marker,
                ReasoningTraceSampling._extract_hash_marker,
            ],
        }

        for extractor in extractors.get(answer_format, extractors["final"]):
            answer = extractor(text)
            if answer:
                return answer, True
        return "", False

    @staticmethod
    def _build_prompt(item: BenchmarkItem) -> str:
        if item.answer_format == "hash":
            suffix = "Show your reasoning, then end with a final line exactly in the form: #### <answer>"
        elif item.answer_format == "boxed":
            suffix = "Show your reasoning, then end with the final answer on its own line in the form: \\boxed{answer}"
        else:
            suffix = "Show your reasoning, then end with a final line exactly in the form: FINAL: <answer>"
        return f"{item.question}\n\n{suffix}"

    @staticmethod
    def _extract_final_marker(text: str) -> str:
        for line in reversed(text.splitlines()):
            match = re.match(r"\s*FINAL\s*:\s*(.+?)\s*$", line, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    @staticmethod
    def _extract_hash_marker(text: str) -> str:
        for line in reversed(text.splitlines()):
            match = re.match(r"\s*####\s*(.+?)\s*$", line)
            if match:
                return match.group(1).strip()
        return ""

    @staticmethod
    def _extract_boxed_marker(text: str) -> str:
        marker = "\\boxed{"
        start = text.rfind(marker)
        if start == -1:
            return ""

        depth = 0
        content: list[str] = []
        for char in text[start + len(marker) :]:
            if char == "{":
                depth += 1
                content.append(char)
                continue
            if char == "}":
                if depth == 0:
                    return "".join(content).strip()
                depth -= 1
                content.append(char)
                continue
            content.append(char)
        return ""

    @staticmethod
    def _strip_answer_line(text: str, *, answer_format: str) -> str:
        lines = [line for line in text.splitlines()]
        if not lines:
            return ""

        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            return ""

        last = lines[-1].strip()
        if answer_format == "hash" and last.startswith("####"):
            lines.pop()
        elif answer_format == "final" and re.match(r"FINAL\s*:", last, flags=re.IGNORECASE):
            lines.pop()
        elif answer_format == "boxed" and "\\boxed{" in last:
            lines.pop()

        return "\n".join(lines).strip()

    @staticmethod
    def _answers_match(predicted_answer: str, gold_answer: str, *, answer_format: str) -> bool:
        predicted = ReasoningTraceSampling._normalize_answer(predicted_answer, answer_format=answer_format)
        gold = ReasoningTraceSampling._normalize_answer(gold_answer, answer_format=answer_format)
        return bool(predicted) and predicted == gold

    @staticmethod
    def _normalize_answer(answer: str, *, answer_format: str) -> str:
        text = answer.strip()
        if not text:
            return ""

        # Common LaTeX wrappers in math benchmarks.
        text = text.replace("\\left", "").replace("\\right", "")
        text = text.replace("$", "")
        text = re.sub(r"\s+", "", text)

        if answer_format == "hash":
            text = text.replace(",", "")

        return text

    @staticmethod
    def _normalize_text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            chunks: list[str] = []
            for item in value:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return "".join(chunks)
        return str(value)

    @staticmethod
    def _extract_usage(usage: object) -> tuple[int | None, int | None, int | None]:
        if not isinstance(usage, dict):
            return None, None, None

        prompt_tokens = ReasoningTraceSampling._maybe_int(
            usage.get("prompt_tokens") or usage.get("input_tokens")
        )
        completion_tokens = ReasoningTraceSampling._maybe_int(
            usage.get("completion_tokens") or usage.get("output_tokens")
        )
        reasoning_tokens = ReasoningTraceSampling._maybe_int(usage.get("reasoning_tokens"))

        details = usage.get("completion_tokens_details")
        if reasoning_tokens is None and isinstance(details, dict):
            reasoning_tokens = ReasoningTraceSampling._maybe_int(details.get("reasoning_tokens"))

        return prompt_tokens, completion_tokens, reasoning_tokens

    @staticmethod
    def _maybe_int(value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _maybe_string(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return str(value)
