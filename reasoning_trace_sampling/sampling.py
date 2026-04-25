"""Single-question sampling and answer parsing helpers.

This module is responsible for one model attempt at a time. It builds the
prompt for a benchmark item, parses the returned response into structured
fields, and optionally falls back to an LLM verifier when deterministic answer
extraction fails.
"""

from __future__ import annotations

import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .api_shim import APIShim, TogetherFatalRequestError
from .data_classes import BenchmarkItem, ReasoningResult


@dataclass(frozen=True)
class AnswerVerifierResult:
    extracted_answer: str
    is_correct: bool


@dataclass(frozen=True)
class AnswerVerifierAttempt:
    extracted_answer: str = ""
    is_correct: bool | None = None
    error: str | None = None


ProgressCallback = Callable[[str, dict[str, object]], None]


class LLMAnswerVerifier:
    """Fallback verifier used when deterministic answer parsing does not succeed."""

    def __init__(self, api_shim: APIShim) -> None:
        self.api_shim = api_shim

    def verify(
        self,
        *,
        response_text: str,
        gold_answer: str,
        answer_format: str,
    ) -> AnswerVerifierResult:
        prompt = self._build_prompt(
            response_text=response_text,
            gold_answer=gold_answer,
            answer_format=answer_format,
        )
        payload = self.api_shim.ask_question(prompt)
        content = self._extract_content(payload)
        raw = self._extract_json_object(content)
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError("Verifier returned JSON that is not an object.")
        is_correct = self._maybe_bool(parsed.get("is_correct"))
        if is_correct is None:
            # Backward compatibility for verifier prompts from older local runs.
            is_correct = self._maybe_bool(parsed.get("matches_gold"))
        if is_correct is None:
            raise RuntimeError("Verifier JSON must include boolean is_correct.")
        return AnswerVerifierResult(
            extracted_answer=str(parsed.get("extracted_answer", "")).strip(),
            is_correct=is_correct,
        )

    @staticmethod
    def _build_prompt(*, response_text: str, gold_answer: str, answer_format: str) -> str:
        verifier_text = LLMAnswerVerifier._trim_response_for_verifier(response_text)
        return (
            "Classify the model response below as RIGHT or WRONG against the gold answer. "
            "You must choose one. Do not return no-answer, unknown, unclear, or null. "
            "Do not solve the problem from scratch; judge the response's own final answer or conclusion. "
            "If the response does not explicitly give a final answer or conclusion, classify it as WRONG.\n\n"
            "Return strict JSON only with these keys:\n"
            '{"extracted_answer": string, "is_correct": boolean}\n\n'
            f"Answer format: {answer_format}\n"
            f"Gold answer: {gold_answer}\n\n"
            "Model response:\n"
            f"{verifier_text}"
        )

    @staticmethod
    def _trim_response_for_verifier(response_text: str, *, max_chars: int = 12000) -> str:
        if len(response_text) <= max_chars:
            return response_text
        return "[earlier response omitted]\n" + response_text[-max_chars:]

    @staticmethod
    def _extract_content(payload: dict[str, object]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Verifier response is missing choices.")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("Verifier response has a malformed choice.")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Verifier response is missing message.")
        content = message.get("content")
        if not isinstance(content, str):
            return str(content or "")
        return content

    @staticmethod
    def _extract_json_object(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise RuntimeError(f"Verifier returned non-JSON output: {text[:500]}")
        return text[start : end + 1]

    @staticmethod
    def _maybe_bool(value: object) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes"}:
                return True
            if lowered in {"false", "no"}:
                return False
        return None


class ReasoningTraceSampling:
    """Turn one benchmark item into one scored reasoning result."""

    def __init__(self, api_shim: APIShim, *, answer_verifier: LLMAnswerVerifier | None = None) -> None:
        self.api_shim = api_shim
        self.answer_verifier = answer_verifier

    def ask_one(
        self,
        item: BenchmarkItem,
        *,
        temperature: float | None = None,
        progress_callback: ProgressCallback | None = None,
        attempt_index: int | None = None,
    ) -> ReasoningResult:
        effective_temperature = self.api_shim.config.temperature if temperature is None else temperature
        try:
            payload = self.api_shim.ask_question(
                self._build_prompt(item),
                temperature=effective_temperature,
            )
            return self._parse_response(
                payload,
                item,
                temperature=effective_temperature,
                progress_callback=progress_callback,
                attempt_index=attempt_index,
            )
        except TogetherFatalRequestError:
            raise
        except Exception as exc:
            return ReasoningResult(
                question_id=item.question_id,
                question=item.question,
                gold_answer=item.gold_answer,
                predicted_answer="",
                has_clear_answer=False,
                is_correct=False,
                answer_source="",
                answer_validation="",
                final_response_text="",
                reasoning_trace="",
                temperature=effective_temperature,
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
                    "has_clear_answer",
                    "is_correct",
                    "answer_source",
                    "answer_validation",
                    "final_response_text",
                    "reasoning_trace",
                    "temperature",
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

    def _parse_response(
        self,
        payload: dict[str, object],
        item: BenchmarkItem,
        *,
        temperature: float | None,
        progress_callback: ProgressCallback | None = None,
        attempt_index: int | None = None,
    ) -> ReasoningResult:
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
        answer_source = "parser" if has_clear_answer else ""
        answer_validation = ""
        verifier_is_correct: bool | None = None
        if not has_clear_answer:
            if progress_callback is not None and self.answer_verifier is not None and final_response_text.strip():
                progress_callback(
                    "verifier_started",
                    {
                        "question_id": item.question_id,
                        "attempt_index": attempt_index,
                    },
                )
            verified = self._verify_answer_with_llm(
                response_text=final_response_text,
                gold_answer=item.gold_answer,
                answer_format=item.answer_format,
            )
            if verified is not None:
                if verified.error is not None:
                    answer_source = "llm_verifier_failed"
                    answer_validation = f"LLM verifier failed: {self._short_error(verified.error)}"
                    if progress_callback is not None:
                        progress_callback(
                            "verifier_failed",
                            {
                                "question_id": item.question_id,
                                "attempt_index": attempt_index,
                                "answer_validation": answer_validation,
                            },
                        )
                elif verified.is_correct is not None:
                    predicted_answer = verified.extracted_answer
                    verifier_is_correct = verified.is_correct
                    has_clear_answer = True
                    answer_source = "llm_verifier"
                    answer_validation = f"LLM validated {'RIGHT' if verifier_is_correct else 'WRONG'}"
                    if progress_callback is not None:
                        progress_callback(
                            "verifier_finished",
                            {
                                "question_id": item.question_id,
                                "attempt_index": attempt_index,
                                "answer_validation": answer_validation,
                            },
                        )
        prompt_tokens, completion_tokens, reasoning_tokens = self._extract_usage(payload.get("usage"))
        is_correct = (
            verifier_is_correct
            if verifier_is_correct is not None
            else self._answers_match(
                predicted_answer,
                item.gold_answer,
                answer_format=item.answer_format,
            )
        )

        return ReasoningResult(
            question_id=item.question_id,
            question=item.question,
            gold_answer=item.gold_answer,
            predicted_answer=predicted_answer,
            has_clear_answer=has_clear_answer,
            is_correct=is_correct,
            answer_source=answer_source,
            answer_validation=answer_validation,
            final_response_text=final_response_text,
            reasoning_trace=reasoning_trace,
            temperature=temperature,
            model=self._maybe_string(payload.get("model")) or self.api_shim.config.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            finish_reason=self._maybe_string(first_choice.get("finish_reason")) or "unknown",
            request_id=self._maybe_string(payload.get("id")),
            error=None,
        )

    def _verify_answer_with_llm(
        self,
        *,
        response_text: str,
        gold_answer: str,
        answer_format: str,
    ) -> AnswerVerifierAttempt | None:
        if self.answer_verifier is None or not response_text.strip():
            return None
        try:
            result = self.answer_verifier.verify(
                response_text=response_text,
                gold_answer=gold_answer,
                answer_format=answer_format,
            )
        except Exception as exc:
            return AnswerVerifierAttempt(error=str(exc))
        extracted = self._clean_extracted_answer(result.extracted_answer)
        if not extracted or self._is_no_answer_text(extracted):
            extracted = gold_answer if result.is_correct else "__LLM_VALIDATED_WRONG__"
        return AnswerVerifierAttempt(extracted_answer=extracted, is_correct=result.is_correct)

    @staticmethod
    def _extract_final_answer(text: str, *, answer_format: str) -> tuple[str, bool]:
        if not text.strip():
            return "", False

        extractors = {
            "final": [
                ReasoningTraceSampling._extract_final_marker,
                ReasoningTraceSampling._extract_hash_marker,
                ReasoningTraceSampling._extract_boxed_marker,
                ReasoningTraceSampling._extract_answer_phrase_marker,
            ],
            "hash": [
                ReasoningTraceSampling._extract_hash_marker,
                ReasoningTraceSampling._extract_final_marker,
                ReasoningTraceSampling._extract_boxed_marker,
                ReasoningTraceSampling._extract_answer_phrase_marker,
            ],
            "boxed": [
                ReasoningTraceSampling._extract_boxed_marker,
                ReasoningTraceSampling._extract_final_marker,
                ReasoningTraceSampling._extract_hash_marker,
                ReasoningTraceSampling._extract_answer_phrase_marker,
            ],
        }

        for extractor in extractors.get(answer_format, extractors["final"]):
            answer = extractor(text)
            if answer:
                return ReasoningTraceSampling._clean_extracted_answer(answer), True
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
            match = re.match(
                r"\s*(?:FINAL(?:\s+ANSWER)?|ANSWER)\s*[:=\-]\s*(.+?)\s*$",
                line,
                flags=re.IGNORECASE,
            )
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
        start = ReasoningTraceSampling._find_last_boxed_start(text)
        if start == -1:
            return ""

        depth = 0
        content: list[str] = []
        for char in text[start:]:
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
    def _find_last_boxed_start(text: str) -> int:
        matches = list(re.finditer(r"\\?(?:boxed|fbox)\s*\{", text, flags=re.IGNORECASE))
        if not matches:
            return -1
        match = matches[-1]
        return match.end()

    @staticmethod
    def _extract_answer_phrase_marker(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""

        # Only inspect the ending so intermediate "answer is..." reasoning does not get promoted.
        for line in reversed(lines[-8:]):
            line = ReasoningTraceSampling._strip_markdown_list_prefix(line)
            boxed = ReasoningTraceSampling._extract_boxed_marker(line)
            if boxed:
                return boxed

            answer = ReasoningTraceSampling._extract_from_answer_phrase(line)
            if answer:
                return answer
        return ""

    @staticmethod
    def _extract_from_answer_phrase(line: str) -> str:
        patterns = [
            r"(?:the\s+)?(?:final\s+)?answer\s+(?:is|=|:|-)\s*(.+?)\s*$",
            r"(?:we\s+get|we\s+obtain|this\s+gives|therefore|thus|hence),?\s*(.+?)\s*$",
        ]
        for pattern in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = match.group(1).strip()
            if not candidate:
                continue
            if "answer" in pattern and candidate:
                return candidate
            if ReasoningTraceSampling._looks_like_standalone_answer(candidate):
                return candidate
        return ""

    @staticmethod
    def _looks_like_standalone_answer(text: str) -> bool:
        candidate = ReasoningTraceSampling._clean_extracted_answer(text)
        if not candidate:
            return False
        if len(candidate) > 80:
            return False
        return bool(re.search(r"\\frac|\\sqrt|\\pi|[0-9]", candidate))

    @staticmethod
    def _strip_markdown_list_prefix(line: str) -> str:
        return re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line).strip()

    @staticmethod
    def _clean_extracted_answer(answer: str) -> str:
        text = answer.strip()
        if not text:
            return ""

        boxed = ReasoningTraceSampling._extract_boxed_marker(text)
        if boxed:
            text = boxed.strip()

        text = re.sub(r"^\s*(?:therefore|thus|hence),?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*(?:the\s+)?(?:final\s+)?answer\s+(?:is|=|:|-)\s*", "", text, flags=re.IGNORECASE)
        text = text.strip()

        # Drop common sentence tails from final-answer prose without touching algebra inside.
        text = re.split(r"\s+(?:as\s+required|as\s+desired|which\s+is\s+the\s+answer)\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = text.strip().strip("`*_")
        text = text.strip().rstrip(".,;:")

        changed = True
        while changed:
            changed = False
            for opener, closer in (("\\(", "\\)"), ("\\[", "\\]"), ("$", "$")):
                if text.startswith(opener) and text.endswith(closer):
                    text = text[len(opener) : -len(closer)].strip().rstrip(".,;:")
                    changed = True

        text = re.sub(r"\s+", " ", text)
        return text.strip()

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
        text = ReasoningTraceSampling._clean_extracted_answer(answer)
        if not text:
            return ""

        # Common LaTeX wrappers in math benchmarks.
        text = text.replace("\\left", "").replace("\\right", "")
        text = text.replace("$", "")
        text = text.replace("\\,", "")
        text = re.sub(r"\\(?:mathrm|text)\{([^{}]*)\}", r"\1", text)
        text = re.sub(r"\s+", "", text)

        if answer_format == "hash":
            text = text.replace(",", "")

        text = re.sub(r"(?:\^?\\circ|°|degrees?)$", "", text, flags=re.IGNORECASE)

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

    @staticmethod
    def _short_error(message: str, *, limit: int = 160) -> str:
        text = re.sub(r"\s+", " ", message).strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _is_no_answer_text(answer: str) -> bool:
        normalized = answer.strip().lower()
        return normalized in {
            "none",
            "null",
            "n/a",
            "na",
            "no answer",
            "no valid answer",
            "no valid answer found",
            "unknown",
        }
