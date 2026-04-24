#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_API_BASE = "https://api.together.xyz/v1"
DEFAULT_API_KEY_ENV = "TOGETHER_API_KEY"
DEFAULT_TIMEOUT_SECONDS = 300.0

REQUIRED_INPUT_COLUMNS = (
    "question_id",
    "sample_id",
    "gold_answer",
    "predicted_answer",
    "is_correct",
    "reasoning_trace",
)

OUTPUT_COLUMNS = (
    "question_id",
    "sample_id",
    "gold_answer",
    "predicted_answer",
    "is_correct",
    "tokenized_trace",
)

PROMPT_INSTRUCTION_PLACEHOLDER = """
You are a reasoning-trace canonicalizer.

Your job is to convert each free-form math reasoning trace into a sequence of canonical reasoning-action tokens.

Important rules:
1. Tokenize by FUNCTION, not wording.
   - Different phrasings that perform the same reasoning action should receive the same token.
   - Example: “let x = 5”, “set x to 5”, and “assume x=5” should all map to instantiate.

2. Ignore stylistic or rhetorical language.
   - Do not encode politeness, filler, repetition, hedging, or surface phrasing unless it changes the reasoning function.

3. Use only the token vocabulary below unless absolutely necessary.
4. Do NOT use whether the answer is correct or incorrect when assigning tokens.
   - Tokenization must be label-blind.
5. Preserve order.
   - Output a sequence of tokens in the same order the reasoning actions occur.
6. Merge nearby text into a single token when it serves one reasoning function.
7. If one sentence performs multiple reasoning actions, you may emit multiple tokens.
8. Be conservative.
   - Prefer a small reusable vocabulary over overly specific tokens.

Token vocabulary:
- analyze: interpret the problem, identify what is being asked, restate givens
- instantiate: assign a value, define a variable, substitute a concrete quantity/object
- compute: perform arithmetic or algebraic calculation
- apply-formula: invoke a known formula, theorem, rule, or standard identity
- rewrite: transform an expression/equation into an equivalent form
- check-constraint: verify a condition, test validity, compare against requirements, sanity-check
- case-split: break into cases or branches
- backtrack: abandon a prior path, correct course, restart from earlier reasoning
- conclude: state the final derived result or answer
- guess: make an unsupported leap, intuitive guess, or unexplained candidate answer
- simplify: reduce an expression without materially changing the plan
- compare: compare two quantities/cases/alternatives
- derive-intermediate: produce a named intermediate fact/result used later

If a trace contains reasoning behavior not covered well by the vocabulary, use the closest existing token rather than inventing a new one unless inventing a new token is absolutely necessary.

For each trace:
- Read the full trace.
- Produce:
  (a) a short one-sentence rationale for the overall tokenization
  (b) the canonical token sequence
  (c) a step-alignment table showing short text span -> token

Tokenize all traces independently!
The goal is to use the same canonical token consistently across traces when the same reasoning function appears.
Differences in correctness should not affect token choice.

Return JSON only with this exact schema:
{
  "tokenized_traces": [
    {
      "sample_id": "string",
      "tokenized_trace": "token1 token2 token3 ..."
    }
  ]
}

Make sure to return a tokenized trace for every trace
"""


@dataclass(frozen=True)
class InputRow:
    question_id: str
    sample_id: str
    gold_answer: str
    predicted_answer: str
    is_correct: str
    reasoning_trace: str


class TogetherClient:
    DEFAULT_USER_AGENT = "reasoning-motifs-batch-tokenizer/0.1"

    def __init__(
        self,
        *,
        api_base: str,
        api_key_env: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout_seconds: float,
        repo_root: Path,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self._load_dotenv(repo_root / "batch_tokenizer" / ".env")
        self._load_dotenv(repo_root / "tokenizer" / ".env")
        self._load_dotenv(repo_root / ".env")
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise EnvironmentError(
                f"Missing API key in environment variable {api_key_env}. "
                f"Set it in {repo_root / 'batch_tokenizer' / '.env'}, "
                f"{repo_root / 'tokenizer' / '.env'}, {repo_root / '.env'}, or your shell."
            )

    def tokenize_question_batch(self, *, question_id: str, rows: list[InputRow]) -> list[str]:
        prompt = build_batch_prompt(question_id=question_id, rows=rows)
        payload = self._post_chat_completion(prompt)
        content = extract_assistant_text(payload)
        return parse_batch_response(content=content, rows=rows)

    def _post_chat_completion(self, prompt: str) -> dict[str, Any]:
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        req = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": self.DEFAULT_USER_AGENT,
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Together request failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Together request failed: {exc.reason}") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Together returned non-JSON output: {body[:500]}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Expected JSON object from Together, got: {type(parsed).__name__}")
        if "error" in parsed:
            raise RuntimeError(f"Together returned an error payload: {parsed['error']}")
        return parsed

    @staticmethod
    def _load_dotenv(path: Path) -> None:
        if not path.exists():
            return
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-tokenize traces by grouping rows by question_id and sending one "
            "Together request per question (all traces in that question in one request)."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Input CSV path.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Defaults to "
            "<input_stem>_tokenized_batch_<model-suffix>_<run-index>.csv next to input."
        ),
    )
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Append to output CSV instead of overwriting it.",
    )
    parser.add_argument(
        "--start-question-id",
        default=None,
        help=(
            "Skip all questions until this question_id is reached; "
            "processing includes this question_id."
        ),
    )
    parser.add_argument("--run-index", type=int, default=0, help="Run index suffix (default: 0).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Together model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help=f"API base (default: {DEFAULT_API_BASE}).")
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help=f"API key env var (default: {DEFAULT_API_KEY_ENV}).",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p.")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max completion tokens.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout seconds (default: {DEFAULT_TIMEOUT_SECONDS:g}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_csv = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else _default_output_path(input_csv=input_csv, model=args.model, run_index=args.run_index)
    )

    rows = load_input_rows(input_csv)
    grouped: dict[str, list[InputRow]] = defaultdict(list)
    for row in rows:
        grouped[row.question_id].append(row)

    repo_root = Path(__file__).resolve().parents[1]
    client = TogetherClient(
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        repo_root=repo_root,
    )

    initialize_output_csv(output_csv=output_csv, append_output=args.append_output)

    question_ids = sorted(grouped.keys(), key=_smart_sort_key)
    start_question_id = _normalize(args.start_question_id) if args.start_question_id is not None else None
    started = start_question_id is None
    if start_question_id is not None and start_question_id not in grouped:
        print(
            f"start-question-id={start_question_id} not found in input; nothing will be processed.",
            file=sys.stderr,
        )
    total_questions = len(question_ids)
    processed_count = 0
    for question_id in question_ids:
        if not started:
            if question_id == start_question_id:
                started = True
            else:
                continue
        processed_count += 1
        question_rows = grouped[question_id]
        print(
            f"[{processed_count}/{total_questions}] Batch tokenizing question_id={question_id} traces={len(question_rows)}",
            file=sys.stderr,
        )
        tokenized_traces = client.tokenize_question_batch(question_id=question_id, rows=question_rows)
        output_rows: list[dict[str, str]] = []
        for row, tokenized in zip(question_rows, tokenized_traces):
            output_rows.append(
                {
                    "question_id": row.question_id,
                    "sample_id": row.sample_id,
                    "gold_answer": row.gold_answer,
                    "predicted_answer": row.predicted_answer,
                    "is_correct": row.is_correct,
                    "tokenized_trace": tokenized,
                }
            )
        append_output_rows(output_csv=output_csv, output_rows=output_rows)
    print(f"Wrote batch tokenized traces to: {output_csv}")


def load_input_rows(path: Path) -> list[InputRow]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {path}")
        missing = [name for name in REQUIRED_INPUT_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")

        rows: list[InputRow] = []
        for record in reader:
            rows.append(
                InputRow(
                    question_id=_normalize(record.get("question_id")),
                    sample_id=_normalize(record.get("sample_id")),
                    gold_answer=_normalize(record.get("gold_answer")),
                    predicted_answer=_normalize(record.get("predicted_answer")),
                    is_correct=_normalize(record.get("is_correct")),
                    reasoning_trace=_normalize(record.get("reasoning_trace")),
                )
            )
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return rows


def build_batch_prompt(*, question_id: str, rows: list[InputRow]) -> str:
    lines: list[str] = []
    lines.append("Tokenize all traces below in one batch.")
    lines.append("")
    lines.append(PROMPT_INSTRUCTION_PLACEHOLDER.strip())
    lines.append("")
    lines.append("Return JSON only, no markdown, no prose.")
    lines.append(f"question_id: {question_id}")
    lines.append("")
    lines.append("traces:")
    for row in rows:
        lines.append(f"- sample_id: {row.sample_id}")
        lines.append("  reasoning_trace: |")
        trace = row.reasoning_trace or ""
        for trace_line in trace.splitlines() or [""]:
            lines.append(f"    {trace_line}")
    return "\n".join(lines)


def extract_assistant_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Missing choices in Together response: {payload}")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError(f"Malformed choice in Together response: {first_choice}")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"Missing message in Together response: {first_choice}")

    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
    return str(content or "")


def parse_batch_response(*, content: str, rows: list[InputRow]) -> list[str]:
    try:
        parsed = _parse_json_from_text(content)
    except RuntimeError:
        return ["MISSING" for _ in rows]

    candidates: Any = None
    if isinstance(parsed, dict):
        candidates = parsed.get("tokenized_traces")
        if candidates is None:
            candidates = parsed.get("traces")
    elif isinstance(parsed, list):
        candidates = parsed

    if not isinstance(candidates, list):
        return ["MISSING" for _ in rows]

    by_sample_id: dict[str, str] = {}
    by_index: dict[int, str] = {}
    for item in candidates:
        if not isinstance(item, dict):
            continue
        tokenized = _extract_tokenized_from_dict(item)
        if tokenized is None:
            continue
        sample_id = _normalize(item.get("sample_id"))
        if sample_id:
            by_sample_id[sample_id] = tokenized
        trace_index = _maybe_int(item.get("trace_index"))
        if trace_index is not None:
            by_index[trace_index] = tokenized

    out: list[str] = []
    for idx, row in enumerate(rows):
        if row.sample_id in by_sample_id:
            out.append(by_sample_id[row.sample_id])
        elif idx in by_index:
            out.append(by_index[idx])
        else:
            out.append("MISSING")
    return out


def initialize_output_csv(*, output_csv: Path, append_output: bool) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if append_output and output_csv.exists() and output_csv.stat().st_size > 0:
        return
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()


def append_output_rows(*, output_csv: Path, output_rows: list[dict[str, str]]) -> None:
    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(OUTPUT_COLUMNS))
        writer.writerows(output_rows)


def _parse_json_from_text(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise RuntimeError("Together response content is empty.")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        inner = fenced_match.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass

    start_obj = stripped.find("{")
    end_obj = stripped.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidate = stripped[start_obj : end_obj + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    start_list = stripped.find("[")
    end_list = stripped.rfind("]")
    if start_list != -1 and end_list != -1 and end_list > start_list:
        candidate = stripped[start_list : end_list + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise RuntimeError(f"Failed to parse JSON from model output:\n{text[:1200]}")


def _extract_tokenized_from_dict(data: dict[str, Any]) -> str | None:
    direct_keys = (
        "tokenized_trace",
        "canonical_token_sequence",
        "token_sequence_text",
        "tokens_text",
    )
    for key in direct_keys:
        if key in data:
            value = _normalize(data.get(key))
            return value if value else ""

    list_keys = ("token_sequence", "tokens", "canonical_tokens")
    for key in list_keys:
        if key in data:
            seq = _stringify_token_sequence(data.get(key))
            if seq is not None:
                return seq

    step_keys = ("step_alignment", "step_alignments", "steps", "alignment")
    for key in step_keys:
        steps = data.get(key)
        if not isinstance(steps, list):
            continue
        tokens: list[str] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            for token_key in ("token", "canonical_token", "label", "name"):
                if token_key in step:
                    token = _normalize(step.get(token_key))
                    if token:
                        tokens.append(token)
                    break
        if tokens:
            return " ".join(tokens)
    return None


def _stringify_token_sequence(value: Any) -> str | None:
    if isinstance(value, str):
        return _normalize(value)
    if isinstance(value, list):
        tokens: list[str] = []
        for item in value:
            if isinstance(item, str):
                token = _normalize(item)
                if token:
                    tokens.append(token)
            elif isinstance(item, dict):
                for key in ("token", "canonical_token", "label", "name"):
                    if key in item:
                        token = _normalize(item.get(key))
                        if token:
                            tokens.append(token)
                        break
        return " ".join(tokens)
    return None


def _normalize(value: Any) -> str:
    return str(value or "").strip()


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _default_output_path(*, input_csv: Path, model: str, run_index: int) -> Path:
    model_suffix = _model_to_suffix(model)
    return input_csv.with_name(f"{input_csv.stem}_tokenized_batch_{model_suffix}_{run_index}.csv")


def _model_to_suffix(model: str) -> str:
    tail = model.rsplit("/", 1)[-1].strip()
    if not tail:
        tail = "model"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", tail)
    safe = re.sub(r"-{2,}", "-", safe).strip("-")
    return safe or "model"


def _smart_sort_key(value: str) -> tuple[int, Any]:
    as_int = _maybe_int(value)
    if as_int is not None:
        return 0, as_int
    return 1, value


if __name__ == "__main__":
    main()
