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

PROMPT_INSTRUCTION_PLACEHOLDER = (
    """
You are a reasoning-trace canonicalizer.

Your job is to tokenize ONE free-form math reasoning trace into a sequence of qualified reasoning-action tokens using the provided tokenization guide.

Each output token MUST have the form  basetype:qualifier
- basetype  is a token from the guide's final vocabulary (e.g. analyze, compute, rewrite)
- qualifier  is a short hyphenated noun phrase (1-3 words) naming the mathematical object,
  concept, or sub-goal being operated on (e.g. constraint, modular-inverse, sum-formula,
  parity-argument, base-case, variable-substitution)

The qualifier makes two traces with the same step type distinguishable when they operate on
different mathematical content.  Choose qualifiers from this suggested list when they fit,
and coin new ones (hyphenated, lowercase) when they don't:
  setup, constraint, formula, equation, expression, variable, value, bound, parity,
  modular, geometric, algebraic, combinatorial, proof, case, identity, sum, product,
  sequence, recurrence, inequality, probability, counting, substitution, factorization,
  simplification, intermediate-result, final-answer

You must follow the guide exactly for the basetype.
Do not use answer correctness when assigning tokens.
Tokenize by reasoning FUNCTION, not wording.

Instructions:
1. Read the tokenization guide carefully.
2. Read the question and the trace.
3. Segment the trace into reasoning spans according to the guide.
4. Map each span to a basetype from the guide's final vocabulary.
5. Choose a qualifier that describes the mathematical content of that span.
6. Emit basetype:qualifier for each span, preserving order.
7. If one sentence performs multiple reasoning functions, emit multiple tokens.
8. Ignore filler, hedging, politeness, and stylistic language unless it changes reasoning function.
9. Use the ambiguity rules when a span could fit multiple basetypes.
10. Use the closest existing basetype rather than inventing a new one; qualifiers may be new.

return tokenized sequence as
{"tokenized_trace": "basetype1:qualifier1 basetype2:qualifier2 ..."}
"""
)

METADATA_INSTRUCTION_PLACEHOLDER = """
You are designing a tokenization guide for canonicalizing free-form math reasoning traces into
discrete qualified reasoning-action tokens of the form  basetype:qualifier.

Your goal is to create a stable, reusable, label-blind tokenization policy that will later be
used to tokenize individual traces one at a time.

This is NOT the final tokenization step.
Do NOT tokenize the traces yet unless needed for brief examples.
Instead, infer the metadata and rules needed for later tokenization.

## Token format

Every emitted token must be  basetype:qualifier  where:
- basetype  is drawn from the fixed base vocabulary below (do not invent new basetypes).
- qualifier  is a short hyphenated noun phrase (1-3 words, lowercase) identifying the
  mathematical object, concept, or sub-goal being acted on in that span.

Two traces that perform the same reasoning function on different mathematical content must
produce different tokens (e.g. analyze:parity vs analyze:geometric-construction).

Qualifiers should be specific enough to distinguish traces but general enough to recur across
questions of the same type.  Suggested qualifier pool (extend if needed):
  setup, constraint, formula, equation, expression, variable, value, bound, parity,
  modular, geometric, algebraic, combinatorial, proof, case, identity, sum, product,
  sequence, recurrence, inequality, probability, counting, substitution, factorization,
  simplification, intermediate-result, final-answer

## Fixed base vocabulary

- analyze
- instantiate
- compute
- apply-formula
- rewrite
- check-constraint
- case-split
- backtrack
- conclude
- guess
- simplify
- compare
- derive-intermediate

## Objectives
1. Keep the base vocabulary fixed and compact.
2. Define each basetype by reasoning FUNCTION, not wording.
3. Define qualifier selection rules that are consistent across traces.
4. Ensure the policy is label-blind: correctness must never affect token assignment.
5. Anticipate ambiguities and define tie-breaking rules.

You should assume that later, another prompt will tokenize traces individually using only the
metadata you produce here.

Tasks:
A. For each basetype in the fixed vocabulary, provide:
   - one-sentence definition
   - inclusion criteria
   - exclusion criteria
   - 2-4 short example text spans with their expected  basetype:qualifier  output

B. Provide qualifier selection rules:
   - how to choose a qualifier for this question's domain
   - when to coin a new qualifier vs reuse a suggested one
   - how to normalize near-synonyms (e.g. "mod" → modular, "eq" → equation)

C. Provide ambiguity-resolution rules for confusing basetype pairs such as:
   - compute vs apply-formula
   - rewrite vs simplify
   - analyze vs derive-intermediate
   - instantiate vs compute
   - check-constraint vs compare
   - backtrack vs rewrite
   - conclude vs guess

D. Provide segmentation rules:
   - when to merge adjacent spans into one token
   - when to emit multiple tokens from one sentence
   - how to treat repeated local computations
   - how to treat rhetorical filler and stylistic text

E. Provide normalization rules:
   - tokenization should depend on reasoning function, not wording
   - equivalent actions across traces should map to the same  basetype:qualifier
   - correctness must not affect token choice

F. Provide a required JSON schema for downstream per-trace tokenization.

Return JSON only with this schema:
{
  "guide_name": "string",
  "version": "string",
  "final_vocabulary": [
    {
      "token": "string",
      "definition": "string",
      "include_when": ["string"],
      "exclude_when": ["string"],
      "examples": ["string (show as basetype:qualifier)"]
    }
  ],
  "qualifier_rules": [
    "string"
  ],
  "ambiguity_rules": [
    {
      "pair": ["token1", "token2"],
      "decision_rule": "string",
      "examples": ["string"]
    }
  ],
  "segmentation_rules": [
    "string"
  ],
  "normalization_rules": [
    "string"
  ],
  "required_output_schema": {
    "trace_id": "string",
    "question_id": "string",
    "tokens": ["string (basetype:qualifier)"],
    "alignment": [
      {
        "text_span": "string",
        "token": "string (basetype:qualifier)"
      }
    ]
  },
  "notes_for_tokenizer": [
    "string"
  ]
}
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
    DEFAULT_USER_AGENT = "reasoning-motifs-tokenizer/0.1"

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
        self.api_key_env = api_key_env
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        # Load tokenizer/.env first (local to this script), then fallback to repo root .env.
        self._load_dotenv(repo_root / "tokenizer" / ".env")
        self._load_dotenv(repo_root / ".env")
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise EnvironmentError(
                f"Missing API key in environment variable {api_key_env}. "
                f"Set it in {repo_root / 'tokenizer' / '.env'}, {repo_root / '.env'}, or your shell."
            )

    def tokenize_trace(self, *, row: InputRow, metadata: dict[str, Any]) -> str:
        prompt = build_single_trace_prompt(row=row, metadata=metadata)
        payload = self._post_chat_completion(prompt)
        content = extract_assistant_text(payload)
        print(content)
        return parse_single_trace_response(content=content)

    def build_question_metadata(self, *, question_id: str, rows: list[InputRow]) -> dict[str, Any]:
        prompt = build_question_metadata_prompt(question_id=question_id, rows=rows)
        payload = self._post_chat_completion(prompt)
        content = extract_assistant_text(payload)
        return parse_metadata_response(content)

    def _post_chat_completion(self, prompt: str) -> dict[str, Any]:
        
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You convert reasoning traces into concise token sequences. "
                        "Return valid JSON only."
                    ),
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
            "Tokenize reasoning traces by grouping input CSV rows by question_id, "
            "sending one Together request per question, and writing a tokenized CSV."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to input CSV (example: math_motif_pilot_v6_qwen25_7b_hot30.csv).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Defaults to "
            "<input_stem>_tokenized_<model-suffix>_<run-index>.csv next to input."
        ),
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        help="Run index to append to default output filename (default: 0).",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help=(
            "Metadata output path (JSONL). Defaults to "
            "<input_stem>_metadata_<model-suffix>_<run-index>.jsonl next to input."
        ),
    )
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Append to output/metadata files instead of overwriting them.",
    )
    parser.add_argument(
        "--start-question-id",
        default=None,
        help=(
            "Skip all questions until this question_id is reached; "
            "processing includes this question_id."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Together model (default: {DEFAULT_MODEL}).")
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"Together API base URL (default: {DEFAULT_API_BASE}).",
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help=f"Environment variable name for Together API key (default: {DEFAULT_API_KEY_ENV}).",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0).")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling parameter (default: 1.0).")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max output tokens per Together request (default: 8192).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout per request in seconds (default: {DEFAULT_TIMEOUT_SECONDS:g}).",
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
    metadata_output = (
        args.metadata_output.resolve()
        if args.metadata_output is not None
        else _default_metadata_output_path(input_csv=input_csv, model=args.model, run_index=args.run_index)
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
    initialize_metadata_jsonl(metadata_output=metadata_output, append_output=args.append_output)

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
            f"[{processed_count}/{total_questions}] Tokenizing question_id={question_id} traces={len(question_rows)}",
            file=sys.stderr,
        )
        metadata = client.build_question_metadata(question_id=question_id, rows=question_rows)
        append_metadata_record(
            metadata_output=metadata_output,
            question_id=question_id,
            rows=question_rows,
            metadata=metadata,
        )

        output_rows: list[dict[str, str]] = []
        for row in question_rows:
            tokenized = client.tokenize_trace(row=row, metadata=metadata)
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

    print(f"Wrote tokenized traces to: {output_csv}")
    print(f"Wrote metadata per question to: {metadata_output}")


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


def build_question_metadata_prompt(*, question_id: str, rows: list[InputRow]) -> str:
    lines: list[str] = []
    lines.append("You are creating shared metadata for a later per-trace tokenization step.")
    lines.append("")
    lines.append(METADATA_INSTRUCTION_PLACEHOLDER.strip())
    lines.append("")
    lines.append("Return JSON only. Do not return markdown or prose outside JSON.")
    lines.append("")
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


def build_single_trace_prompt(*, row: InputRow, metadata: dict[str, Any]) -> str:
    metadata_json = json.dumps(metadata, ensure_ascii=True)
    lines: list[str] = []
    lines.append("Tokenize this single reasoning trace using the shared metadata below.")
    lines.append("")
    lines.append(f"INSTRUCTIONS_PLACEHOLDER: {PROMPT_INSTRUCTION_PLACEHOLDER.strip()}")
    lines.append("")
    lines.append("Shared metadata (from step 1):")
    lines.append(metadata_json)
    lines.append("")
    lines.append('Return JSON only with this exact shape: {"tokenized_trace":"..."}')
    lines.append("If no valid tokenization can be produced, return tokenized_trace as MISSING.")
    lines.append("")
    lines.append(f"question_id: {row.question_id}")
    lines.append(f"sample_id: {row.sample_id}")
    lines.append("reasoning_trace: |")
    trace = row.reasoning_trace or ""
    for trace_line in trace.splitlines() or [""]:
        lines.append(f"  {trace_line}")
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


def parse_metadata_response(content: str) -> dict[str, Any]:
    try:
        parsed = _parse_json_from_text(content)
    except RuntimeError:
        return {"raw_metadata": content[:4000]}

    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"token_dictionary": parsed}
    return {"raw_metadata": content[:4000]}


def parse_single_trace_response(*, content: str) -> str:
    try:
        parsed = _parse_json_from_text(content)
    except RuntimeError:
        return _extract_tokenized_trace_from_text(content)

    extracted = _extract_tokenized_trace_from_parsed(parsed)
    if extracted is not None:
        return extracted
    return _extract_tokenized_trace_from_text(content)


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


def initialize_metadata_jsonl(*, metadata_output: Path, append_output: bool) -> None:
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    if append_output and metadata_output.exists() and metadata_output.stat().st_size > 0:
        return
    with metadata_output.open("w", encoding="utf-8"):
        pass


def append_metadata_record(
    *,
    metadata_output: Path,
    question_id: str,
    rows: list[InputRow],
    metadata: dict[str, Any],
) -> None:
    with metadata_output.open("a", encoding="utf-8") as f:
        record = {
            "question_id": question_id,
            "num_traces": len(rows),
            "sample_ids": [row.sample_id for row in rows],
            "metadata": metadata,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def _normalize(value: Any) -> str:
    return str(value or "").strip()


def _extract_tokenized_trace_from_parsed(parsed: Any) -> str | None:
    if isinstance(parsed, dict):
        direct = _extract_tokenized_from_dict(parsed)
        if direct is not None:
            return direct

        candidates = parsed.get("tokenized_traces")
        if isinstance(candidates, list) and candidates:
            first = candidates[0]
            if isinstance(first, dict):
                nested = _extract_tokenized_from_dict(first)
                if nested is not None:
                    return nested

    if isinstance(parsed, list) and parsed:
        first = parsed[0]
        if isinstance(first, dict):
            nested = _extract_tokenized_from_dict(first)
            if nested is not None:
                return nested
    return None


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

    list_keys = (
        "token_sequence",
        "tokens",
        "canonical_tokens",
    )
    for key in list_keys:
        if key in data:
            seq = _stringify_token_sequence(data.get(key))
            if seq is not None:
                return seq

    # If model returns step-alignment only, reconstruct sequence from step tokens.
    step_keys = ("step_alignment", "step_alignments", "steps", "alignment")
    for key in step_keys:
        if key not in data:
            continue
        steps = data.get(key)
        if isinstance(steps, list):
            tokens: list[str] = []
            for step in steps:
                if not isinstance(step, dict):
                    continue
                t = _assemble_qualified_token(step)
                if t:
                    tokens.append(t)
            if tokens:
                return " ".join(tokens)
    return None


def _assemble_qualified_token(step: dict[str, Any]) -> str:
    """Build a basetype:qualifier string from a step dict.

    Handles models that return the token as a single "token" key already in
    basetype:qualifier form, or as separate "basetype"/"qualifier" keys, or
    as a plain token with no qualifier.
    """
    # Already combined.
    for key in ("token", "canonical_token", "label"):
        if key in step:
            t = _normalize(step[key])
            if t:
                qualifier = _normalize(step.get("qualifier", ""))
                if qualifier and ":" not in t:
                    return f"{t}:{qualifier}"
                return t
    # Separate basetype + qualifier keys.
    basetype = _normalize(step.get("basetype", step.get("base_type", step.get("type", ""))))
    qualifier = _normalize(step.get("qualifier", step.get("subject", step.get("content", ""))))
    if basetype and qualifier:
        return f"{basetype}:{qualifier}"
    if basetype:
        return basetype
    return ""


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
                t = _assemble_qualified_token(item)
                if t:
                    tokens.append(t)
        return " ".join(tokens)
    return None


def _extract_tokenized_trace_from_text(content: str) -> str:
    stripped = content.strip()
    if not stripped:
        return "MISSING"

    # Common plain-text fallback patterns.
    patterns = (
        r"canonical token sequence\s*:\s*(.+)",
        r"tokenized trace\s*:\s*(.+)",
        r"token sequence\s*:\s*(.+)",
        r"tokens\s*:\s*(.+)",
    )
    for pattern in patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match:
            value = _normalize(match.group(1))
            if value:
                return value
            return ""

    return "MISSING"


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _default_output_path(*, input_csv: Path, model: str, run_index: int) -> Path:
    model_suffix = _model_to_suffix(model)
    return input_csv.with_name(f"{input_csv.stem}_tokenized_{model_suffix}_{run_index}.csv")


def _default_metadata_output_path(*, input_csv: Path, model: str, run_index: int) -> Path:
    model_suffix = _model_to_suffix(model)
    return input_csv.with_name(f"{input_csv.stem}_metadata_{model_suffix}_{run_index}.jsonl")


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
