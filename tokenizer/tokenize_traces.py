#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import sys
import threading
import time
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

CHUNKING_INSTRUCTION_PLACEHOLDER = """
You are segmenting ONE free-form math reasoning trace into ordered spans.

Your job is to split the trace into chunks and label each chunk as either:
- `reasoning`: interpretable math reasoning
- `noise`: corrupted, garbled, irrelevant, or non-math content

Rules:
1. Preserve original order.
2. Chunks should be semantically coherent spans, not arbitrary fixed-length slices.
3. If a trace mixes valid reasoning with corrupted garbage, keep the reasoning spans and isolate the corrupted parts as `noise`.
4. Do not drop content silently. Every meaningful part of the trace should be covered by some chunk.
5. If a span is unreadable, code-like garbage, repeated corruption, multilingual junk unrelated to the math, or formatting debris, mark it as `noise`.
6. If a span contains both valid reasoning and corruption, split it if possible.
7. Keep the number of chunks reasonable. Prefer larger coherent spans over many tiny ones.

Return JSON only with this exact shape:
{
  "chunks": [
    {
      "text_span": "string",
      "span_type": "reasoning|noise"
    }
  ]
}
"""

PROMPT_INSTRUCTION_PLACEHOLDER = (
    """
You are a reasoning-trace canonicalizer.

Your job is to tokenize ONE free-form math reasoning trace into a sequence of canonical reasoning tokens using the provided tokenization guide.

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
2. Read the question and the provided pre-segmented chunks.
3. Ignore chunks labeled `noise` except to emit the canonical token `noise:corrupted-span`.
4. Map each reasoning chunk to one or more tokens from the guide's final vocabulary.
5. Preserve order.
6. If one chunk performs multiple reasoning functions, emit multiple tokens.
7. Ignore filler, hedging, politeness, and stylistic language unless it changes reasoning function.
8. Use the ambiguity rules when a span could fit multiple tokens.
9. Use the closest existing token rather than inventing a new one.
10. Every emitted token must include an explicit type prefix:
   - `action:<name>` for local reasoning moves
   - `strategy:<name>` for higher-level plan choices or subgoal framing
   - `milestone:<name>` for intermediate conclusions or state updates reached by the reasoning
   - `noise:corrupted-span` for chunks labeled as noise
11. Keep `strategy` and `milestone` vocabularies compact and reusable across traces.
12. Do not emit a `strategy` or `milestone` token for every sentence. Emit them only when the trace clearly commits to a plan or reaches a meaningful intermediate conclusion.

return tokenized sequence as
{"tokenized_trace": "action:token1 noise:corrupted-span milestone:token3 ..."}
"""
)

METADATA_INSTRUCTION_PLACEHOLDER = """
You are designing a tokenization guide for canonicalizing free-form math reasoning traces into discrete reasoning tokens.

Your goal is to create a stable, reusable, label-blind tokenization policy that will later be
used to tokenize individual traces one at a time.

This is NOT the final tokenization step.
Do NOT tokenize the traces yet unless needed for brief examples.
Instead, infer the metadata and rules needed for later tokenization.

Objectives:
1. Create a shared canonical token vocabulary for math reasoning traces.
2. Use FOUR token types:
   - action: local reasoning moves
   - strategy: higher-level plan choices or subgoal framing
   - milestone: intermediate conclusions or state updates reached by the reasoning
   - noise: reserved for corrupted or unreadable spans only
3. Define each token by reasoning FUNCTION, not wording.
4. Make the vocabulary compact, reusable, and robust across many questions.
5. Ensure the policy is label-blind: correctness must never affect token assignment.
6. Anticipate ambiguities and define tie-breaking rules.

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

Base action vocabulary candidate:
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

Tasks:
A. Review the candidate action vocabulary and decide:
   - which tokens should remain as-is
   - which should be merged
   - which need sharper definitions
   - whether any additional action tokens are absolutely necessary

B. Propose compact `strategy` and `milestone` vocabularies.
   - `strategy` tokens should represent plans such as choosing a representation,
     reducing to a known form, splitting cases, switching viewpoints, or setting
     a subgoal.
   - `milestone` tokens should represent achieved intermediate states such as
     deriving a key relation, reducing to one variable, obtaining a closed-form
     expression, identifying a boundary case, or proving a required condition.
   - Milestones may be more specific than actions, but they should still be normalized.
   - Prefer milestone templates such as:
     - `key-relation:x=y`
     - `key-relation:expr=const`
     - `key-relation:quadratic-in-one-var`
     - `reduced-to:one-variable`
     - `derived:closed-form`
     - `identified:boundary-case`
   - Avoid raw free-form milestone text and avoid highly question-specific labels when a reusable abstraction is possible.

C. For each token in the final vocabulary, provide:
   - token name
   - token type (`action`, `strategy`, `milestone`, or `noise`)
   - one-sentence definition
   - inclusion criteria
   - exclusion criteria
   - 2-4 short example text spans with their expected  basetype:qualifier  output

D. Provide ambiguity-resolution rules for confusing pairs such as:
   - compute vs apply-formula
   - rewrite vs simplify
   - analyze vs derive-intermediate
   - instantiate vs compute
   - check-constraint vs compare
   - backtrack vs rewrite
   - conclude vs guess
   - action vs strategy
   - action vs milestone
   - strategy vs milestone
   - reasoning vs noise

E. Provide segmentation rules:
   - when to merge adjacent spans into one token
   - when to emit multiple tokens from one sentence
   - how to treat repeated local computations
   - how to treat rhetorical filler and stylistic text
   - when a single span should emit both an `action` and a `milestone`
   - when a planning sentence should emit a `strategy`
   - when a chunk should be labeled `noise`

F. Provide normalization rules:
   - tokenization should depend on reasoning function, not wording
   - equivalent actions across traces should map to the same token
   - equivalent strategies across traces should map to the same token
   - equivalent intermediate conclusions across traces should map to the same token
   - correctness must not affect token choice
   - downstream traces will be flattened into strings like `action:compute milestone:reduced-expression`
   - milestone names should use normalized templates rather than free-form prose
   - the only allowed noise token is `noise:corrupted-span`

G. Provide a required JSON schema for downstream per-trace tokenization.

Return JSON only with this schema:
{
  "guide_name": "string",
  "version": "string",
  "final_vocabulary": [
    {
      "token": "string",
      "token_type": "action|strategy|milestone|noise",
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
<<<<<<< HEAD
    "tokens": ["string (basetype:qualifier)"],
    "alignment": [
      {
        "text_span": "string",
        "token": "string (basetype:qualifier)"
=======
    "tokenized_trace": "space-delimited string of type-prefixed tokens",
    "tokens": [
      {
        "type": "action|strategy|milestone|noise",
        "token": "string",
        "text_span": "string"
      }
    ],
    "alignment": [
      {
        "text_span": "string",
        "token": "string",
        "token_type": "action|strategy|milestone|noise"
>>>>>>> 3ec13d5d5029df1065fa0d04f9bc6696d6b0151d
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


@dataclass(frozen=True)
class TraceChunk:
    text_span: str
    span_type: str


@dataclass(frozen=True)
class ChunkedTrace:
    row: InputRow
    chunks: list[TraceChunk]


@dataclass(frozen=True)
class QuestionTokenizationResult:
    question_id: str
    chunked_traces: list[ChunkedTrace]
    metadata: dict[str, Any]
    output_rows: list[dict[str, str]]


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
        prompt = build_single_trace_prompt(row=row, metadata=metadata, chunks=[])
        payload = self._post_chat_completion(prompt)
        content = extract_assistant_text(payload)
        return parse_single_trace_response(content=content)

    def chunk_trace(self, *, row: InputRow) -> list[TraceChunk]:
        prompt = build_trace_chunking_prompt(row=row)
        payload = self._post_chat_completion(prompt)
        content = extract_assistant_text(payload)
        return parse_chunking_response(row=row, content=content)

    def build_question_metadata(
        self,
        *,
        question_id: str,
        chunked_traces: list[ChunkedTrace],
    ) -> dict[str, Any]:
        prompt = build_question_metadata_prompt(
            question_id=question_id,
            chunked_traces=chunked_traces,
        )
        payload = self._post_chat_completion(prompt)
        content = extract_assistant_text(payload)
        return parse_metadata_response(content)

    def tokenize_chunked_trace(
        self,
        *,
        chunked_trace: ChunkedTrace,
        metadata: dict[str, Any],
    ) -> str:
        prompt = build_single_trace_prompt(
            row=chunked_trace.row,
            metadata=metadata,
            chunks=chunked_trace.chunks,
        )
        payload = self._post_chat_completion(prompt)
        content = extract_assistant_text(payload)
        return parse_single_trace_response(content=content)

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
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of questions to tokenize concurrently (default: 4).",
    )
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=None,
        help="Optional JSONL path for live progress events.",
    )
    parser.add_argument(
        "--progress-print-every",
        type=int,
        default=5,
        help="Print per-question trace progress every N traces (default: 5).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the entire run on the first question-level error (default: keep going).",
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
    progress_log = args.progress_log.resolve() if args.progress_log is not None else None
    existing_output_keys = (
        load_existing_output_keys(output_csv) if args.append_output else set()
    )

    rows = load_input_rows(input_csv)
    grouped: dict[str, list[InputRow]] = defaultdict(list)
    for row in rows:
        grouped[row.question_id].append(row)

    skipped_trace_count = 0
    skipped_question_count = 0
    if existing_output_keys:
        for question_id, question_rows in list(grouped.items()):
            remaining_rows = [
                row
                for row in question_rows
                if _row_output_key(question_id=row.question_id, sample_id=row.sample_id)
                not in existing_output_keys
            ]
            skipped_here = len(question_rows) - len(remaining_rows)
            if skipped_here > 0:
                skipped_trace_count += skipped_here
                if len(remaining_rows) == 0:
                    skipped_question_count += 1
                    print(
                        f"[cached] question_id={question_id} traces={len(question_rows)} already tokenized; skipping",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[resume] question_id={question_id} skipping {skipped_here} cached trace(s), processing {len(remaining_rows)}",
                        file=sys.stderr,
                    )
            grouped[question_id] = remaining_rows

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
    progress_lock = threading.Lock()
    initialize_progress_log(
        progress_log=progress_log,
        append_output=args.append_output,
    )

    question_ids = sorted(grouped.keys(), key=_smart_sort_key)
    start_question_id = _normalize(args.start_question_id) if args.start_question_id is not None else None
    started = start_question_id is None
    if start_question_id is not None and start_question_id not in grouped:
        print(
            f"start-question-id={start_question_id} not found in input; nothing will be processed.",
            file=sys.stderr,
        )
    selected_question_ids: list[str] = []
    for question_id in question_ids:
        if not started:
            if question_id == start_question_id:
                started = True
            else:
                continue
        if not grouped[question_id]:
            continue
        selected_question_ids.append(question_id)

    total_questions = len(selected_question_ids)
    if total_questions == 0:
        print("No questions selected for tokenization.", file=sys.stderr)
        print(f"Wrote tokenized traces to: {output_csv}")
        print(f"Wrote metadata per question to: {metadata_output}")
        return
    append_progress_event(
        progress_log=progress_log,
        lock=progress_lock,
        event={
            "event": "run_started",
            "input_csv": str(input_csv),
            "output_csv": str(output_csv),
            "metadata_output": str(metadata_output),
            "total_questions": total_questions,
            "workers": max(1, args.workers),
            "cached_trace_skips": skipped_trace_count,
            "cached_question_skips": skipped_question_count,
        },
    )

    workers = max(1, args.workers)
    failed_questions: list[tuple[str, str]] = []
    if workers == 1:
        for completed_count, question_id in enumerate(selected_question_ids, start=1):
            question_rows = grouped[question_id]
            print(
                f"[{completed_count}/{total_questions}] Tokenizing question_id={question_id} traces={len(question_rows)}",
                file=sys.stderr,
            )
            append_progress_event(
                progress_log=progress_log,
                lock=progress_lock,
                event={
                    "event": "question_started",
                    "question_id": question_id,
                    "expected_traces": len(question_rows),
                },
            )
            try:
                result = tokenize_question(
                    question_id=question_id,
                    rows=question_rows,
                    client=client,
                    progress_callback=make_progress_callback(
                        progress_log=progress_log,
                        lock=progress_lock,
                        question_id=question_id,
                        expected_traces=len(question_rows),
                        print_every=max(1, args.progress_print_every),
                    ),
                )
            except Exception as exc:
                failed_questions.append((question_id, str(exc)))
                append_progress_event(
                    progress_log=progress_log,
                    lock=progress_lock,
                    event={
                        "event": "question_error",
                        "question_id": question_id,
                        "error": str(exc),
                    },
                )
                print(f"[error] question_id={question_id}: {exc}", file=sys.stderr)
                if args.fail_fast:
                    raise
                continue
            write_question_result(
                result=result,
                output_csv=output_csv,
                metadata_output=metadata_output,
            )
            append_progress_event(
                progress_log=progress_log,
                lock=progress_lock,
                event={
                    "event": "question_done",
                    "question_id": result.question_id,
                    "expected_traces": len(result.chunked_traces),
                },
            )
    else:
        for queued_count, question_id in enumerate(selected_question_ids, start=1):
            question_rows = grouped[question_id]
            print(
                f"[queued {queued_count}/{total_questions}] question_id={question_id} traces={len(question_rows)}",
                file=sys.stderr,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_question: dict[concurrent.futures.Future[QuestionTokenizationResult], str] = {}
            for question_id in selected_question_ids:
                append_progress_event(
                    progress_log=progress_log,
                    lock=progress_lock,
                    event={
                        "event": "question_started",
                        "question_id": question_id,
                        "expected_traces": len(grouped[question_id]),
                    },
                )
                future = executor.submit(
                    tokenize_question,
                    question_id=question_id,
                    rows=grouped[question_id],
                    client=client,
                    progress_callback=make_progress_callback(
                        progress_log=progress_log,
                        lock=progress_lock,
                        question_id=question_id,
                        expected_traces=len(grouped[question_id]),
                        print_every=max(1, args.progress_print_every),
                    ),
                )
                future_to_question[future] = question_id
            for completed_count, future in enumerate(
                concurrent.futures.as_completed(future_to_question),
                start=1,
            ):
                question_id = future_to_question[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failed_questions.append((question_id, str(exc)))
                    append_progress_event(
                        progress_log=progress_log,
                        lock=progress_lock,
                        event={
                            "event": "question_error",
                            "question_id": question_id,
                            "error": str(exc),
                        },
                    )
                    print(f"[error] question_id={question_id}: {exc}", file=sys.stderr)
                    if args.fail_fast:
                        raise
                    continue
                write_question_result(
                    result=result,
                    output_csv=output_csv,
                    metadata_output=metadata_output,
                )
                append_progress_event(
                    progress_log=progress_log,
                    lock=progress_lock,
                    event={
                        "event": "question_done",
                        "question_id": result.question_id,
                        "expected_traces": len(result.chunked_traces),
                    },
                )
                print(
                    f"[done {completed_count}/{total_questions}] question_id={result.question_id} traces={len(result.chunked_traces)}",
                    file=sys.stderr,
                )

    append_progress_event(
        progress_log=progress_log,
        lock=progress_lock,
        event={
            "event": "run_finished",
            "total_questions": total_questions,
            "failed_questions": len(failed_questions),
            "cached_trace_skips": skipped_trace_count,
            "cached_question_skips": skipped_question_count,
        },
    )
    if failed_questions:
        print(f"Run finished with {len(failed_questions)} failed question(s):", file=sys.stderr)
        for qid, err in failed_questions:
            print(f"  - question_id={qid}: {err}", file=sys.stderr)
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


def _row_output_key(*, question_id: str, sample_id: str) -> tuple[str, str]:
    return (_normalize(question_id), _normalize(sample_id))


def load_existing_output_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return set()
        if "question_id" not in reader.fieldnames or "sample_id" not in reader.fieldnames:
            return set()
        keys: set[tuple[str, str]] = set()
        for row in reader:
            keys.add(
                _row_output_key(
                    question_id=row.get("question_id", ""),
                    sample_id=row.get("sample_id", ""),
                )
            )
    return keys


def build_noise_only_metadata() -> dict[str, Any]:
    return {
        "guide_name": "NoiseOnlyFallbackGuide",
        "version": "1.0",
        "final_vocabulary": [
            {
                "token": "corrupted-span",
                "token_type": "noise",
                "definition": "Reserved token for unreadable or corrupted trace spans.",
                "include_when": ["The span is garbled, code-like, or otherwise not interpretable math reasoning."],
                "exclude_when": ["The span contains interpretable math reasoning."],
                "examples": ["RuntimeException ... mongodb ...", "garbled multilingual junk"],
            }
        ],
        "notes_for_tokenizer": [
            "Fallback metadata used when no reasoning chunks were detected.",
            "The only allowed emitted token is noise:corrupted-span.",
        ],
    }


def tokenize_question(
    *,
    question_id: str,
    rows: list[InputRow],
    client: TogetherClient,
    progress_callback: Any = None,
) -> QuestionTokenizationResult:
    chunked_traces: list[ChunkedTrace] = []
    for chunked_index, row in enumerate(rows, start=1):
        chunks = client.chunk_trace(row=row)
        chunked_traces.append(ChunkedTrace(row=row, chunks=chunks))
        if callable(progress_callback):
            progress_callback("trace_chunked", chunked_index)
    if any(
        chunk.span_type == "reasoning"
        for chunked_trace in chunked_traces
        for chunk in chunked_trace.chunks
    ):
        metadata = client.build_question_metadata(
            question_id=question_id,
            chunked_traces=chunked_traces,
        )
    else:
        metadata = build_noise_only_metadata()
    if callable(progress_callback):
        progress_callback("metadata_ready", 0)
    output_rows: list[dict[str, str]] = []
    for tokenized_index, chunked_trace in enumerate(chunked_traces, start=1):
        if all(chunk.span_type == "noise" for chunk in chunked_trace.chunks):
            raw_tokenized = "noise:corrupted-span"
        else:
            raw_tokenized = client.tokenize_chunked_trace(
                chunked_trace=chunked_trace,
                metadata=metadata,
            )
        tokenized = normalize_tokenized_trace(
            tokenized_trace=raw_tokenized,
            chunks=chunked_trace.chunks,
        )
        output_rows.append(
            {
                "question_id": chunked_trace.row.question_id,
                "sample_id": chunked_trace.row.sample_id,
                "gold_answer": chunked_trace.row.gold_answer,
                "predicted_answer": chunked_trace.row.predicted_answer,
                "is_correct": chunked_trace.row.is_correct,
                "tokenized_trace": tokenized,
            }
        )
        if callable(progress_callback):
            progress_callback("trace_tokenized", tokenized_index)
    return QuestionTokenizationResult(
        question_id=question_id,
        chunked_traces=chunked_traces,
        metadata=metadata,
        output_rows=output_rows,
    )


def write_question_result(
    *,
    result: QuestionTokenizationResult,
    output_csv: Path,
    metadata_output: Path,
) -> None:
    append_metadata_record(
        metadata_output=metadata_output,
        question_id=result.question_id,
        rows=[chunked_trace.row for chunked_trace in result.chunked_traces],
        metadata=result.metadata,
    )
    append_output_rows(output_csv=output_csv, output_rows=result.output_rows)


def initialize_progress_log(*, progress_log: Path | None, append_output: bool) -> None:
    if progress_log is None:
        return
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    if append_output and progress_log.exists() and progress_log.stat().st_size > 0:
        return
    with progress_log.open("w", encoding="utf-8"):
        pass


def append_progress_event(
    *,
    progress_log: Path | None,
    lock: threading.Lock,
    event: dict[str, Any],
) -> None:
    if progress_log is None:
        return
    payload = dict(event)
    payload.setdefault("ts", time.time())
    with lock:
        with progress_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def make_progress_callback(
    *,
    progress_log: Path | None,
    lock: threading.Lock,
    question_id: str,
    expected_traces: int,
    print_every: int,
) -> Any:
    def _callback(event_name: str, count: int) -> None:
        append_progress_event(
            progress_log=progress_log,
            lock=lock,
            event={
                "event": event_name,
                "question_id": question_id,
                "count": count,
                "expected_traces": expected_traces,
            },
        )
        if event_name in {"trace_chunked", "trace_tokenized"}:
            if count % print_every == 0 or count == expected_traces:
                print(
                    f"[{event_name}] question_id={question_id} {count}/{expected_traces}",
                    file=sys.stderr,
                )
        elif event_name == "metadata_ready":
            print(f"[metadata_ready] question_id={question_id}", file=sys.stderr)

    return _callback


def build_trace_chunking_prompt(*, row: InputRow) -> str:
    lines: list[str] = []
    lines.append("Segment this single reasoning trace into ordered reasoning/noise chunks.")
    lines.append("")
    lines.append(CHUNKING_INSTRUCTION_PLACEHOLDER.strip())
    lines.append("")
    lines.append("Return JSON only. Do not return markdown or prose outside JSON.")
    lines.append("")
    lines.append(f"question_id: {row.question_id}")
    lines.append(f"sample_id: {row.sample_id}")
    lines.append("reasoning_trace: |")
    trace = row.reasoning_trace or ""
    for trace_line in trace.splitlines() or [""]:
        lines.append(f"  {trace_line}")
    return "\n".join(lines)


def build_question_metadata_prompt(
    *,
    question_id: str,
    chunked_traces: list[ChunkedTrace],
) -> str:
    lines: list[str] = []
    lines.append("You are creating shared metadata for a later per-trace tokenization step.")
    lines.append("")
    lines.append(METADATA_INSTRUCTION_PLACEHOLDER.strip())
    lines.append("")
    lines.append("Return JSON only. Do not return markdown or prose outside JSON.")
    lines.append("")
    lines.append(f"question_id: {question_id}")
    lines.append("")
    lines.append("chunked_traces:")
    for chunked_trace in chunked_traces:
        lines.append(f"- sample_id: {chunked_trace.row.sample_id}")
        for chunk in chunked_trace.chunks:
            if chunk.span_type != "reasoning":
                continue
            lines.append(f"  - span_type: {chunk.span_type}")
            lines.append("    text_span: |")
            for trace_line in chunk.text_span.splitlines() or [""]:
                lines.append(f"      {trace_line}")
    return "\n".join(lines)


def build_single_trace_prompt(
    *,
    row: InputRow,
    metadata: dict[str, Any],
    chunks: list[TraceChunk],
) -> str:
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
    lines.append(
        "Flatten every token using explicit type prefixes like "
        "`action:compute`, `strategy:reduce-to-equation`, or `milestone:derived-key-relation`."
    )
    lines.append(
        "If a span performs a local action and reaches a meaningful subresult, emit the action token "
        "followed by the milestone token."
    )
    lines.append(
        "Use normalized milestone templates when possible, for example "
        "`milestone:key-relation:x=y`, `milestone:reduced-to:one-variable`, or "
        "`milestone:derived:closed-form`."
    )
    lines.append("")
    lines.append(f"question_id: {row.question_id}")
    lines.append(f"sample_id: {row.sample_id}")
    lines.append("presegmented_chunks:")
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(f"- chunk_index: {idx}")
        lines.append(f"  span_type: {chunk.span_type}")
        lines.append("  text_span: |")
        for trace_line in chunk.text_span.splitlines() or [""]:
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


def parse_chunking_response(*, row: InputRow, content: str) -> list[TraceChunk]:
    try:
        parsed = _parse_json_from_text(content)
    except RuntimeError:
        return _fallback_chunk_trace(row.reasoning_trace)

    chunks = _extract_chunks_from_parsed(parsed)
    if not chunks:
        return _fallback_chunk_trace(row.reasoning_trace)
    return chunks


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


def _extract_chunks_from_parsed(parsed: Any) -> list[TraceChunk]:
    if isinstance(parsed, dict):
        candidates = parsed.get("chunks")
        if isinstance(candidates, list):
            chunks = _parse_chunk_list(candidates)
            if chunks:
                return chunks
    if isinstance(parsed, list):
        chunks = _parse_chunk_list(parsed)
        if chunks:
            return chunks
    return []


def _parse_chunk_list(items: list[Any]) -> list[TraceChunk]:
    chunks: list[TraceChunk] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text_span = _normalize(item.get("text_span") or item.get("text") or item.get("span"))
        if not text_span:
            continue
        raw_span_type = _normalize(item.get("span_type") or item.get("type") or item.get("label"))
        span_type = raw_span_type.lower().replace("_", "-")
        if span_type not in {"reasoning", "noise"}:
            span_type = "noise" if _looks_like_noise(text_span) else "reasoning"
        chunks.append(TraceChunk(text_span=text_span, span_type=span_type))
    return _merge_adjacent_chunks(chunks)


def _merge_adjacent_chunks(chunks: list[TraceChunk]) -> list[TraceChunk]:
    merged: list[TraceChunk] = []
    for chunk in chunks:
        if merged and merged[-1].span_type == chunk.span_type:
            merged[-1] = TraceChunk(
                text_span=f"{merged[-1].text_span}\n{chunk.text_span}".strip(),
                span_type=chunk.span_type,
            )
        else:
            merged.append(chunk)
    return merged


def _fallback_chunk_trace(trace: str) -> list[TraceChunk]:
    cleaned = trace.strip()
    if not cleaned:
        return [TraceChunk(text_span="EMPTY_TRACE", span_type="noise")]

    raw_spans = [part.strip() for part in re.split(r"\n\s*\n+", cleaned) if part.strip()]
    if not raw_spans:
        raw_spans = [cleaned]

    chunks = [
        TraceChunk(
            text_span=span,
            span_type="noise" if _looks_like_noise(span) else "reasoning",
        )
        for span in raw_spans
    ]
    return _merge_adjacent_chunks(chunks)


def _looks_like_noise(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True

    suspicious_substrings = (
        "runtimeexception",
        "mongodb",
        "httpservletresponse",
        "@repository",
        "stringutils",
        "fixture",
        "onclick",
        "<style",
        "userid",
        "json",
        "hostname",
    )
    lowered = stripped.lower()
    if any(marker in lowered for marker in suspicious_substrings):
        return True

    total_chars = len(stripped)
    if total_chars == 0:
        return True

    alnum_chars = sum(1 for ch in stripped if ch.isalnum())
    symbol_chars = sum(1 for ch in stripped if not ch.isalnum() and not ch.isspace())
    long_garble = total_chars >= 120 and symbol_chars > alnum_chars
    return long_garble


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
<<<<<<< HEAD
                t = _assemble_qualified_token(step)
                if t:
                    tokens.append(t)
=======
                token = _format_typed_token_item(step)
                if token:
                    tokens.append(token)
>>>>>>> 3ec13d5d5029df1065fa0d04f9bc6696d6b0151d
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
<<<<<<< HEAD
                t = _assemble_qualified_token(item)
                if t:
                    tokens.append(t)
=======
                token = _format_typed_token_item(item)
                if token:
                    tokens.append(token)
>>>>>>> 3ec13d5d5029df1065fa0d04f9bc6696d6b0151d
        return " ".join(tokens)
    return None


def _format_typed_token_item(item: dict[str, Any]) -> str:
    token_name = ""
    for key in ("token", "canonical_token", "label", "name"):
        if key in item:
            token_name = _normalize(item.get(key))
            if token_name:
                break
    if not token_name:
        return ""

    token_type = ""
    for key in ("type", "token_type", "category", "kind"):
        if key in item:
            token_type = _normalize(item.get(key)).lower()
            if token_type:
                break

    if token_type in {"action", "strategy", "milestone", "noise"}:
        if token_name.startswith(f"{token_type}:"):
            return token_name
        return f"{token_type}:{token_name}"
    return token_name


def normalize_tokenized_trace(*, tokenized_trace: str, chunks: list[TraceChunk]) -> str:
    raw = _normalize(tokenized_trace)
    if not raw or raw == "MISSING":
        return _fallback_tokens_from_chunks(chunks)

    tokens: list[str] = []
    for piece in raw.split():
        normalized = _normalize_single_token(piece)
        if normalized:
            tokens.append(normalized)

    if not tokens:
        return _fallback_tokens_from_chunks(chunks)

    tokens = _compress_duplicate_runs(tokens)
    return " ".join(tokens)


def _normalize_single_token(token: str) -> str:
    cleaned = _normalize(token).lower().replace("_", "-")
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.strip(",.;")
    if not cleaned:
        return ""

    parts = [part for part in cleaned.split(":") if part]
    if not parts:
        return ""

    token_type = parts[0]
    if token_type not in {"action", "strategy", "milestone", "noise"}:
        return ""

    if token_type == "noise":
        return "noise:corrupted-span"

    normalized_parts = [_canonicalize_token_part(part) for part in parts[1:]]
    normalized_parts = [part for part in normalized_parts if part]
    if not normalized_parts:
        return ""
    return ":".join([token_type, *normalized_parts])


def _canonicalize_token_part(part: str) -> str:
    canonical = part
    aliases = {
        "applyformula": "apply-formula",
        "apply-formula": "apply-formula",
        "deriveintermediate": "derive-intermediate",
        "derive-intermediate": "derive-intermediate",
        "keyrelation": "key-relation",
        "key-relation": "key-relation",
        "choose-representation": "choose-representation",
        "chooserepresentation": "choose-representation",
        "setsubgoal": "set-subgoal",
        "set-subgoal": "set-subgoal",
        "stepwise": "plan-stepwise",
    }
    if canonical in aliases:
        canonical = aliases[canonical]
    canonical = re.sub(r"[^a-z0-9=+\-^]+", "-", canonical)
    canonical = re.sub(r"-{2,}", "-", canonical).strip("-")
    return canonical


def _compress_duplicate_runs(tokens: list[str]) -> list[str]:
    compressed: list[str] = []
    for token in tokens:
        if compressed and compressed[-1] == token:
            continue
        compressed.append(token)
    return compressed


def _fallback_tokens_from_chunks(chunks: list[TraceChunk]) -> str:
    fallback_tokens: list[str] = []
    for chunk in chunks:
        if chunk.span_type == "noise":
            fallback_tokens.append("noise:corrupted-span")
        else:
            fallback_tokens.append("action:analyze")
    fallback_tokens = _compress_duplicate_runs(fallback_tokens)
    return " ".join(fallback_tokens) if fallback_tokens else "noise:corrupted-span"


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
