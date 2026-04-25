"""Stage 1: granular ``basetype:qualifier`` tokenization.

Two-phase per question:
  1. metadata: one LLM call per question_id -> tokenization guide
  2. per-trace: one LLM call per trace -> ``basetype:qualifier`` sequence

Tokens are validated against a fixed ``BASETYPE_VOCAB``; malformed traces
are dropped (never silently accepted).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ._common import (
    file_sha256,
    fmt_duration,
    git_short_sha,
    log,
    read_csv_rows,
    safe_model_suffix,
    short_err,
    subset_by_questions,
    text_sha256,
    write_csv_rows,
    write_json,
)
from .llm_client import LLMClient


BASETYPE_VOCAB = (
    "analyze",
    "instantiate",
    "compute",
    "apply-formula",
    "rewrite",
    "check-constraint",
    "case-split",
    "backtrack",
    "conclude",
    "guess",
    "simplify",
    "compare",
    "derive-intermediate",
)
_BASETYPE_SET = frozenset(BASETYPE_VOCAB)


METADATA_SYSTEM = """\
You design a tokenization guide for canonicalizing free-form math reasoning
traces into qualified reasoning-action tokens of the form  basetype:qualifier.

Your output will later constrain a per-trace tokenizer.

Token format: every emitted token is  basetype:qualifier.
- basetype is drawn from the FIXED vocabulary below; do not invent new basetypes.
- qualifier is a short hyphenated noun phrase (1-3 words, lowercase) naming
  the mathematical object, concept, or sub-goal being acted on.

Two traces that perform the same reasoning function on different mathematical
content must produce different tokens.

Fixed basetype vocabulary:
- analyze, instantiate, compute, apply-formula, rewrite, check-constraint,
  case-split, backtrack, conclude, guess, simplify, compare, derive-intermediate

Suggested qualifier pool (extend if needed):
  setup, constraint, formula, equation, expression, variable, value, bound,
  parity, modular, geometric, algebraic, combinatorial, proof, case, identity,
  sum, product, sequence, recurrence, inequality, probability, counting,
  substitution, factorization, simplification, intermediate-result, final-answer

Return JSON only, with this schema:
{
  "guide_name": "string",
  "version": "string",
  "qualifier_rules": ["string"],
  "ambiguity_rules": [
    {"pair": ["t1","t2"], "decision_rule": "string", "examples": ["string"]}
  ],
  "segmentation_rules": ["string"],
  "normalization_rules": ["string"],
  "notes_for_tokenizer": ["string"]
}
"""


TOKENIZE_SYSTEM = """\
You tokenize ONE free-form math reasoning trace into a sequence of qualified
reasoning-action tokens of the form  basetype:qualifier  using the supplied
guide.

Rules:
- Use ONLY the fixed basetype vocabulary; never invent a new basetype.
- Every output token MUST contain exactly one ":" with a non-empty basetype
  and qualifier.
- Tokenize by reasoning FUNCTION, not wording.
- Do not use answer correctness when assigning tokens.
- Preserve order. If one sentence performs multiple reasoning functions,
  emit multiple tokens.
- Ignore filler, hedging, politeness, and stylistic language unless it
  changes reasoning function.

Fixed basetype vocabulary:
- analyze, instantiate, compute, apply-formula, rewrite, check-constraint,
  case-split, backtrack, conclude, guess, simplify, compare, derive-intermediate

Return JSON only:
{"tokenized_trace": "basetype1:qualifier1 basetype2:qualifier2 ..."}

If no valid tokenization is possible, return  {"tokenized_trace": "MISSING"}.
"""


REQUIRED_INPUT_COLS = (
    "question_id",
    "sample_id",
    "gold_answer",
    "predicted_answer",
    "is_correct",
    "reasoning_trace",
)


OUTPUT_COLS = [
    "question_id",
    "sample_id",
    "gold_answer",
    "predicted_answer",
    "is_correct",
    "tokenized_trace",
]


# Cap how much of each trace flows into the metadata prompt to bound tokens.
# (per-trace tokenization receives the FULL trace.)
_METADATA_TRACE_PREVIEW_CHARS = 2000
_METADATA_MAX_TRACES_IN_PROMPT = 6


def build_metadata_user(
    question_id: str, rows: list[dict[str, str]], question_text: str
) -> str:
    sample_payload = []
    for r in rows[:_METADATA_MAX_TRACES_IN_PROMPT]:
        sample_payload.append(
            {
                "sample_id": r.get("sample_id", ""),
                "reasoning_trace": (r.get("reasoning_trace") or "")[:_METADATA_TRACE_PREVIEW_CHARS],
            }
        )
    payload = {
        "question_id": question_id,
        "question": question_text,
        "sample_traces": sample_payload,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_tokenize_user(row: dict[str, str], metadata: dict[str, Any]) -> str:
    payload = {
        "guide": metadata,
        "question_id": row.get("question_id", ""),
        "sample_id": row.get("sample_id", ""),
        "reasoning_trace": row.get("reasoning_trace", ""),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def is_valid_token(tok: str) -> bool:
    if not tok or tok.count(":") != 1:
        return False
    basetype, qualifier = tok.split(":", 1)
    if basetype not in _BASETYPE_SET:
        return False
    if not qualifier:
        return False
    return True


def parse_tokenized(parsed: dict[str, Any]) -> tuple[str, list[str]]:
    raw = parsed.get("tokenized_trace")
    if isinstance(raw, list):
        toks = [str(t).strip() for t in raw if str(t).strip()]
    elif isinstance(raw, str):
        toks = [t for t in raw.strip().split() if t]
    else:
        toks = []
    if not toks:
        return "MISSING", []
    if any(t == "MISSING" for t in toks):
        return "MISSING", []
    invalid = [t for t in toks if not is_valid_token(t)]
    if invalid:
        return "MISSING", invalid
    return " ".join(toks), []


def run_tokenize(
    *,
    input_csv: Path,
    output_csv: Path,
    summary_path: Path,
    tokenizer_model: str,
    cache_dir: Path,
    repo_root: Path,
    keep_only_label: bool = True,
    max_workers: int = 4,
    max_questions: int | None = None,
    api_base: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    t_start = time.time()
    rows = read_csv_rows(input_csv)
    if not rows:
        raise ValueError(f"No rows in {input_csv}")
    missing = [c for c in REQUIRED_INPUT_COLS if c not in rows[0]]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if keep_only_label and "keep" in rows[0]:
        before = len(rows)
        rows = [r for r in rows if r.get("keep", "True") == "True"]
        log("tokenize", f"filter kept {len(rows)}/{before} rows after audit-drop")

    if max_questions:
        before = len(rows)
        rows = subset_by_questions(rows, max_questions)
        log("tokenize", f"limit max_questions={max_questions}: {len(rows)}/{before} rows")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        grouped[r["question_id"]].append(r)

    client = LLMClient(
        model=tokenizer_model,
        cache_dir=cache_dir,
        repo_root=repo_root,
        api_base=api_base or "https://api.together.xyz/v1",
        api_key_env=api_key_env or "TOGETHER_API_KEY",
        max_tokens=8192,
    )

    log("tokenize", f"input  csv={input_csv}  rows={len(rows)}  questions={len(grouped)}")
    log(
        "tokenize",
        f"config model={tokenizer_model}  workers={max_workers}  cache={cache_dir}",
    )

    metadata_by_q: dict[str, dict[str, Any]] = {}
    n_q = len(grouped)
    n_meta_ok = 0
    n_meta_fail = 0
    log("tokenize", f"phase  metadata: 1 LLM call per question ({n_q} total)")
    for i, (qid, q_rows) in enumerate(grouped.items(), start=1):
        question_text = q_rows[0].get("question", "")
        user = build_metadata_user(qid, q_rows, question_text)
        cache_key = text_sha256(METADATA_SYSTEM + "\n---\n" + user)
        try:
            metadata_by_q[qid] = client.complete_json(
                system=METADATA_SYSTEM,
                user=user,
                cache_key=cache_key,
                cache_subdir="metadata",
                max_tokens=4096,
            )
            n_meta_ok += 1
            log(
                "tokenize",
                f"meta   {i}/{n_q} qid={qid} OK   "
                f"cache=h{client.cache_hits}/m{client.cache_misses}",
            )
        except Exception as exc:
            n_meta_fail += 1
            metadata_by_q[qid] = {"error": str(exc)}
            log("tokenize", f"meta   {i}/{n_q} qid={qid} FAIL {short_err(exc)}")
    log(
        "tokenize",
        f"phase  metadata done: {n_meta_ok} ok, {n_meta_fail} fail "
        f"in {fmt_duration(time.time() - t_start)}",
    )

    output_rows: list[dict[str, str]] = []
    dropped: list[dict[str, Any]] = []

    def task(row: dict[str, str]) -> dict[str, Any]:
        qid = row["question_id"]
        metadata = metadata_by_q.get(qid, {})
        user = build_tokenize_user(row, metadata)
        cache_key = text_sha256(TOKENIZE_SYSTEM + "\n---\n" + user)
        try:
            parsed = client.complete_json(
                system=TOKENIZE_SYSTEM,
                user=user,
                cache_key=cache_key,
                cache_subdir="per_trace",
                max_tokens=4096,
            )
        except Exception as exc:
            return {
                "row": row,
                "tokenized": "MISSING",
                "drop_reason": f"tokenize_error: {exc}",
                "invalid_tokens": [],
            }
        tokenized, invalid = parse_tokenized(parsed)
        if invalid:
            drop_reason = "invalid_tokens"
        elif tokenized == "MISSING":
            drop_reason = "missing"
        else:
            drop_reason = ""
        return {
            "row": row,
            "tokenized": tokenized,
            "drop_reason": drop_reason,
            "invalid_tokens": invalid,
        }

    n_total = len(rows)
    progress_every = max(10, n_total // 10)
    t_phase = time.time()
    log("tokenize", f"phase  per-trace: 1 LLM call per row ({n_total} total)")
    first_error: str | None = None
    n_errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(task, r) for r in rows]
        for j, f in enumerate(as_completed(futures), start=1):
            res = f.result()
            row = res["row"]
            out = {
                "question_id": row.get("question_id", ""),
                "sample_id": row.get("sample_id", ""),
                "gold_answer": row.get("gold_answer", ""),
                "predicted_answer": row.get("predicted_answer", ""),
                "is_correct": row.get("is_correct", ""),
                "tokenized_trace": res["tokenized"],
            }
            if res["tokenized"] == "MISSING":
                dropped.append(
                    {
                        "question_id": out["question_id"],
                        "sample_id": out["sample_id"],
                        "drop_reason": res["drop_reason"],
                        "invalid_tokens": res["invalid_tokens"][:10],
                    }
                )
                if res["drop_reason"].startswith("tokenize_error"):
                    n_errors += 1
                    if first_error is None:
                        first_error = res["drop_reason"]
                        log(
                            "tokenize",
                            f"FIRST ERROR qid={out['question_id']} "
                            f"sid={out['sample_id']}: {short_err(first_error)}",
                        )
            else:
                output_rows.append(out)
            if j == 1 or j % progress_every == 0 or j == n_total:
                elapsed = time.time() - t_phase
                rate = j / elapsed if elapsed > 0 else 0
                eta = (n_total - j) / rate if rate > 0 else 0
                log(
                    "tokenize",
                    f"trace  {j}/{n_total} ({100*j/n_total:.0f}%)  "
                    f"ok={len(output_rows)} drop={len(dropped)} err={n_errors}  "
                    f"cache=h{client.cache_hits}/m{client.cache_misses}  "
                    f"elapsed={fmt_duration(elapsed)}  eta={fmt_duration(eta)}",
                )

    write_csv_rows(output_csv, output_rows, OUTPUT_COLS)

    per_q_stats: dict[str, dict[str, int]] = {}
    for r in output_rows:
        s = per_q_stats.setdefault(
            r["question_id"], {"total": 0, "success": 0, "failure": 0}
        )
        s["total"] += 1
        if str(r["is_correct"]).strip().lower() == "true":
            s["success"] += 1
        else:
            s["failure"] += 1
    n_mixed = sum(
        1 for s in per_q_stats.values() if s["success"] > 0 and s["failure"] > 0
    )

    summary = {
        "input_csv": str(input_csv),
        "input_sha256": file_sha256(input_csv),
        "output_csv": str(output_csv),
        "tokenizer_model": tokenizer_model,
        "n_input_rows": len(rows),
        "n_output_rows": len(output_rows),
        "n_dropped": len(dropped),
        "n_questions": len(grouped),
        "n_mixed_outcome_questions": n_mixed,
        "cache_hits": client.cache_hits,
        "cache_misses": client.cache_misses,
        "dropped_traces": dropped[:200],
        "git_sha": git_short_sha(repo_root),
    }
    write_json(summary_path, summary)

    drop_reasons = Counter(d["drop_reason"].split(":", 1)[0] for d in dropped)
    n_complete_q = sum(1 for s in per_q_stats.values() if s["total"] >= 1)
    log(
        "tokenize",
        f"per-q  questions_with_output={n_complete_q}/{len(grouped)}  "
        f"mixed_outcome|M|={n_mixed}",
    )
    if drop_reasons:
        log("tokenize", f"drops  {dict(drop_reasons)}")
    log(
        "tokenize",
        f"DONE   rows={len(output_rows)}/{len(rows)}  "
        f"in {fmt_duration(time.time() - t_start)}  "
        f"cache=h{client.cache_hits}/m{client.cache_misses}  "
        f"errors={n_errors}",
    )
    log("tokenize", f"wrote  {output_csv}")
    log("tokenize", f"wrote  {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--tokenizer-model", default="openai/gpt-oss-120b")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "cache" / "tokenize",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="If set, keep only the first N unique question_ids (smoke test).",
    )
    parser.add_argument(
        "--include-flagged",
        action="store_true",
        help="Include rows where keep=False (audit drop list).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cache_dir = args.cache_dir / safe_model_suffix(args.tokenizer_model)
    run_tokenize(
        input_csv=args.input.resolve(),
        output_csv=args.output.resolve(),
        summary_path=args.summary.resolve(),
        tokenizer_model=args.tokenizer_model,
        cache_dir=cache_dir,
        repo_root=repo_root,
        keep_only_label=not args.include_flagged,
        max_workers=args.max_workers,
        max_questions=args.max_questions,
    )


if __name__ == "__main__":
    main()
