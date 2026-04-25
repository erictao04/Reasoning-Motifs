"""Stage 0: LLM-based label audit.

For each row in the input CSV, ask a judge LLM whether the predicted answer
is mathematically equivalent to the gold answer (and whether the trace
actually concludes that predicted answer). Apply pre-registered decision
rules to drop / re-label / flag rows.

See ../docs/research_plan.md and DESIGN.md for the prompt contract and
decision rules.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ._common import (
    file_sha256,
    filter_mixed_outcome,
    fmt_duration,
    git_short_sha,
    log,
    parse_bool,
    read_csv_rows,
    safe_model_suffix,
    short_err,
    subset_by_questions,
    text_sha256,
    write_csv_rows,
    write_json,
)
from .llm_client import LLMClient


JUDGE_SYSTEM = """\
You are a strict but fair math grader. Given a question, the gold answer,
the model's predicted answer, and a short excerpt of the model's reasoning
trace, decide whether the predicted answer is mathematically equivalent to
the gold answer.

Be robust to formatting:
- fractions vs decimals (1/2 = 0.5)
- latex vs plain (\\frac{1}{2} = 1/2, 2\\sqrt{3} = 2*sqrt(3))
- equivalent algebraic forms
- ordering in unordered sets / pairs / tuples

Verdict legend:
- "correct"     : predicted answer matches gold under math equivalence
- "incorrect"   : predicted differs from gold under math equivalence
- "ambiguous"   : gold or predicted is genuinely ambiguous (multiple
                  acceptable forms, missing context)
- "non_attempt" : the trace does not seriously attempt the problem
                  (refusal, gibberish, blank)

Confidence in [0.0, 1.0]: reflect uncertainty about ambiguity, NOT about
formatting. If formatting is the only obstacle, you should be confident.

Set "trace_concludes_predicted" = true iff the trace excerpt clearly states
the predicted answer as its conclusion. If the trace concludes a different
answer than the predicted_answer field, set this to false.

Return JSON only, with this exact shape:
{
  "verdict": "correct|incorrect|ambiguous|non_attempt",
  "confidence": 0.0,
  "reason": "one short sentence",
  "trace_concludes_predicted": true
}
"""


REQUIRED_INPUT_COLS = (
    "question_id",
    "sample_id",
    "question",
    "gold_answer",
    "predicted_answer",
    "is_correct",
    "reasoning_trace",
)


AUDIT_OUTPUT_COLS = [
    "judge_verdict",
    "judge_confidence",
    "judge_reason",
    "judge_trace_concludes_predicted",
    "judge_model",
    "is_correct_original",
    "is_relabeled",
    "is_low_confidence_label",
    "keep",
    "drop_reason",
]


def _truncate_tail(text: str, n: int = 500) -> str:
    text = text or ""
    if len(text) <= n:
        return text
    return "..." + text[-n:]


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_user_payload(row: dict[str, str]) -> str:
    excerpt = (row.get("final_response_text") or "").strip()
    if not excerpt:
        excerpt = _truncate_tail(row.get("reasoning_trace") or "", 500)
    payload = {
        "question_id": row.get("question_id", ""),
        "question": row.get("question", ""),
        "gold_answer": row.get("gold_answer", ""),
        "predicted_answer": row.get("predicted_answer", ""),
        "trace_excerpt": excerpt,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def judge_one(client: LLMClient, row: dict[str, str]) -> dict[str, Any]:
    user = build_user_payload(row)
    cache_key = text_sha256(JUDGE_SYSTEM + "\n---\n" + user)
    try:
        return client.complete_json(
            system=JUDGE_SYSTEM, user=user, cache_key=cache_key, max_tokens=512
        )
    except Exception as exc:
        return {
            "verdict": "ambiguous",
            "confidence": 0.0,
            "reason": f"judge_error: {exc}",
            "trace_concludes_predicted": False,
        }


def apply_decision_rules(
    row: dict[str, str], verdict: dict[str, Any], confidence_threshold: float
) -> dict[str, str]:
    v = (verdict.get("verdict") or "ambiguous").strip().lower()
    confidence = _safe_float(verdict.get("confidence"), 0.0)
    concludes = bool(verdict.get("trace_concludes_predicted", False))
    original = parse_bool(row.get("is_correct"))

    keep = True
    drop_reason = ""

    if v == "non_attempt":
        keep = False
        drop_reason = "non_attempt"
    elif (not concludes) and confidence >= confidence_threshold:
        keep = False
        drop_reason = "extraction_failure"

    new_label = original
    is_relabeled = False
    is_low_confidence = False

    if keep:
        if v in ("correct", "incorrect") and confidence >= confidence_threshold:
            verdict_bool = (v == "correct")
            if verdict_bool != original:
                new_label = verdict_bool
                is_relabeled = True
        else:
            is_low_confidence = True

    return {
        "judge_verdict": v,
        "judge_confidence": f"{confidence:.3f}",
        "judge_reason": (verdict.get("reason") or "").strip(),
        "judge_trace_concludes_predicted": str(concludes),
        "is_correct_original": str(original),
        "is_correct": str(new_label),
        "is_relabeled": str(is_relabeled),
        "is_low_confidence_label": str(is_low_confidence),
        "keep": str(keep),
        "drop_reason": drop_reason,
    }


def run_audit(
    *,
    input_csv: Path,
    output_csv: Path,
    summary_path: Path,
    judge_model: str,
    cache_dir: Path,
    repo_root: Path,
    confidence_threshold: float = 0.8,
    max_workers: int = 4,
    max_questions: int | None = None,
    prefilter_mixed_outcome: bool = True,
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

    n_q_total = len({r.get("question_id", "") for r in rows})
    n_rows_total = len(rows)

    prefilter_drops: dict[str, str] = {}
    if prefilter_mixed_outcome:
        rows, prefilter_drops = filter_mixed_outcome(rows)
        n_drop_q = len(prefilter_drops)
        n_all_correct = sum(1 for v in prefilter_drops.values() if v == "all_correct")
        n_all_incorrect = sum(
            1 for v in prefilter_drops.values() if v == "all_incorrect"
        )
        log(
            "audit",
            f"prefilter mixed-outcome: kept {n_q_total - n_drop_q}/{n_q_total} questions, "
            f"{len(rows)}/{n_rows_total} rows  "
            f"(dropped {n_all_correct} all-correct, {n_all_incorrect} all-incorrect)",
        )
        if not rows:
            raise ValueError(
                "Prefilter removed every row: no mixed-outcome questions in input. "
                "Re-run with --no-prefilter to disable."
            )

    if max_questions:
        before = len(rows)
        rows = subset_by_questions(rows, max_questions)
        log("audit", f"limit max_questions={max_questions}: {len(rows)}/{before} rows")

    n_questions = len({r.get("question_id", "") for r in rows})
    log("audit", f"input  csv={input_csv}  rows={len(rows)}  questions={n_questions}")
    log(
        "audit",
        f"config judge={judge_model}  workers={max_workers}  "
        f"threshold={confidence_threshold}  cache={cache_dir}",
    )

    client = LLMClient(
        model=judge_model,
        cache_dir=cache_dir,
        repo_root=repo_root,
        api_base=api_base or "https://api.together.xyz/v1",
        api_key_env=api_key_env or "TOGETHER_API_KEY",
        max_tokens=512,
    )

    audited: list[dict[str, str]] = [{} for _ in rows]
    n_total = len(rows)
    progress_every = max(10, n_total // 10)
    first_error: str | None = None

    def task(idx: int) -> tuple[int, dict[str, str], str | None]:
        row = rows[idx]
        verdict = judge_one(client, row)
        err = None
        if (verdict.get("reason") or "").startswith("judge_error:"):
            err = verdict["reason"]
        flags = apply_decision_rules(row, verdict, confidence_threshold)
        flags["judge_model"] = judge_model
        merged: dict[str, str] = dict(row)
        merged.update(flags)
        return idx, merged, err

    n_errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(task, i) for i in range(n_total)]
        for j, f in enumerate(as_completed(futures), start=1):
            idx, merged, err = f.result()
            audited[idx] = merged
            if err:
                n_errors += 1
                if first_error is None:
                    first_error = err
                    log("audit", f"FIRST ERROR @ row {idx}: {short_err(err)}")
            if j == 1 or j % progress_every == 0 or j == n_total:
                elapsed = time.time() - t_start
                rate = j / elapsed if elapsed > 0 else 0
                eta = (n_total - j) / rate if rate > 0 else 0
                log(
                    "audit",
                    f"progress {j}/{n_total} ({100*j/n_total:.0f}%)  "
                    f"cache=h{client.cache_hits}/m{client.cache_misses}  "
                    f"errors={n_errors}  "
                    f"elapsed={fmt_duration(elapsed)}  "
                    f"eta={fmt_duration(eta)}",
                )

    n = len(audited)
    n_drop = sum(1 for r in audited if r["keep"] == "False")
    n_relabel = sum(1 for r in audited if r["is_relabeled"] == "True")
    n_low_conf = sum(1 for r in audited if r["is_low_confidence_label"] == "True")
    by_verdict: dict[str, int] = {}
    by_drop_reason: dict[str, int] = {}
    for r in audited:
        by_verdict[r["judge_verdict"]] = by_verdict.get(r["judge_verdict"], 0) + 1
        if r["keep"] == "False":
            by_drop_reason[r["drop_reason"]] = by_drop_reason.get(r["drop_reason"], 0) + 1

    fieldnames = list(rows[0].keys()) + [
        c for c in AUDIT_OUTPUT_COLS if c not in rows[0]
    ]
    write_csv_rows(output_csv, audited, fieldnames)

    summary = {
        "input_csv": str(input_csv),
        "input_sha256": file_sha256(input_csv),
        "output_csv": str(output_csv),
        "judge_model": judge_model,
        "n_rows_input": n_rows_total,
        "n_questions_input": n_q_total,
        "prefilter_mixed_outcome": prefilter_mixed_outcome,
        "prefilter_dropped_questions": len(prefilter_drops),
        "prefilter_dropped_breakdown": {
            "all_correct": sum(
                1 for v in prefilter_drops.values() if v == "all_correct"
            ),
            "all_incorrect": sum(
                1 for v in prefilter_drops.values() if v == "all_incorrect"
            ),
        },
        "n_rows": n,
        "n_drop": n_drop,
        "n_relabel": n_relabel,
        "n_low_confidence": n_low_conf,
        "by_verdict": by_verdict,
        "by_drop_reason": by_drop_reason,
        "cache_hits": client.cache_hits,
        "cache_misses": client.cache_misses,
        "git_sha": git_short_sha(repo_root),
        "confidence_threshold": confidence_threshold,
    }
    write_json(summary_path, summary)
    log(
        "audit",
        f"verdicts {dict(sorted(by_verdict.items()))}  "
        f"drops={n_drop} {dict(by_drop_reason) if by_drop_reason else ''}  "
        f"relabels={n_relabel}  low_conf={n_low_conf}",
    )
    log(
        "audit",
        f"DONE   {n} rows in {fmt_duration(time.time() - t_start)}  "
        f"cache=h{client.cache_hits}/m{client.cache_misses}  "
        f"errors={n_errors}",
    )
    log("audit", f"wrote  {output_csv}")
    log("audit", f"wrote  {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--judge-model", default="openai/gpt-oss-120b")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "cache" / "audit",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="If set, keep only the first N unique question_ids (smoke test).",
    )
    parser.add_argument(
        "--no-prefilter",
        action="store_true",
        help="Disable the mixed-outcome prefilter (default: prefilter on).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cache_dir = args.cache_dir / safe_model_suffix(args.judge_model)
    run_audit(
        input_csv=args.input.resolve(),
        output_csv=args.output.resolve(),
        summary_path=args.summary.resolve(),
        judge_model=args.judge_model,
        cache_dir=cache_dir,
        repo_root=repo_root,
        confidence_threshold=args.confidence_threshold,
        max_workers=args.max_workers,
        max_questions=args.max_questions,
        prefilter_mixed_outcome=not args.no_prefilter,
    )


if __name__ == "__main__":
    main()
