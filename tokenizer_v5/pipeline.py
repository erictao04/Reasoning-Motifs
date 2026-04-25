"""End-to-end pipeline: audit -> tokenize -> manifest."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ._common import (
    ensure_dir,
    file_sha256,
    git_short_sha,
    run_id,
    safe_model_suffix,
    write_json,
)
from .audit import run_audit
from .tokenize import run_tokenize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw expanded_pool*.csv")
    parser.add_argument("--judge-model", default="openai/gpt-oss-120b")
    parser.add_argument("--tokenizer-model", default="openai/gpt-oss-120b")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="If set, keep only the first N unique question_ids (smoke test).",
    )
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-tokenize", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    rid = args.run_id or run_id()
    results_dir = ensure_dir(args.results_root / rid)
    audit_cache = (
        Path(__file__).resolve().parent / "cache" / "audit"
        / safe_model_suffix(args.judge_model)
    )
    tokenize_cache = (
        Path(__file__).resolve().parent / "cache" / "tokenize"
        / safe_model_suffix(args.tokenizer_model)
    )

    audited_csv = results_dir / "audited_traces.csv"
    audit_summary = results_dir / "audit_summary.json"
    tokenized_csv = results_dir / "tokenized_traces.csv"
    tokenize_summary = results_dir / "tokenize_summary.json"

    audit_result = None
    tokenize_result = None
    audit_seconds = None
    tokenize_seconds = None

    if not args.skip_audit:
        t0 = time.time()
        audit_result = run_audit(
            input_csv=args.input.resolve(),
            output_csv=audited_csv,
            summary_path=audit_summary,
            judge_model=args.judge_model,
            cache_dir=audit_cache,
            repo_root=repo_root,
            confidence_threshold=args.confidence_threshold,
            max_workers=args.max_workers,
            max_questions=args.max_questions,
        )
        audit_seconds = round(time.time() - t0, 2)
        tokenize_input = audited_csv
    else:
        tokenize_input = args.input.resolve()

    if not args.skip_tokenize:
        t1 = time.time()
        tokenize_result = run_tokenize(
            input_csv=tokenize_input,
            output_csv=tokenized_csv,
            summary_path=tokenize_summary,
            tokenizer_model=args.tokenizer_model,
            cache_dir=tokenize_cache,
            repo_root=repo_root,
            max_workers=args.max_workers,
            max_questions=args.max_questions if args.skip_audit else None,
        )
        tokenize_seconds = round(time.time() - t1, 2)

    manifest = {
        "run_id": rid,
        "input_csv": str(args.input.resolve()),
        "input_sha256": file_sha256(args.input.resolve()),
        "judge_model": args.judge_model,
        "tokenizer_model": args.tokenizer_model,
        "confidence_threshold": args.confidence_threshold,
        "max_workers": args.max_workers,
        "results_dir": str(results_dir),
        "audit_seconds": audit_seconds,
        "tokenize_seconds": tokenize_seconds,
        "git_sha": git_short_sha(repo_root),
        "audit_summary": audit_result,
        "tokenize_summary": tokenize_result,
    }
    write_json(results_dir / "manifest.json", manifest)
    print(f"\n[pipeline] DONE  run_id={rid}")
    print(f"  results: {results_dir}")
    if audit_result:
        print(
            f"  audit:    drop={audit_result['n_drop']}  "
            f"relabel={audit_result['n_relabel']}  "
            f"low_conf={audit_result['n_low_confidence']}"
        )
    if tokenize_result:
        print(
            f"  tokenize: rows={tokenize_result['n_output_rows']}/"
            f"{tokenize_result['n_input_rows']}  "
            f"|M|={tokenize_result['n_mixed_outcome_questions']}"
        )


if __name__ == "__main__":
    main()
