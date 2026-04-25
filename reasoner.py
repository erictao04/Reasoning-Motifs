from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from reasoning_trace_sampling.api_shim import TogetherFatalRequestError, TogetherRateLimitError
from reasoning_trace_sampling import (
    APIShim,
    AdaptiveTraceCollector,
    BenchmarkItem,
    BenchmarkRegistry,
    LLMAnswerVerifier,
    ProgressReporter,
    QuestionTraceAnalyzer,
    ReasoningTraceSampling,
    RequestConfig,
    TraceCollector,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_API_BASE = "https://api.together.xyz/v1"
DEFAULT_API_KEY_ENV = "TOGETHER_API_KEY"
DEFAULT_SYSTEM_PROMPT = (
    "Solve the problem step by step in the visible response. "
    "Keep the reasoning concise but complete. "
    "End with the benchmark-specific final answer marker."
)
DEFAULT_OUTPUT = ROOT / "data" / "traces" / "together_qwen35_9b.csv"
DEFAULT_ADAPTIVE_OUTPUT = ROOT / "data" / "traces" / "adaptive_qwen35_9b.csv"
DEFAULT_STATS_OUTPUT = ROOT / "data" / "traces" / "question_stats.csv"


def build_request_config(args: argparse.Namespace) -> RequestConfig:
    return RequestConfig(
        model=args.model,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        enable_thinking=args.enable_thinking,
        timeout_seconds=args.timeout_seconds,
    )


def build_verifier_config(args: argparse.Namespace) -> RequestConfig:
    return RequestConfig(
        model=args.verifier_model or args.model,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        system_prompt=(
            "You extract and validate final answers from model responses. "
            "Return strict JSON only. Do not solve problems from scratch."
        ),
        temperature=args.verifier_temperature,
        top_p=1.0,
        max_tokens=args.verifier_max_tokens,
        enable_thinking=False,
        timeout_seconds=args.verifier_timeout_seconds,
    )


def build_components(args: argparse.Namespace) -> tuple[BenchmarkRegistry, ReasoningTraceSampling, TraceCollector]:
    config = build_request_config(args)
    api_shim = APIShim(config=config, repo_root=ROOT)
    answer_verifier = None
    if args.llm_validate_answers:
        verifier_api_shim = APIShim(config=build_verifier_config(args), repo_root=ROOT)
        answer_verifier = LLMAnswerVerifier(api_shim=verifier_api_shim)
    registry = BenchmarkRegistry(repo_root=ROOT)
    sampler = ReasoningTraceSampling(api_shim=api_shim, answer_verifier=answer_verifier)
    collector = TraceCollector(sampler=sampler)
    return registry, sampler, collector


def preflight_generator(sampler: ReasoningTraceSampling) -> None:
    sampler.api_shim.ask_question("Reply exactly: FINAL: 0", temperature=0.0)


def compact_together_error(message: str) -> str:
    match = re.search(r'"message"\s*:\s*"([^"]+)"', message)
    if match:
        return match.group(1)
    return re.sub(r"\s+", " ", message).strip()


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--temperature-schedule",
        default=None,
        help="Comma-separated per-attempt temperatures, e.g. 0.2,0.5,0.8. Overrides --temperature during collection.",
    )
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--llm-validate-answers",
        action="store_true",
        help="Use an LLM verifier as a fallback when deterministic answer parsing fails.",
    )
    parser.add_argument("--verifier-model", default=None, help="Verifier model. Defaults to --model.")
    parser.add_argument("--verifier-temperature", type=float, default=0.0)
    parser.add_argument("--verifier-max-tokens", type=int, default=256)
    parser.add_argument("--verifier-timeout-seconds", type=float, default=30.0)


def parse_temperature_schedule(raw_value: str | None) -> list[float] | None:
    if raw_value is None:
        return None
    values: list[float] = []
    for part in raw_value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = float(stripped)
        if value < 0:
            raise ValueError("--temperature-schedule values must be non-negative")
        values.append(value)
    if not values:
        raise ValueError("--temperature-schedule must contain at least one number")
    return values


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark-name", default="sample_math")
    parser.add_argument("--benchmark-path", type=Path, default=None)
    parser.add_argument("--min-level", type=int, default=None)
    parser.add_argument("--max-level", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=17)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modular Together-based reasoning trace runner."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmarks_parser = subparsers.add_parser("benchmarks", help="List benchmark presets.")
    add_shared_args(benchmarks_parser)

    call_parser = subparsers.add_parser("call", help="Make one minimal Together API call.")
    call_parser.add_argument("--question", required=True)
    add_shared_args(call_parser)

    one_parser = subparsers.add_parser("one", help="Ask a single benchmark question.")
    one_parser.add_argument("--question-index", type=int, default=0)
    one_parser.add_argument("--samples-per-question", type=int, default=1)
    one_parser.add_argument("--max-attempts-per-question", type=int, default=4)
    one_parser.add_argument("--show-progress", action="store_true")
    add_shared_args(one_parser)
    add_benchmark_args(one_parser)

    many_parser = subparsers.add_parser("many", help="Ask multiple benchmark questions efficiently.")
    many_parser.add_argument("--limit", type=int, default=None)
    many_parser.add_argument("--workers", type=int, default=8)
    many_parser.add_argument("--samples-per-question", type=int, default=1)
    many_parser.add_argument("--max-attempts-per-question", type=int, default=4)
    many_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    many_parser.add_argument("--log-jsonl", type=Path, default=None)
    many_parser.add_argument("--show-progress", action="store_true")
    add_shared_args(many_parser)
    add_benchmark_args(many_parser)

    adaptive_many_parser = subparsers.add_parser(
        "adaptive-many",
        help="Scout each question with a few traces, then densify only mixed-outcome questions.",
    )
    adaptive_many_parser.add_argument("--limit", type=int, default=None)
    adaptive_many_parser.add_argument("--workers", type=int, default=8)
    adaptive_many_parser.add_argument("--scout-samples", type=int, default=3)
    adaptive_many_parser.add_argument("--target-samples", type=int, default=10)
    adaptive_many_parser.add_argument("--scout-max-attempts-per-question", type=int, default=12)
    adaptive_many_parser.add_argument("--max-attempts-per-question", type=int, default=40)
    adaptive_many_parser.add_argument("--output", type=Path, default=DEFAULT_ADAPTIVE_OUTPUT)
    adaptive_many_parser.add_argument("--summary-output", type=Path, default=None)
    adaptive_many_parser.add_argument("--log-jsonl", type=Path, default=None)
    adaptive_many_parser.add_argument("--show-progress", action="store_true")
    add_shared_args(adaptive_many_parser)
    add_benchmark_args(adaptive_many_parser)

    stats_parser = subparsers.add_parser("stats", help="Summarize and filter questions from a trajectory CSV.")
    stats_parser.add_argument("--input", type=Path, required=True)
    stats_parser.add_argument("--output", type=Path, default=DEFAULT_STATS_OUTPUT)
    stats_parser.add_argument("--only-mixed", action="store_true")
    stats_parser.add_argument("--min-total", type=int, default=1)
    stats_parser.add_argument("--min-right", type=int, default=0)
    stats_parser.add_argument("--min-wrong", type=int, default=0)

    return parser.parse_args()


def run_benchmarks(args: argparse.Namespace) -> None:
    registry = BenchmarkRegistry(repo_root=ROOT)
    rows = []
    for preset in registry.list_presets():
        rows.append(
            {
                "name": preset.name,
                "path": str(preset.path),
                "exists": preset.path.exists(),
                "description": preset.description,
            }
        )
    print(json.dumps(rows, indent=2))


def run_call(args: argparse.Namespace) -> None:
    _registry, sampler, _collector = build_components(args)
    item = BenchmarkItem(question_id=0, question=args.question, gold_answer="")
    print(json.dumps(sampler.ask_one(item).to_dict(), indent=2))


def run_one(args: argparse.Namespace) -> None:
    registry, _sampler, collector = build_components(args)
    benchmark_path = registry.resolve(
        benchmark_name=args.benchmark_name,
        benchmark_path=args.benchmark_path,
    )
    answer_format = registry.resolve_answer_format(
        benchmark_name=args.benchmark_name,
        benchmark_path=args.benchmark_path,
    )
    items = registry.load_items(benchmark_path, answer_format=answer_format)
    items = registry.filter_items(
        items,
        min_level=args.min_level,
        max_level=args.max_level,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )
    if args.question_index < 0 or args.question_index >= len(items):
        raise IndexError(f"question-index {args.question_index} is out of range for {len(items)} questions.")
    progress_reporter = None
    if args.show_progress:
        progress_reporter = ProgressReporter(total_questions=1, total_target_samples=args.samples_per_question)
    rows = collector.collect_for_item(
        items[args.question_index],
        samples_per_question=args.samples_per_question,
        max_attempts_per_question=args.max_attempts_per_question,
        temperature_schedule=parse_temperature_schedule(args.temperature_schedule),
        progress_reporter=progress_reporter,
    )
    if progress_reporter is not None:
        attempts_used = rows[-1].attempt_index + 1 if len(rows) >= args.samples_per_question and rows else args.max_attempts_per_question
        progress_reporter.record_question_done(
            question_id=items[args.question_index].question_id,
            accepted_for_question=len(rows),
            samples_per_question=args.samples_per_question,
            attempts_used=attempts_used,
        )
        progress_reporter.finish()
    if not rows:
        raise RuntimeError(
            "No accepted trajectory found. "
            f"Tried {args.max_attempts_per_question} attempts and none produced a clear final answer marker."
        )
    if args.samples_per_question == 1:
        print(json.dumps(rows[0].to_dict(), indent=2))
    else:
        print(json.dumps([row.to_dict() for row in rows], indent=2))


def run_many(args: argparse.Namespace) -> None:
    registry, sampler, collector = build_components(args)
    benchmark_path = registry.resolve(
        benchmark_name=args.benchmark_name,
        benchmark_path=args.benchmark_path,
    )
    answer_format = registry.resolve_answer_format(
        benchmark_name=args.benchmark_name,
        benchmark_path=args.benchmark_path,
    )
    items = registry.load_items(benchmark_path, answer_format=answer_format)
    items = registry.filter_items(
        items,
        min_level=args.min_level,
        max_level=args.max_level,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )
    if args.limit is not None:
        items = items[: args.limit]

    preflight_generator(sampler)

    output_path = args.output.resolve()
    rows = collector.collect_many(
        items,
        samples_per_question=args.samples_per_question,
        max_attempts_per_question=args.max_attempts_per_question,
        workers=args.workers,
        show_progress=args.show_progress,
        log_path=args.log_jsonl.resolve() if args.log_jsonl is not None else None,
        temperature_schedule=parse_temperature_schedule(args.temperature_schedule),
        stream_output_path=output_path,
    )
    collector.save_csv(output_path, rows)

    correct = sum(1 for row in rows if row.is_correct)
    accepted = len(rows)
    target = len(items) * args.samples_per_question
    print(f"Saved {len(rows)} rows to {output_path}")
    if args.log_jsonl is not None:
        print(f"Saved event log to {args.log_jsonl.resolve()}")
    print(f"Accuracy: {correct}/{len(rows)}" if rows else "Accuracy: n/a (no accepted trajectories)")
    print(f"Accepted trajectories: {accepted}/{target}")
    if args.min_level is not None or args.max_level is not None or args.sample_size is not None:
        print(
            f"Question slice: levels {args.min_level if args.min_level is not None else '-inf'}"
            f" to {args.max_level if args.max_level is not None else '+inf'}, "
            f"sample_size={args.sample_size if args.sample_size is not None else 'all'}, "
            f"sample_seed={args.sample_seed}"
        )


def run_adaptive_many(args: argparse.Namespace) -> None:
    if args.scout_max_attempts_per_question > args.max_attempts_per_question:
        raise ValueError("--scout-max-attempts-per-question must be <= --max-attempts-per-question")
    if args.target_samples < args.scout_samples:
        raise ValueError("--target-samples must be >= --scout-samples")

    registry, sampler, collector = build_components(args)
    adaptive_collector = AdaptiveTraceCollector(trace_collector=collector)
    benchmark_path = registry.resolve(
        benchmark_name=args.benchmark_name,
        benchmark_path=args.benchmark_path,
    )
    answer_format = registry.resolve_answer_format(
        benchmark_name=args.benchmark_name,
        benchmark_path=args.benchmark_path,
    )
    items = registry.load_items(benchmark_path, answer_format=answer_format)
    items = registry.filter_items(
        items,
        min_level=args.min_level,
        max_level=args.max_level,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )
    if args.limit is not None:
        items = items[: args.limit]

    preflight_generator(sampler)

    output_path = args.output.resolve()
    summary_output_path = (
        args.summary_output.resolve()
        if args.summary_output is not None
        else output_path.with_name(f"{output_path.stem}_question_summary.csv")
    )

    rows, summaries = adaptive_collector.collect_many(
        items,
        scout_samples=args.scout_samples,
        target_samples=args.target_samples,
        scout_max_attempts_per_question=args.scout_max_attempts_per_question,
        max_attempts_per_question=args.max_attempts_per_question,
        workers=args.workers,
        show_progress=args.show_progress,
        log_path=args.log_jsonl.resolve() if args.log_jsonl is not None else None,
        temperature_schedule=parse_temperature_schedule(args.temperature_schedule),
        stream_output_path=output_path,
        summary_stream_output_path=summary_output_path,
    )

    collector.save_csv(output_path, rows)
    adaptive_collector.save_summary_csv(summary_output_path, summaries)

    correct = sum(1 for row in rows if row.is_correct)
    accepted = len(rows)
    max_target = len(items) * args.target_samples
    scout_only = sum(1 for row in summaries if not row.was_densified)
    densified = sum(1 for row in summaries if row.was_densified)
    mixed_final = sum(1 for row in summaries if row.right_final > 0 and row.wrong_final > 0)
    decision_counts: dict[str, int] = {}
    for row in summaries:
        decision_counts[row.decision] = decision_counts.get(row.decision, 0) + 1

    print(f"Saved {len(rows)} rows to {output_path}")
    print(f"Saved adaptive question summary to {summary_output_path}")
    if args.log_jsonl is not None:
        print(f"Saved event log to {args.log_jsonl.resolve()}")
    print(f"Accuracy: {correct}/{len(rows)}" if rows else "Accuracy: n/a (no accepted trajectories)")
    print(f"Accepted trajectories: {accepted}/{max_target}")
    print(f"Questions: total={len(summaries)} scout_only={scout_only} densified={densified} mixed_final={mixed_final}")
    print("Decisions: " + ", ".join(f"{key}={value}" for key, value in sorted(decision_counts.items())))
    if args.min_level is not None or args.max_level is not None or args.sample_size is not None:
        print(
            f"Question slice: levels {args.min_level if args.min_level is not None else '-inf'}"
            f" to {args.max_level if args.max_level is not None else '+inf'}, "
            f"sample_size={args.sample_size if args.sample_size is not None else 'all'}, "
            f"sample_seed={args.sample_seed}"
        )


def run_stats(args: argparse.Namespace) -> None:
    analyzer = QuestionTraceAnalyzer()
    rows = analyzer.load_stats(args.input.resolve())
    filtered = analyzer.filter_stats(
        rows,
        only_mixed=args.only_mixed,
        min_total=args.min_total,
        min_right=args.min_right,
        min_wrong=args.min_wrong,
    )
    analyzer.save_csv(args.output.resolve(), filtered)
    print(f"Saved {len(filtered)} question rows to {args.output.resolve()}")
    print(f"Loaded question stats from {args.input.resolve()}")


def main() -> None:
    args = parse_args()
    try:
        if args.command == "benchmarks":
            run_benchmarks(args)
            return
        if args.command == "call":
            run_call(args)
            return
        if args.command == "one":
            run_one(args)
            return
        if args.command == "many":
            run_many(args)
            return
        if args.command == "adaptive-many":
            run_adaptive_many(args)
            return
        if args.command == "stats":
            run_stats(args)
            return
        raise ValueError(f"Unsupported command: {args.command}")
    except TogetherFatalRequestError as exc:
        raise SystemExit(f"Fatal Together API configuration error: {compact_together_error(str(exc))}") from exc
    except TogetherRateLimitError as exc:
        raise SystemExit(f"Together API rate limit persisted after retries: {compact_together_error(str(exc))}") from exc
    except Exception as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
