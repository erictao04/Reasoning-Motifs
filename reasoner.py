from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


TRACES_DIR = Path(__file__).resolve().parent / "data" / "traces"
DEFAULT_BENCHMARK = [
    {"question": "What is 12 + 7?", "answer": "19"},
    {"question": "If x + 3 = 10, what is x?", "answer": "7"},
    {"question": "What is 6 * 8?", "answer": "48"},
]


@dataclass
class TraceRecord:
    question_id: int
    question: str
    gold_answer: str
    predicted_answer: str
    reasoning_trace: str
    is_correct: bool


class PlaceholderQwenReasoner:
    """A lightweight stand-in for an open-source reasoning model.

    Replace this class with a real Qwen integration later, for example using
    `transformers` or a vLLM/OpenAI-compatible local server.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> None:
        self.model_name = model_name

    def generate(self, question: str) -> tuple[str, str]:
        """Return (reasoning_trace, predicted_answer).

        This placeholder uses tiny rule-based logic so the rest of the pipeline
        can be developed and tested before the real model is wired in.
        """
        reasoning = [f"Read problem: {question}"]

        add_match = re.fullmatch(r"What is (\d+) \+ (\d+)\?", question)
        mult_match = re.fullmatch(r"What is (\d+) \* (\d+)\?", question)
        linear_match = re.fullmatch(r"If x \+ (\d+) = (\d+), what is x\?", question)

        if add_match:
            a, b = map(int, add_match.groups())
            reasoning.extend([
                f"Instantiate values a={a}, b={b}.",
                f"Compute {a} + {b} = {a + b}.",
                "Conclude with the final answer.",
            ])
            return "\n".join(reasoning), str(a + b)

        if mult_match:
            a, b = map(int, mult_match.groups())
            reasoning.extend([
                f"Instantiate values a={a}, b={b}.",
                f"Compute {a} * {b} = {a * b}.",
                "Conclude with the final answer.",
            ])
            return "\n".join(reasoning), str(a * b)

        if linear_match:
            addend, total = map(int, linear_match.groups())
            x = total - addend
            reasoning.extend([
                f"Rewrite equation x + {addend} = {total} into x = {total} - {addend}.",
                f"Compute {total} - {addend} = {x}.",
                "Check the constraint by substitution.",
                f"Since {x} + {addend} = {total}, conclude x = {x}.",
            ])
            return "\n".join(reasoning), str(x)

        reasoning.extend([
            "Use a generic reasoning strategy.",
            "No real model is connected yet, so this answer is a placeholder.",
            "Conclude with UNKNOWN.",
        ])
        return "\n".join(reasoning), "UNKNOWN"


def ensure_data_dirs() -> None:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)


def get_next_run_index() -> int:
    existing = sorted(TRACES_DIR.glob("traces_*.csv"))
    max_idx = -1
    for path in existing:
        try:
            idx = int(path.stem.split("_")[-1])
            max_idx = max(max_idx, idx)
        except ValueError:
            continue
    return max_idx + 1


def load_benchmark() -> Sequence[dict[str, str]]:
    """Load benchmark items.

    Replace this function with a real benchmark loader later, e.g. GSM8K/MATH.
    """
    return DEFAULT_BENCHMARK


def evaluate_prediction(predicted_answer: str, gold_answer: str) -> bool:
    return predicted_answer.strip() == gold_answer.strip()


def generate_traces(model_name: str, benchmark: Iterable[dict[str, str]]) -> List[TraceRecord]:
    model = PlaceholderQwenReasoner(model_name=model_name)
    rows: List[TraceRecord] = []
    for idx, item in enumerate(benchmark):
        reasoning_trace, predicted_answer = model.generate(item["question"])
        rows.append(
            TraceRecord(
                question_id=idx,
                question=item["question"],
                gold_answer=item["answer"],
                predicted_answer=predicted_answer,
                reasoning_trace=reasoning_trace,
                is_correct=evaluate_prediction(predicted_answer, item["answer"]),
            )
        )
    return rows


def save_traces(run_index: int, rows: Sequence[TraceRecord]) -> Path:
    output_path = TRACES_DIR / f"traces_{run_index}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "question",
                "gold_answer",
                "predicted_answer",
                "reasoning_trace",
                "is_correct",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "question_id": row.question_id,
                    "question": row.question,
                    "gold_answer": row.gold_answer,
                    "predicted_answer": row.predicted_answer,
                    "reasoning_trace": row.reasoning_trace,
                    "is_correct": row.is_correct,
                }
            )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reasoning traces for a math benchmark.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Open-source model name placeholder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_data_dirs()
    run_index = get_next_run_index()
    benchmark = load_benchmark()
    rows = generate_traces(model_name=args.model, benchmark=benchmark)
    output_path = save_traces(run_index=run_index, rows=rows)
    print(f"Saved {len(rows)} traces to {output_path}")


if __name__ == "__main__":
    main()
