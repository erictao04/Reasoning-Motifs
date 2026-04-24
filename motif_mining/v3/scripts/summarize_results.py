from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def top_preview(path: Path, n: int = 5) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df.head(n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize analysis outputs from one run directory")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.outdir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print("=== Summary ===")
        for key in ["num_traces", "num_success", "num_failure", "success_rate", "avg_trace_len"]:
            if key in summary:
                print(f"{key}: {summary[key]}")
        print()

    for name in [
        "skipgrams_success.csv",
        "skipgrams_failure.csv",
        "sequential_patterns_success.csv",
        "sequential_patterns_failure.csv",
        "rules_success.csv",
    ]:
        path = args.outdir / name
        if not path.exists():
            continue
        print(f"=== {name} ===")
        preview = top_preview(path, n=5)
        if preview.empty:
            print("(empty)")
        else:
            cols = [c for c in ["motif", "support_difference", "lift", "log_odds_ratio", "q_value"] if c in preview.columns]
            print(preview[cols].to_string(index=False))
        print()


if __name__ == "__main__":
    main()
