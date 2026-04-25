#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reasoning_motifs_web.exporter import export_webapp_data
from reasoning_motifs_web.fixtures import (
    DEFAULT_RAW_TRACE_CSV,
    DEFAULT_TOKENIZED_TRACE_CSV,
    DEFAULT_WEBAPP_ARTIFACT_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export curated pilot traces into webapp-ready JSON artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_WEBAPP_ARTIFACT_DIR,
        help=f"Artifact output directory (default: {DEFAULT_WEBAPP_ARTIFACT_DIR}).",
    )
    parser.add_argument(
        "--tokenized-csv",
        type=Path,
        default=DEFAULT_TOKENIZED_TRACE_CSV,
        help=f"Tokenized trace CSV (default: {DEFAULT_TOKENIZED_TRACE_CSV}).",
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=DEFAULT_RAW_TRACE_CSV,
        help=f"Raw trace CSV (default: {DEFAULT_RAW_TRACE_CSV}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = export_webapp_data(
        args.output_dir,
        tokenized_csv=args.tokenized_csv,
        raw_csv=args.raw_csv,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
