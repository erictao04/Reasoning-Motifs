"""Shared helpers: run id, hashing, csv/json IO, git SHA, model name sanitizer."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def run_id() -> str:
    """Timestamp + short random id, e.g. ``20260425-180501-3f9a``."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{secrets.token_hex(2)}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_short_sha(repo_root: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root or Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv_rows(
    path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]
) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_model_suffix(model: str) -> str:
    tail = model.rsplit("/", 1)[-1].strip() or "model"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", tail)
    return re.sub(r"-{2,}", "-", safe).strip("-") or "model"


def normalize_str(value: Any) -> str:
    return str(value or "").strip()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def subset_by_questions(
    rows: list[dict[str, str]], max_questions: int | None
) -> list[dict[str, str]]:
    """Keep all rows whose question_id is among the first N unique ids.

    Order is the order of first appearance in ``rows``. ``None`` or
    non-positive values return the input unchanged.
    """
    if not max_questions or max_questions <= 0:
        return rows
    seen: list[str] = []
    seen_set: set[str] = set()
    for r in rows:
        qid = r.get("question_id", "")
        if qid not in seen_set:
            seen_set.add(qid)
            seen.append(qid)
            if len(seen) >= max_questions:
                break
    keep_set = set(seen)
    return [r for r in rows if r.get("question_id", "") in keep_set]
