"""Shared helpers: run id, hashing, csv/json IO, git SHA, model name sanitizer."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


_LOG_T0 = time.time()


def log(stage: str, msg: str) -> None:
    """Single-line stderr log with seconds-since-start prefix."""
    elapsed = time.time() - _LOG_T0
    print(f"[{elapsed:7.1f}s {stage:8}] {msg}", file=sys.stderr, flush=True)


def short_err(exc: BaseException | str, n: int = 200) -> str:
    """Squash exception/string into a single line, capped at ``n`` chars."""
    s = str(exc)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


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


def filter_mixed_outcome(
    rows: list[dict[str, str]], *, label_col: str = "is_correct"
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Drop rows whose question_id is uniformly correct or uniformly incorrect.

    Returns ``(kept_rows, drop_reasons)`` where ``drop_reasons`` maps each
    dropped question_id to ``"all_correct"`` or ``"all_incorrect"``.
    """
    by_q_correct: dict[str, list[bool]] = {}
    for r in rows:
        qid = r.get("question_id", "")
        by_q_correct.setdefault(qid, []).append(parse_bool(r.get(label_col)))
    drop_reasons: dict[str, str] = {}
    keep_qids: set[str] = set()
    for qid, labels in by_q_correct.items():
        if not labels:
            continue
        if all(labels):
            drop_reasons[qid] = "all_correct"
        elif not any(labels):
            drop_reasons[qid] = "all_incorrect"
        else:
            keep_qids.add(qid)
    kept = [r for r in rows if r.get("question_id", "") in keep_qids]
    return kept, drop_reasons
