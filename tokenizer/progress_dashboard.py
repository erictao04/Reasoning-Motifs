#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tiny dashboard for tokenizer progress by question."
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="Input trace CSV used for tokenization.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Tokenized output CSV path.")
    parser.add_argument("--metadata-output", type=Path, required=True, help="Metadata JSONL output path.")
    parser.add_argument("--progress-log", type=Path, default=None, help="Optional progress JSONL path.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default: 8765).")
    parser.add_argument("--refresh-ms", type=int, default=2000, help="Refresh interval in ms (default: 2000).")
    return parser.parse_args()


def _safe_read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _safe_read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def _maybe_int(value: str) -> tuple[int, str]:
    try:
        return (0, str(int(value)))
    except ValueError:
        return (1, value)


def build_status(
    *,
    input_csv: Path,
    output_csv: Path,
    metadata_output: Path,
    progress_log: Path | None,
) -> dict[str, Any]:
    input_rows = _safe_read_csv_rows(input_csv)
    output_rows = _safe_read_csv_rows(output_csv)
    metadata_rows = _safe_read_jsonl(metadata_output)
    progress_rows = _safe_read_jsonl(progress_log) if progress_log is not None else []

    expected_by_q: dict[str, int] = defaultdict(int)
    for row in input_rows:
        expected_by_q[str(row.get("question_id", "")).strip()] += 1

    output_by_q: dict[str, int] = defaultdict(int)
    for row in output_rows:
        output_by_q[str(row.get("question_id", "")).strip()] += 1

    metadata_done_qids = {
        str(row.get("question_id", "")).strip()
        for row in metadata_rows
        if str(row.get("question_id", "")).strip()
    }

    progress_by_q: dict[str, dict[str, Any]] = {}
    run_started = False
    run_finished = False
    for event in progress_rows:
        name = str(event.get("event", "")).strip()
        if name == "run_started":
            run_started = True
            continue
        if name == "run_finished":
            run_finished = True
            continue
        qid = str(event.get("question_id", "")).strip()
        if not qid:
            continue
        rec = progress_by_q.setdefault(
            qid,
            {
                "stage": "pending",
                "chunked_count": 0,
                "tokenized_count": 0,
                "started": False,
                "done": False,
                "error": "",
                "updated_ts": None,
            },
        )
        rec["updated_ts"] = event.get("ts")
        if name == "question_started":
            rec["started"] = True
            rec["stage"] = "started"
        elif name == "trace_chunked":
            rec["started"] = True
            rec["stage"] = "chunking"
            rec["chunked_count"] = max(rec["chunked_count"], int(event.get("count", 0)))
        elif name == "metadata_ready":
            rec["started"] = True
            rec["stage"] = "metadata_ready"
        elif name == "trace_tokenized":
            rec["started"] = True
            rec["stage"] = "tokenizing"
            rec["tokenized_count"] = max(rec["tokenized_count"], int(event.get("count", 0)))
        elif name == "question_done":
            rec["done"] = True
            rec["stage"] = "done"
        elif name == "question_error":
            rec["stage"] = "error"
            rec["error"] = str(event.get("error", ""))

    question_rows: list[dict[str, Any]] = []
    for qid in sorted(expected_by_q, key=_maybe_int):
        expected = expected_by_q[qid]
        output_count = output_by_q.get(qid, 0)
        progress = progress_by_q.get(qid, {})
        chunked_count = int(progress.get("chunked_count", 0))
        tokenized_count = int(progress.get("tokenized_count", 0))
        done = bool(progress.get("done", False)) or (qid in metadata_done_qids) or (output_count >= expected and expected > 0)
        error = str(progress.get("error", "")).strip()
        if error:
            status = "error"
        elif done:
            status = "done"
        elif tokenized_count > 0:
            status = "tokenizing"
        elif chunked_count > 0 or progress.get("stage") in {"chunking", "metadata_ready", "started"}:
            status = "chunking"
        elif run_started:
            status = "queued"
        else:
            status = "pending"

        question_rows.append(
            {
                "question_id": qid,
                "expected_traces": expected,
                "chunked_traces": min(chunked_count, expected),
                "tokenized_traces": min(tokenized_count, expected),
                "written_traces": output_count,
                "status": status,
                "error": error,
                "updated_ts": progress.get("updated_ts"),
            }
        )

    counts = defaultdict(int)
    for row in question_rows:
        counts[row["status"]] += 1

    return {
        "run_started": run_started,
        "run_finished": run_finished,
        "generated_at": time.time(),
        "totals": {
            "questions": len(question_rows),
            "expected_traces": sum(row["expected_traces"] for row in question_rows),
            "written_traces": sum(row["written_traces"] for row in question_rows),
            "done_questions": counts["done"],
            "chunking_questions": counts["chunking"],
            "tokenizing_questions": counts["tokenizing"],
            "queued_questions": counts["queued"],
            "pending_questions": counts["pending"],
            "error_questions": counts["error"],
        },
        "questions": question_rows,
    }


def html_page(refresh_ms: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tokenizer Progress</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --border: #e5e7eb;
      --ok: #059669;
      --warn: #d97706;
      --info: #2563eb;
      --err: #dc2626;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
    }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 16px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 12px; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 10px; }}
    .k {{ font-size: 12px; color: var(--muted); }}
    .v {{ font-size: 22px; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--card); border: 1px solid var(--border); }}
    th, td {{ border-bottom: 1px solid var(--border); padding: 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f9fafb; position: sticky; top: 0; }}
    .status-done {{ color: var(--ok); font-weight: 700; }}
    .status-tokenizing {{ color: var(--info); font-weight: 700; }}
    .status-chunking {{ color: var(--warn); font-weight: 700; }}
    .status-queued, .status-pending {{ color: var(--muted); font-weight: 700; }}
    .status-error {{ color: var(--err); font-weight: 700; }}
    .small {{ color: var(--muted); font-size: 12px; margin: 8px 0; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h2>Tokenizer Progress</h2>
    <div id="meta" class="small">Loading...</div>
    <div id="cards" class="cards"></div>
    <table>
      <thead>
        <tr>
          <th>Question</th>
          <th>Status</th>
          <th>Expected</th>
          <th>Chunked</th>
          <th>Tokenized</th>
          <th>Written</th>
          <th>Error</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
  <script>
    const REFRESH_MS = {refresh_ms};
    function statusClass(value) {{
      return "status-" + value;
    }}
    function render(data) {{
      const meta = document.getElementById("meta");
      const t = data.totals;
      meta.textContent = `updated: ${{new Date(data.generated_at * 1000).toLocaleTimeString()}} | run_started=${{data.run_started}} run_finished=${{data.run_finished}}`;
      const cards = [
        ["Questions", t.questions],
        ["Expected Traces", t.expected_traces],
        ["Written Traces", t.written_traces],
        ["Done", t.done_questions],
        ["Chunking", t.chunking_questions],
        ["Tokenizing", t.tokenizing_questions],
        ["Queued", t.queued_questions],
        ["Error", t.error_questions],
      ];
      document.getElementById("cards").innerHTML = cards.map(([k, v]) => `<div class="card"><div class="k">${{k}}</div><div class="v">${{v}}</div></div>`).join("");
      document.getElementById("rows").innerHTML = data.questions.map((q) => `
        <tr>
          <td>${{q.question_id}}</td>
          <td class="${{statusClass(q.status)}}">${{q.status}}</td>
          <td>${{q.expected_traces}}</td>
          <td>${{q.chunked_traces}}</td>
          <td>${{q.tokenized_traces}}</td>
          <td>${{q.written_traces}}</td>
          <td>${{q.error || ""}}</td>
        </tr>
      `).join("");
    }}
    async function tick() {{
      try {{
        const response = await fetch("/api/status", {{ cache: "no-store" }});
        const data = await response.json();
        render(data);
      }} catch (err) {{
        document.getElementById("meta").textContent = "Failed to load status: " + err;
      }}
    }}
    tick();
    setInterval(tick, REFRESH_MS);
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    refresh_ms = max(500, args.refresh_ms)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/status":
                payload = build_status(
                    input_csv=args.input_csv,
                    output_csv=args.output_csv,
                    metadata_output=args.metadata_output,
                    progress_log=args.progress_log,
                )
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path in {"/", "/index.html"}:
                body = html_page(refresh_ms=refresh_ms).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Input: {args.input_csv}")
    print(f"Output: {args.output_csv}")
    print(f"Metadata: {args.metadata_output}")
    if args.progress_log is not None:
        print(f"Progress log: {args.progress_log}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

