"""Together-compatible chat completion client with on-disk JSON cache.

Single class; stdlib only. Cache key is the SHA-256 of (system + user) so
identical prompts at the same model are free on subsequent runs.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from threading import Lock
from typing import Any
from urllib import error, request

from ._common import ensure_dir, read_json, write_json


DEFAULT_API_BASE = "https://api.together.xyz/v1"
DEFAULT_API_KEY_ENV = "TOGETHER_API_KEY"
DEFAULT_TIMEOUT = 300.0
USER_AGENT = "reasoning-motifs-tokenizer-v5/0.1"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class LLMClient:
    """Minimal Together-compatible chat client.

    The cache layout is ``<cache_dir>/<cache_subdir>/<sha256>.json``. A
    ``cache_subdir`` of ``""`` puts entries directly under ``cache_dir``.
    """

    def __init__(
        self,
        *,
        model: str,
        cache_dir: Path,
        api_base: str = DEFAULT_API_BASE,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        repo_root: Path | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        if repo_root is not None:
            _load_dotenv(repo_root / "tokenizer_v5" / ".env")
            _load_dotenv(repo_root / "tokenizer" / ".env")
            _load_dotenv(repo_root / ".env")
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise EnvironmentError(
                f"Missing API key in environment variable {api_key_env}."
            )
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.cache_dir = ensure_dir(cache_dir)
        self._counter_lock = Lock()
        self.cache_hits = 0
        self.cache_misses = 0

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        cache_key: str,
        cache_subdir: str = "",
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        cache_path = self.cache_dir / cache_subdir / f"{cache_key}.json"
        if cache_path.exists():
            try:
                cached = read_json(cache_path)
                with self._counter_lock:
                    self.cache_hits += 1
                return cached["parsed"]
            except (json.JSONDecodeError, KeyError):
                cache_path.unlink(missing_ok=True)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False,
        }
        url = f"{self.api_base}/chat/completions"
        req = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Together HTTP {exc.code}: {detail[:500]}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"Together request failed: {exc.reason}") from exc

        raw_response = json.loads(body)
        text = _extract_text(raw_response)
        parsed = _parse_json(text)
        write_json(cache_path, {"parsed": parsed, "raw_text": text})
        with self._counter_lock:
            self.cache_misses += 1
        return parsed


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("Together response has no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return str(content or "")


def _parse_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise RuntimeError("Empty model response.")
    parsed: Any = None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
        if m:
            try:
                parsed = json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                parsed = None
    if parsed is None:
        a, b = stripped.find("{"), stripped.rfind("}")
        if a != -1 and b != -1 and b > a:
            try:
                parsed = json.loads(stripped[a : b + 1])
            except json.JSONDecodeError:
                parsed = None
    if parsed is None:
        raise RuntimeError(f"Could not parse JSON from model output:\n{text[:500]}")
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object, got {type(parsed).__name__}")
    return parsed
