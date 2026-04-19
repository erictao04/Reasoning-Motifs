from __future__ import annotations

import json
import os
from pathlib import Path
from urllib import error, request

from .data_classes import RequestConfig


class APIShim:
    DEFAULT_USER_AGENT = "reasoning-motifs/0.1"

    def __init__(self, config: RequestConfig, repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root
        self._load_dotenv(repo_root / ".env")
        self.api_key = os.getenv(config.api_key_env)
        if not self.api_key:
            raise EnvironmentError(
                f"Missing API key in environment variable {config.api_key_env}. "
                f"Set it in {repo_root / '.env'} or in your shell."
            )

    def ask_question(self, question: str) -> dict[str, object]:
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        payload = self._build_payload(question)
        return self._post_json(url, payload)

    def _build_payload(self, question: str) -> dict[str, object]:
        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": question},
            ],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "reasoning": {"enabled": self.config.enable_thinking},
        }

    def _post_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:
        req = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": self.DEFAULT_USER_AGENT,
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Together request failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Together request failed: {exc.reason}") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Together returned non-JSON output: {body[:500]}") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError(f"Expected a JSON object from Together, got: {type(parsed).__name__}")
        if "error" in parsed:
            raise RuntimeError(f"Together returned an error payload: {parsed['error']}")
        return parsed

    @staticmethod
    def _load_dotenv(path: Path) -> None:
        if not path.exists():
            return

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
