from __future__ import annotations

import json
import logging
import time

import requests

from . import (
    DEFAULT_CONNECT_RETRIES,
    RETRY_BACKOFF_S,
    GenerationResult,
    pooled_session,
)

log = logging.getLogger("system3.llm.ollama")


class PartialStreamError(RuntimeError):
    """Connection lost after tokens were delivered — unsafe to retry."""


class OllamaClient:
    provider = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3.5:4b-mlx",
        connect_retries: int = DEFAULT_CONNECT_RETRIES,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.connect_retries = max(0, connect_retries)
        self._session = pooled_session()

    def generate(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        num_predict: int = 1400,
        timeout_s: int = 300,
    ) -> GenerationResult:
        attempts = self.connect_retries + 1
        for attempt in range(attempts):
            try:
                return self._stream_once(
                    system, prompt, temperature, num_predict, timeout_s
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt + 1 >= attempts:
                    raise
                delay = RETRY_BACKOFF_S * (attempt + 1)
                log.warning(
                    "ollama unreachable (%s); retry %d/%d in %.2fs",
                    type(exc).__name__,
                    attempt + 1,
                    self.connect_retries,
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError("unreachable")  # pragma: no cover

    def _stream_once(
        self,
        system: str,
        prompt: str,
        temperature: float,
        num_predict: int,
        timeout_s: int,
    ) -> GenerationResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "format": "json",
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        begin = time.perf_counter()
        ttft_ms: float | None = None
        chunks: list[str] = []
        prompt_tokens: int | None = None
        output_tokens: int | None = None
        prompt_eval_ms: float | None = None
        eval_ms: float | None = None

        try:
            with self._session.post(
                f"{self.base_url}/api/chat", json=payload, stream=True, timeout=timeout_s
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if ttft_ms is None:
                        ttft_ms = round((time.perf_counter() - begin) * 1000, 3)
                    content = (data.get("message") or {}).get("content")
                    if content:
                        chunks.append(content)
                    if data.get("prompt_eval_count") is not None:
                        prompt_tokens = data["prompt_eval_count"]
                    if data.get("eval_count") is not None:
                        output_tokens = data["eval_count"]
                    if data.get("prompt_eval_duration") is not None:
                        prompt_eval_ms = round(data["prompt_eval_duration"] / 1e6, 3)
                    if data.get("eval_duration") is not None:
                        eval_ms = round(data["eval_duration"] / 1e6, 3)
                    if data.get("done"):
                        break
        except (requests.ConnectionError, requests.Timeout) as exc:
            if chunks:
                raise PartialStreamError(
                    f"stream interrupted after {len(chunks)} chunk(s): {exc}"
                ) from exc
            raise

        generation_ms = round((time.perf_counter() - begin) * 1000, 3)
        return GenerationResult(
            text="".join(chunks),
            model=self.model,
            provider=self.provider,
            ttft_ms=ttft_ms,
            generation_ms=generation_ms,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            prompt_eval_ms=prompt_eval_ms,
            eval_ms=eval_ms,
        )
