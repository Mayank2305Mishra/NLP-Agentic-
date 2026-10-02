"""Bonsai local-endpoint adapter (AGENTS_SYSTEM_4.md §25).

Local-first: only ever talks to a configured localhost/LAN endpoint. The
response parser accepts the common local-serving shapes (``response``,
``text``, ``completion``, ``choices[0].text``, ``choices[0].message.content``)
so minor server differences never break generation.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from system4.llm.base import GenerationResult, LLMError


def _extract_text(payload: object) -> str:
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return ""
    for key in ("response", "text", "completion", "output", "result"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    usage = payload.get("usage")
    if isinstance(usage, dict):
        return ""
    return ""


class BonsaiLLM:
    def __init__(self, url: str, model: str = "", timeout: float = 90.0) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.identifier = f"bonsai:{model or self.url}"

    def generate(self, prompt: str, **kwargs: object) -> str:
        return self.generate_measured(prompt, **kwargs).text

    def generate_measured(self, prompt: str, **kwargs: object) -> GenerationResult:
        timeout = float(kwargs.get("timeout", self.timeout) or self.timeout)
        endpoint = self.url
        if not endpoint.endswith(("/generate", "/v1/completions", "/v1/chat/completions", "/api/generate")):
            endpoint = endpoint + "/generate"
        body = {"prompt": prompt, "temperature": 0, "stream": False}
        if self.model:
            body["model"] = self.model
        if endpoint.endswith("/v1/chat/completions"):
            body = {
                "model": self.model or "bonsai",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "stream": False,
            }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LLMError(f"bonsai request failed: {exc}") from exc
        text = _extract_text(payload)
        if not text:
            raise LLMError("bonsai returned an empty response")
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        usage = payload.get("usage") if isinstance(payload, dict) else None
        return GenerationResult(
            text=text,
            ttft_ms=0.0,  # non-streaming endpoint: TTFT not observable, never fabricated
            generation_ms=elapsed,
            input_tokens=int(usage.get("prompt_tokens", 0)) if isinstance(usage, dict) else 0,
            output_tokens=int(usage.get("completion_tokens", 0)) if isinstance(usage, dict) else 0,
            streaming=False,
        )
