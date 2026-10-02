"""Approved cloud adapter: any OpenAI-compatible chat-completions endpoint
(Groq, OpenAI, etc.) — AGENTS_SYSTEM_4.md §25, §41.

Privacy gate: never used unless the operator explicitly opts in with
``SYSTEM4_ALLOW_REMOTE=1``. Local adapters are preferred for confidential
transcripts.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from system4.llm.base import GenerationResult, LLMError


def remote_allowed() -> bool:
    return os.getenv("SYSTEM4_ALLOW_REMOTE", "").strip() == "1"


class CloudLLM:
    def __init__(
        self,
        url: str,
        api_key: str,
        model: str,
        timeout: float = 90.0,
        temperature: float = 0.0,
    ) -> None:
        if not remote_allowed():
            raise LLMError("remote LLM disabled: set SYSTEM4_ALLOW_REMOTE=1 to opt in (§41)")
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.identifier = f"cloud:{model}"

    def generate(self, prompt: str, **kwargs: object) -> str:
        return self.generate_measured(prompt, **kwargs).text

    def generate_measured(self, prompt: str, **kwargs: object) -> GenerationResult:
        timeout = float(kwargs.get("timeout", self.timeout) or self.timeout)
        endpoint = self.url
        if not endpoint.endswith("/chat/completions"):
            endpoint = endpoint + "/chat/completions"
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "stream": False,
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            raise LLMError(f"cloud endpoint returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LLMError(f"cloud request failed: {exc}") from exc
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("cloud response missing choices[0].message.content") from exc
        if not isinstance(text, str) or not text:
            raise LLMError("cloud returned an empty response")
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        usage = payload.get("usage") or {}
        return GenerationResult(
            text=text,
            ttft_ms=0.0,
            generation_ms=elapsed,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            streaming=False,
        )
