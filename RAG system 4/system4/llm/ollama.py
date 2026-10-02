"""Local Ollama adapter (AGENTS_SYSTEM_4.md §25, §26, §32).

Streams responses so time-to-first-token is measured rather than assumed,
keeps temperature 0 for deterministic evaluation, and reports token usage
when the server provides it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from system4.llm.base import GenerationResult, LLMError


class OllamaLLM:
    def __init__(self, model: str, url: str = "http://localhost:11434", timeout: float = 90.0) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.identifier = f"ollama:{model}"

    def generate(self, prompt: str, **kwargs: object) -> str:
        return self.generate_measured(prompt, **kwargs).text

    def generate_measured(self, prompt: str, **kwargs: object) -> GenerationResult:
        timeout = float(kwargs.get("timeout", self.timeout) or self.timeout)
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "format": "json",
                "options": {"temperature": 0},
            }
        ).encode()
        request = urllib.request.Request(
            self.url + "/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        chunks: list[str] = []
        ttft_ms = 0.0
        input_tokens = output_tokens = 0
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = event.get("response") or ""
                    if piece:
                        if not chunks:
                            ttft_ms = round((time.perf_counter() - started) * 1000, 2)
                        chunks.append(piece)
                    if event.get("prompt_eval_count"):
                        input_tokens = int(event["prompt_eval_count"])
                    if event.get("eval_count"):
                        output_tokens = int(event["eval_count"])
                    if event.get("done"):
                        break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(f"ollama request failed: {exc}") from exc
        generation_ms = round((time.perf_counter() - started) * 1000, 2)
        if not chunks:
            raise LLMError("ollama returned an empty response")
        return GenerationResult(
            text="".join(chunks),
            ttft_ms=ttft_ms,
            generation_ms=generation_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            streaming=True,
        )
