"""LLM factory: local-first provider selection (AGENTS_SYSTEM_4.md §25, §41).

Selection order: Ollama (local) → Bonsai (local) → approved cloud endpoint
(only with explicit ``SYSTEM4_ALLOW_REMOTE=1``). Returns ``None`` when no
provider is configured, which engages the deterministic evidence-cited
fallback so the pipeline stays usable.
"""

from __future__ import annotations

import os

from system4.llm.base import GenerationResult, LLM, LLMError, MeasuredLLM, llm_identifier, measure
from system4.llm.structured import extract_json, parse_answer

__all__ = [
    "GenerationResult",
    "LLM",
    "LLMError",
    "MeasuredLLM",
    "extract_json",
    "llm_identifier",
    "make_llm",
    "measure",
    "parse_answer",
]


def make_llm(timeout: float = 90.0):
    """Build the configured provider, or ``None`` for deterministic mode."""
    ollama_model = os.getenv("OLLAMA_MODEL", "").strip()
    if ollama_model:
        from system4.llm.ollama import OllamaLLM

        return OllamaLLM(
            ollama_model,
            os.getenv("OLLAMA_URL", "http://localhost:11434").strip() or "http://localhost:11434",
            timeout=timeout,
        )
    bonsai_url = os.getenv("BONSAI_URL", "").strip()
    if bonsai_url:
        from system4.llm.bonsai import BonsaiLLM

        return BonsaiLLM(bonsai_url, os.getenv("BONSAI_MODEL", "").strip(), timeout=timeout)
    cloud_url = os.getenv("CLOUD_URL", "").strip()
    cloud_key = os.getenv("CLOUD_API_KEY", "").strip()
    cloud_model = os.getenv("CLOUD_MODEL", "").strip()
    if cloud_url and cloud_key and cloud_model:
        from system4.llm.cloud import CloudLLM, remote_allowed

        if remote_allowed():
            return CloudLLM(cloud_url, cloud_key, cloud_model, timeout=timeout)
        # §41: refuse to silently send transcripts to external APIs.
        import warnings

        warnings.warn(
            "CLOUD_* is configured but SYSTEM4_ALLOW_REMOTE is not '1'; "
            "staying local (deterministic fallback). Set SYSTEM4_ALLOW_REMOTE=1 to opt in.",
            RuntimeWarning,
            stacklevel=2,
        )
    return None
