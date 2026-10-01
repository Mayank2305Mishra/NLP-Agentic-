"""Live smoke test — requires local Ollama with qwen3.5:4b-mlx. Opt-in only.

Run: RUN_LIVE_OLLAMA_TESTS=1 pytest -m live
Never required for ordinary unit tests.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live


@pytest.mark.skipif(os.getenv("RUN_LIVE_OLLAMA_TESTS") != "1", reason="live test opt-in only")
def test_ollama_live_health():
    from app.config import get_settings
    from app.generation.ollama_client import OllamaClient
    h = OllamaClient(get_settings()).health()
    assert h.get("model_present") is True, f"Ollama model missing: {h}"
