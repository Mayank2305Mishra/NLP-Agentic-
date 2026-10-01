"""Central configuration. Precedence: defaults < .env/file < env vars < CLI args.

Streamlit must not mutate benchmark-affecting config at runtime; CLI args
override env explicitly at call sites.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


def default_source_path() -> Path:
    # AGENTS.md names final_transcript.json; the shipped dataset in this
    # workspace is final_transcripts.json. Support both deterministically.
    for name in ("final_transcript.json", "final_transcripts.json"):
        p = ROOT / name
        if p.exists():
            return p
    return ROOT / "final_transcript.json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    source_path: Path = default_source_path()  # type: ignore[assignment]
    generated_dir: Path = ROOT / "generated"

    # Embeddings
    embedding_backend: str = "hashing"  # hashing | sentence-transformers
    embedding_model: str = "hashing-768"
    embedding_dim: int = 768
    embedding_normalize: bool = True
    max_embedding_batch: int = 32

    # Vector store
    vector_store_type: str = "auto"  # auto | faiss | numpy

    # Chunking
    chunk_target_tokens: int = 450
    chunk_max_tokens: int = 600
    chunk_overlap_turns: int = 2
    chunk_include_metadata: bool = True

    # Retrieval / evidence
    retrieval_top_k: int = 8
    evidence_top_k: int = 8
    evidence_max_per_transcript: int = 3
    evidence_max_context_tokens: int = 6000

    # Laya System-1
    laya_enabled: bool = True
    laya_mode: str = "local"
    laya_allow_degraded: bool = True

    # Generation: local Ollama — the sole generator. Nothing leaves the laptop.
    generation_timeout_seconds: float = 180.0
    generation_connect_timeout_seconds: float = 5.0
    generation_max_retries: int = 2
    generation_temperature: float = 0.0
    generation_top_p: float = 1.0
    generation_max_tokens: int = 2000
    max_generation_concurrency: int = 1
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.5:4b-mlx"

    # Prompts
    prompt_version_task1: str = "v3"
    prompt_version_task2: str = "v1"

    # Runtime
    app_debug: bool = False
    allow_stale_index: bool = False
    log_level: str = "INFO"

    @property
    def index_dir(self) -> Path:
        return Path(self.generated_dir) / "index"

    @property
    def normalized_dir(self) -> Path:
        return Path(self.generated_dir) / "normalized"

    @property
    def traces_dir(self) -> Path:
        return Path(self.generated_dir) / "traces"

    @property
    def benchmarks_dir(self) -> Path:
        return Path(self.generated_dir) / "benchmarks"

    @property
    def cache_dir(self) -> Path:
        return Path(self.generated_dir) / "cache"

    def chunking_config(self) -> dict:
        return {
            "strategy": "turn_window",
            "target_tokens": self.chunk_target_tokens,
            "max_tokens": self.chunk_max_tokens,
            "overlap_turns": self.chunk_overlap_turns,
            "include_metadata": self.chunk_include_metadata,
        }

    def retrieval_config(self) -> dict:
        return {
            "top_k": self.retrieval_top_k,
            "evidence_top_k": self.evidence_top_k,
            "max_chunks_per_transcript": self.evidence_max_per_transcript,
            "max_context_tokens": self.evidence_max_context_tokens,
        }

    def generation_config(self) -> dict:
        return {
            "backend": "ollama",
            "base_url": self.ollama_base_url,
            "model": self.ollama_model,
            "timeout_seconds": self.generation_timeout_seconds,
            "connect_timeout_seconds": self.generation_connect_timeout_seconds,
            "max_retries": self.generation_max_retries,
            "temperature": self.generation_temperature,
            "top_p": self.generation_top_p,
            "max_tokens": self.generation_max_tokens,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Allow SOURCE_PATH env override (also honored by pydantic via `source_path`).
    override = os.getenv("SOURCE_PATH")
    s = Settings()
    if override:
        s.source_path = Path(override)
    return s


def reset_settings_cache() -> None:
    get_settings.cache_clear()
