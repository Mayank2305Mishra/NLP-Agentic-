from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "system2.yaml"


@dataclass
class Weights:
    """Calibrated on the retrieval fixture benchmark per spec §10 (sum = 1.00).

    Spec starting values were 0.70/0.10/0.08/0.07/0.05; calibration showed the
    inferred-metadata and saturated causal/outcome cues at conversation level
    were overpowering BM25 (hit@1 0.41 → 0.73 after re-weighting).
    """

    bm25: float = 0.85
    metadata: float = 0.05
    phrase: float = 0.05
    causal: float = 0.02
    outcome: float = 0.03


@dataclass
class DatasetConfig:
    path: str = "final_transcripts.json"
    processed_dir: str = "data/processed"
    index_dir: str = "data/index"
    sample_size: int = 8000
    sample_seed: int = 42
    stratification_key: str = "intent"

    @property
    def raw_path(self) -> Path:
        p = Path(self.path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def processed_path(self) -> Path:
        p = Path(self.processed_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def index_path(self) -> Path:
        p = Path(self.index_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def db_path(self) -> Path:
        return self.index_path / "lexical.sqlite"

    @property
    def conversations_jsonl(self) -> Path:
        return self.processed_path / "conversations.jsonl"

    @property
    def turns_jsonl(self) -> Path:
        return self.processed_path / "turns.jsonl"

    @property
    def sampling_manifest(self) -> Path:
        return self.processed_path / "sampling_manifest.json"


@dataclass
class RetrievalConfig:
    backend: str = "sqlite_fts5"
    conversation_text_mode: str = "full"
    conversation_pool: int = 50
    conversation_top_k: int = 3
    turns_per_conversation: int = 5
    turn_pool_per_conversation: int = 60
    neighbor_window: int = 2
    neighbor_window_max: int = 4
    query_expansion: bool = True
    metadata_boost: bool = True
    hard_filter_on_explicit_intent: bool = True
    causal_scoring: bool = True
    include_post_call: bool = False
    weights: Weights = field(default_factory=Weights)


@dataclass
class EvidenceConfig:
    task1_max_tokens: int = 1500
    task2_max_tokens: int = 2500
    min_score: float = 0.30
    min_conversation_score: float = 0.25
    expand_causal_gap: bool = True


@dataclass
class GenerationConfig:
    provider: str = "ollama"
    model: str = "qwen3.5:4b-mlx"
    temperature: float = 0.0
    num_ctx: int = 4096
    num_predict: int = 1400
    request_timeout_s: int = 300
    parse_retries: int = 1
    ollama_url: str = "http://localhost:11434"
    bonsai_url: str = ""
    bonsai_model: str = ""


@dataclass
class GroundingConfig:
    min_claim_support: float = 0.4
    reject_on_contamination: bool = True


@dataclass
class CacheConfig:
    enabled: bool = True
    max_entries: int = 256


@dataclass
class LoggingConfig:
    level: str = "INFO"
    path: str = "logs/system2.jsonl"
    debug_evidence: bool = False

    @property
    def log_path(self) -> Path:
        p = Path(self.path)
        return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class Config:
    system: str = "system2"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ablation: dict[str, Any] = field(default_factory=lambda: {"mode": "full"})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _build(cls: type, data: dict[str, Any]) -> Any:
    fields = {f for f in cls.__dataclass_fields__}
    kwargs: dict[str, Any] = {}
    for key, value in (data or {}).items():
        if key not in fields:
            continue
        if key == "weights" and isinstance(value, dict):
            kwargs[key] = _build(Weights, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config(
        system=raw.get("system", "system2"),
        dataset=_build(DatasetConfig, raw.get("dataset", {})),
        retrieval=_build(RetrievalConfig, raw.get("retrieval", {})),
        evidence=_build(EvidenceConfig, raw.get("evidence", {})),
        generation=_build(GenerationConfig, raw.get("generation", {})),
        grounding=_build(GroundingConfig, raw.get("grounding", {})),
        cache=_build(CacheConfig, raw.get("cache", {})),
        logging=_build(LoggingConfig, raw.get("logging", {})),
        ablation=raw.get("ablation", {"mode": "full"}),
    )
