"""Typed, deterministic configuration for System 4.

All thresholds are initial operating values; benchmark-calibrate them
(AGENTS_SYSTEM_4.md §12, §19, §35 — never tune only on the eval set).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

#: Runtime-only fields that must not change retrieval/answer cache keys.
_RUNTIME_KEYS = ("cache_answers", "trace_enabled", "trace_dir", "trace_max_bytes")


@dataclass(frozen=True)
class Settings:
    # --- Broad retrieval (§8) -------------------------------------------------
    dense_k: int = 40
    lexical_k: int = 40
    rrf_k: int = 60
    hash_dimensions: int = 2048

    # --- Conversation consistency gate (§10) ---------------------------------
    min_consistency_score: float = 0.05
    consistency_keep_floor: int = 8

    # --- Reranking (§11, §12) ------------------------------------------------
    reranker: str = "feature"  # none | feature | cross_encoder
    rerank_k: int = 8
    w_semantic: float = 0.24
    w_lexical: float = 0.18
    w_entity: float = 0.12
    w_metadata: float = 0.16
    w_causal_density: float = 0.10
    w_action_density: float = 0.08
    w_outcome_density: float = 0.06
    w_fusion: float = 0.06
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_max_chars: int = 4096

    # --- Turn selection & expansion (§13, §14) -------------------------------
    turns_per_conversation: int = 4
    neighbor_window: int = 2
    selection_min_score: float = 0.0
    include_post_call: bool = False

    # --- Evidence budgets & compression (§3.4, §21) --------------------------
    evidence_token_budget_task1: int = 1200
    evidence_token_budget_task2: int = 2200
    token_estimator: str = "chars4"  # chars4 | words
    default_compression: str = "LIGHT"

    # --- Gates (§10, §23, §29, §30) ------------------------------------------
    min_rerank_score: float = 0.12
    min_grounding_ratio: float = 0.55
    min_claim_support: float = 0.35
    allow_cross_conversation: bool = False

    # --- Laya thresholds (§19) -----------------------------------------------
    laya_causal_coverage_expand: float = 0.25

    # --- Failure handling & caching (§43, §39) -------------------------------
    cache_answers: bool = True

    # --- Observability (§40) -------------------------------------------------
    trace_enabled: bool = True
    trace_dir: str = "artifacts/traces"
    trace_max_bytes: int = 5_000_000

    # --- LLM (§25, §26) ------------------------------------------------------
    llm_timeout_s: float = 90.0

    # --- Lexicons ------------------------------------------------------------
    causal_terms: tuple[str, ...] = field(
        default=(
            "because", "since", "therefore", "so", "due", "reason", "why", "caused",
            "problem", "issue", "unable", "could not", "refund", "cancel", "resolved",
            "fixed", "approved", "denied", "offer", "discount", "help", "confirm",
        )
    )

    def fingerprint(self) -> str:
        """Stable hash over answer-affecting configuration (§39)."""
        import hashlib

        data = asdict(self)
        for key in _RUNTIME_KEYS:
            data.pop(key, None)
        payload = json.dumps(data, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:16]
