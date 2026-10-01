"""Pipeline gates built on the System-1 engine (batched, one call per stage)."""
from __future__ import annotations

from app.config import Settings
from app.domain.errors import System1UnavailableError
from app.system1.laya import LayaDecisionEngine
from app.system1.routing import get_engine
from app.system1.schemas import answer_questions, evidence_questions, routing_questions


def build_engine(settings: Settings) -> LayaDecisionEngine | None:
    if not settings.laya_enabled:
        return None
    engine = get_engine(settings.laya_mode, enabled=True)
    assert isinstance(engine, LayaDecisionEngine) or hasattr(engine, "decide")
    inner = getattr(engine, "_inner", engine)
    return inner if isinstance(inner, LayaDecisionEngine) else LayaDecisionEngine()


def routing_decision(engine, *, query_text: str, task: str,
                     domain=None, intent=None):
    """Single batched call: task routing + retrieval gating."""
    if engine is None:
        return {"task_type": task, "requires_metadata_filter": "no",
                "degraded": True, "latency_ms": 0.0}
    try:
        res = engine.decide(
            {"query_text": query_text, "task": task, "domain": domain, "intent": intent},
            routing_questions(),
        )
        return {"task_type": res.get("task_type", task),
                "requires_metadata_filter": res.get("requires_metadata_filter", "no"),
                "degraded": False, "latency_ms": res.latency_ms,
                "confidence": res.decisions.get("task_type").confidence
                if res.decisions.get("task_type") else 0.0}
    except System1UnavailableError:
        return {"task_type": task, "requires_metadata_filter": "no",
                "degraded": True, "latency_ms": 0.0}


def evidence_gate(engine, *, num_chunks: int, mean_score: float, query_text: str = ""):
    if engine is None:
        return {"evidence_sufficient": "sufficient" if num_chunks else "insufficient",
                "score": 0.5, "degraded": True, "latency_ms": 0.0}
    try:
        res = engine.decide(
            {"query_text": query_text, "num_chunks": num_chunks, "mean_score": mean_score},
            evidence_questions(),
        )
        return {"evidence_sufficient": res.get("evidence_sufficient", "insufficient"),
                "score": res.get("evidence_quality_score", 0.0),
                "degraded": False, "latency_ms": res.latency_ms}
    except System1UnavailableError:
        return {"evidence_sufficient": "sufficient" if num_chunks else "insufficient",
                "score": 0.5, "degraded": True, "latency_ms": 0.0}


def answer_gate(engine, *, num_cited: int, num_valid_ids: int):
    if engine is None:
        return {"answer_grounded": "grounded" if num_valid_ids else "unsupported",
                "degraded": True, "latency_ms": 0.0}
    try:
        res = engine.decide(
            {"num_cited": num_cited, "num_valid_ids": num_valid_ids},
            answer_questions(),
        )
        return {"answer_grounded": res.get("answer_grounded", "unsupported"),
                "degraded": False, "latency_ms": res.latency_ms}
    except System1UnavailableError:
        return {"answer_grounded": "grounded" if num_valid_ids else "unsupported",
                "degraded": True, "latency_ms": 0.0}
