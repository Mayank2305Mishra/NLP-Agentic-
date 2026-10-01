"""Canonical application pipeline (AGENTS.md section 49).

Streamlit, CLI, benchmark harness, and tests all call QueryService —
no duplicated pipeline logic anywhere else.

Flow: normalize query -> Laya routing (batched) -> dense retrieve ->
evidence pack -> Laya evidence gate -> Ollama task prompt ->
parse/validate (1 repair retry) -> grounding -> Laya answer gate -> trace.
"""
from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any

from app.config import Settings
from app.domain.errors import (
    ConfigurationError,
    ContextBudgetError,
    RAGError,
    System1UnavailableError,
)
from app.domain.models import EvidencePack, Task1Result, Task2Result
from app.indexing.builder import verify_manifest
from app.logging import get_logger, log_event
from app.observability.metrics import REGISTRY, RequestMetrics
from app.observability.redaction import redact_dict
from app.observability.trace import RequestTrace, Timer
from app.retrieval.dense import DenseRetriever
from app.retrieval.evidence import build_evidence_pack
from app.retrieval.query import normalize_query
from app.system1 import gates as system1_gates
from app.tasks.task1_extraction import run_task1
from app.tasks.task2_synthesis import run_task2

log = get_logger("service")

SYSTEM_LABEL = "Traditional Dense Vector RAG"
SYSTEM1_LABEL = "Laya (local)"


def llm_label(model: str) -> str:
    """Model-accurate label — never hard-code a model name that may not match .env."""
    return f"Ollama {model} (local)" if model else "Ollama (local)"


class QueryService:
    def __init__(self, settings: Settings, *, retriever=None,
                 generator=None, laya_engine=None):
        self.settings = settings
        self._retriever = retriever
        self._generator = generator
        self._laya = laya_engine
        self._laya_ready: bool | None = None

    # -- lazy deps (keeps import light; index verified on first use) --------
    @property
    def retriever(self) -> DenseRetriever:
        if self._retriever is None:
            self._retriever = DenseRetriever(self.settings)
        return self._retriever

    @property
    def generator(self):
        if self._generator is None:
            from app.generation.ollama_client import OllamaClient
            self._generator = OllamaClient(self.settings)
        return self._generator

    @property
    def laya(self):
        if self._laya is None and self._laya_ready is None:
            try:
                eng = system1_gates.build_engine(self.settings)
                if eng is not None:
                    eng.ping()
                self._laya = eng
                self._laya_ready = True
            except System1UnavailableError:
                self._laya_ready = False
        return self._laya

    def manifest_info(self) -> dict:
        try:
            m = verify_manifest(self.settings, allow_stale=True)
            return m.model_dump()
        except RAGError as e:
            return {"error": e.to_dict()}

    # -- main entry ----------------------------------------------------------
    def run(self, *, task: str, query: str, domain=None, intent=None,
            top_k: int | None = None, save_trace: bool = True) -> dict[str, Any]:
        trace = RequestTrace()
        trace.start("request")
        laya_ms = 0.0
        metrics = RequestMetrics(request_id=trace.request_id, task=task)
        result: dict[str, Any] = {"request_id": trace.request_id, "task": task,
                                  "success": False}
        try:
            with Timer(trace, "query_normalization"):
                rq = normalize_query(query, task=task, domain=domain, intent=intent,
                                     top_k=top_k or self.settings.retrieval_top_k)
            laya_ms = 0.0
            with Timer(trace, "system1_routing"):
                routing = system1_gates.routing_decision(
                    self.laya, query_text=rq.text, task=rq.task,
                    domain=rq.domain, intent=rq.intent)
                laya_ms += trace.stage_ms("system1_routing")
            if routing.get("degraded") and not self.settings.laya_allow_degraded:
                raise System1UnavailableError("Laya unavailable and degraded mode is disabled")
            candidates = self.retriever.retrieve(rq, trace)
            metrics.embedding_ms = trace.stage_ms("embedding")
            metrics.retrieval_ms = trace.stage_ms("vector_search")
            pack: EvidencePack | None = None
            gate: dict = {}
            with Timer(trace, "evidence_pack"):
                try:
                    pack = build_evidence_pack(candidates, self.settings)
                except ContextBudgetError as e:
                    # No evidence fits / nothing retrieved: fail closed with a
                    # typed insufficient result (never an unsupported generation).
                    payload = self._insufficient(task, None, reason="insufficient_context")
                    result.update(payload)
                    result["success"] = True
                    result["evidence"] = []
                    result["evidence_truncated"] = False
                    result["routing"] = routing
                    gate = {"evidence_sufficient": "insufficient", "score": 0.0,
                            "degraded": False, "reason": "insufficient_context"}
                    result["evidence_gate"] = gate
                    result["context_budget_error"] = e.to_dict()
            if pack is None:
                pass  # insufficient_context already recorded above
            else:
                metrics.num_chunks = len(pack.chunks)
                metrics.context_tokens = pack.context_tokens
                mean_score = (statistics.mean([c.score for c in pack.chunks])
                              if pack.chunks else 0.0)
                with Timer(trace, "system1_evidence_gate"):
                    gate = system1_gates.evidence_gate(
                        self.laya, num_chunks=len(pack.chunks),
                        mean_score=mean_score, query_text=rq.text)
                    laya_ms += trace.stage_ms("system1_evidence_gate")
                if gate["evidence_sufficient"] == "insufficient":
                    # Fail closed: do not hallucinate; return typed insufficient result.
                    payload = self._insufficient(task, pack, reason="evidence_gate_insufficient")
                    result.update(payload)
                    result["success"] = True
                    result["degraded"] = {"laya_degraded": routing.get("degraded", False)}
                else:
                    with Timer(trace, "generation"):
                        payload = self._generate(task, rq.text, pack, trace)
                    metrics.generation_ms = trace.stage_ms_total("generation_request")
                    metrics.validation_ms = (trace.stage_ms_total("output_parse")
                                             + trace.stage_ms_total("grounding_validation"))
                    result.update(payload)
                    result["success"] = True
                    # Answer gate (advisory classification, recorded in trace).
                    with Timer(trace, "system1_answer_gate"):
                        cited = self._count_cited(payload.get("result"))
                        ag = system1_gates.answer_gate(
                            self.laya, num_cited=cited,
                            num_valid_ids=self._count_valid(payload.get("result"), pack))
                        laya_ms += trace.stage_ms("system1_answer_gate")
                        result["answer_gate"] = ag
                    metrics.retry_count = payload.get("info", {}).get("retries", 0)
                    metrics.validation_failures = payload.get("info", {}).get("validation_failures", 0)
                result["evidence"] = [c.model_dump() for c in pack.chunks]
                result["evidence_truncated"] = pack.truncated
                result["routing"] = routing
                result["evidence_gate"] = gate
        except RAGError as e:
            trace.end("request")
            metrics.success = False
            metrics.error_code = e.code
            result.update({"success": False, "error": e.to_dict()})
            log_event(log, logging.WARNING, "request.failed",
                      extra={"request_id": trace.request_id, "code": e.code})
        except Exception:  # never leak stack traces to callers
            trace.end("request")
            metrics.success = False
            metrics.error_code = "internal"
            result.update({"success": False, "error": {
                "code": "internal", "message": "Internal error processing request.",
                "correlation_id": trace.request_id}})
            log_event(log, logging.ERROR, "request.internal_error",
                      extra={"request_id": trace.request_id})
        finally:
            try:
                trace.end("request")
            except Exception:
                pass
            metrics.total_ms = trace.total_ms()
            metrics.laya_ms = laya_ms
            if not metrics.generation_ms:
                # Generation failed before/while returning: still report the
                # time actually spent in the generator instead of 0.0.
                metrics.generation_ms = trace.stage_ms_total("generation_request")
            REGISTRY.record(metrics)
            result["trace"] = trace.to_dict()
            result["metrics"] = {"total_ms": metrics.total_ms,
                                 "retrieval_ms": metrics.retrieval_ms,
                                 "embedding_ms": metrics.embedding_ms,
                                 "generation_ms": metrics.generation_ms,
                                 "num_chunks": metrics.num_chunks,
                                 "context_tokens": metrics.context_tokens}
            result["system"] = {"pipeline": SYSTEM_LABEL, "llm": llm_label(self.settings.ollama_model),
                                "system1": SYSTEM1_LABEL,
                                "index": self.manifest_info().get("index_type", "unknown"),
                                "generation": self.settings.generation_config(),
                                "prompt_versions": {"task1": self.settings.prompt_version_task1,
                                                    "task2": self.settings.prompt_version_task2}}
            if save_trace:
                self._save_trace(trace.request_id, result, pack_chunks_for_log=False)
        return result

    def _generate(self, task: str, query_text: str, pack: EvidencePack,
                  trace: RequestTrace) -> dict:
        if task == "task1":
            parsed, info = run_task1(settings=self.settings, client=self.generator,
                                     query_text=query_text, pack=pack, trace=trace)
            return {"result": parsed.model_dump(), "info": info}
        if task == "task2":
            # Task2 consumes validated Task-1 output (never hallucinates standalone).
            t1, info1 = run_task1(settings=self.settings, client=self.generator,
                                  query_text=query_text, pack=pack, trace=trace)
            if t1.insufficient_evidence:
                return {"result": Task2Result(
                    synthesis="", key_causes=[], business_effects=[],
                    supporting_evidence=[], uncertainties=["Insufficient evidence in Task-1 extraction."],
                    confidence=0.0).model_dump(),
                    "info": {**info1, "task1_insufficient": True},
                    "task1": t1.model_dump()}
            parsed, info = run_task2(settings=self.settings, client=self.generator,
                                     query_text=query_text, pack=pack, task1=t1, trace=trace)
            return {"result": parsed.model_dump(), "info": info, "task1": t1.model_dump()}
        raise ConfigurationError(f"Unknown task '{task}'")

    def _insufficient(self, task: str, pack: EvidencePack, reason: str) -> dict:
        if task == "task1":
            res = Task1Result(causal_rationales=[], confidence=0.0,
                              insufficient_evidence=True).model_dump()
        else:
            res = Task2Result(synthesis="", key_causes=[], business_effects=[],
                              supporting_evidence=[],
                              uncertainties=[f"Insufficient evidence ({reason})."],
                              confidence=0.0).model_dump()
        return {"result": res, "info": {"insufficient_evidence": True, "reason": reason}}

    @staticmethod
    def _count_cited(result: dict | None) -> int:
        if not isinstance(result, dict):
            return 0
        if "causal_rationales" in result:
            return sum(len(r.get("evidence", [])) for r in result.get("causal_rationales", []))
        return len(result.get("supporting_evidence", []))

    @staticmethod
    def _count_valid(result: dict | None, pack: EvidencePack) -> int:
        if not isinstance(result, dict):
            return 0
        valid = {c.chunk_id for c in pack.chunks}
        if "causal_rationales" in result:
            return sum(1 for r in result.get("causal_rationales", [])
                       for e in r.get("evidence", []) if e.get("source_id") in valid)
        return sum(1 for s in result.get("supporting_evidence", []) if s in valid)

    def _save_trace(self, request_id: str, result: dict, *, pack_chunks_for_log: bool) -> None:
        try:
            d = Path(self.settings.traces_dir)
            d.mkdir(parents=True, exist_ok=True)
            safe = dict(result)
            # Redact conversation-derived text in stored traces; keep IDs/scores.
            if isinstance(safe.get("evidence"), list):
                safe["evidence"] = [redact_dict(c) if isinstance(c, dict) else c
                                    for c in safe["evidence"]]
            if isinstance(safe.get("result"), dict):
                safe["result"] = redact_dict(safe["result"])
            (d / f"{request_id}.json").write_text(
                json.dumps(safe, default=str)[:200_000], encoding="utf-8")
        except Exception:
            pass
