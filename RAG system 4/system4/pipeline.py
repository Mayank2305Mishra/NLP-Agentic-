"""Canonical System 4 pipeline (AGENTS_SYSTEM_4.md §4, §52).

``query → normalize → task route → broad retrieval → consistency gate →
reranker → anti-contamination gate → turn selection → Laya → (expand) →
compression → evidence pack → prompt → LLM → structured parse → grounding →
causal/consistency validation → answer gate → answer|abstain``

Every stage is timed independently (§32), failures degrade through the
documented fallbacks (§43), and the trace never contains transcript text (§41).
"""

from __future__ import annotations

import copy
import time
import uuid
from typing import Callable

from system4.cache import AnswerCache, answer_cache_key
from system4.config.prompts import PROMPT_VERSION, prompt as build_prompt
from system4.config.settings import Settings
from system4.decision.laya import Laya
from system4.evidence.compressor import compress
from system4.evidence.gate import EvidenceFeatures, compute_features
from system4.evidence.packer import estimate_tokens, pack
from system4.evidence.selection import expand, select
from system4.llm import llm_identifier, make_llm, measure, parse_answer
from system4.observability import TraceLogger
from system4.retrieval.broad_retrieval import retrieve
from system4.retrieval.index import LocalIndex
from system4.retrieval.metadata import apply_gate
from system4.reranking.reranker import make_reranker
from system4.schema import (
    CHAIN_ACTION_SIDE,
    CHAIN_CAUSE_SIDE,
    CHAIN_OUTCOME_SIDE,
    SYSTEM,
    SYSTEM_VERSION,
    AnswerGate,
    CompressionLevel,
    Decision,
    Evidence,
    EvidenceAction,
    Result,
    TaskRoute,
)
from system4.validation.answer_gate import GateOutcome, final_gate
from system4.validation.causal import validate_causal_shape
from system4.validation.consistency import transcript_consistency
from system4.validation.grounding import validate as validate_grounding

_TASKS = {"task1", "task2", "auto"}


class QualityRAG:
    def __init__(
        self,
        index: LocalIndex,
        settings: Settings | None = None,
        llm: object | None = None,
        *,
        source_hash: str = "",
        reranker: object | None = None,
        laya: Laya | None = None,
        trace_logger: TraceLogger | None = None,
    ) -> None:
        self.index = index
        self.settings = settings or Settings()
        self.source_hash = source_hash
        self.llm = make_llm(self.settings.llm_timeout_s) if llm is None else llm
        self.reranker = reranker or make_reranker(self.settings)
        self.laya = laya or Laya()
        self.cache = AnswerCache() if self.settings.cache_answers else None
        self.traces = trace_logger or TraceLogger.from_settings(self.settings)

    # ------------------------------------------------------------------ ask --
    def ask(
        self,
        query: str,
        task: str = "task1",
        cross_conversation: bool = False,
        detail: str = "detailed",
        debug: bool = False,
    ) -> Result:
        if task not in _TASKS:
            raise ValueError("task must be task1, task2, or auto")
        if not query.strip():
            raise ValueError("query must not be empty")
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        stamps: dict[str, float] = {}
        errors: list[dict[str, str]] = []

        def stage(name: str, fn: Callable[[], object], default: object = None) -> object:
            point = time.perf_counter()
            try:
                value = fn()
            except Exception as exc:  # §43: degrade, never silently fabricate
                errors.append({"stage": name, "error": f"{type(exc).__name__}: {exc}"})
                value = default
            stamps[name + "_ms"] = round((time.perf_counter() - point) * 1000, 2)
            return value

        raw_query = query
        query = str(stage("query_normalization", lambda: " ".join(query.lower().split()), "") or "")

        # --- task routing (§4, §18) ---------------------------------------
        if task == "auto":
            task = str(stage("task_route", lambda: self.laya.route(raw_query), "task1") or "task1")

        # --- answer cache (§39) -------------------------------------------
        cache_key = None
        if self.cache is not None:
            cache_key = answer_cache_key(
                source_hash=self.source_hash,
                system_version=SYSTEM_VERSION,
                query=query,
                task=task,
                cross_conversation=cross_conversation,
                detail=detail,
                settings_fingerprint=self.settings.fingerprint(),
                reranker=getattr(self.reranker, "name", "unknown"),
                prompt_version=PROMPT_VERSION,
                model_identifier=llm_identifier(self.llm),
            )
            hit = self.cache.get(cache_key)
            if hit is not None:
                answer, evidence, trace = copy.deepcopy(hit)
                trace = {**trace, "request_id": request_id, "cache_hit": True,
                         "total_ms": round((time.perf_counter() - start) * 1000, 2)}
                return Result(answer, evidence, trace)

        # --- broad retrieval (§8, §9) --------------------------------------
        candidates = list(stage("candidate_retrieval", lambda: retrieve(self.index, query, self.settings, stamps), []) or [])
        if not candidates:
            return self._abstain(task, stamps, start, "no retrieval candidates", request_id=request_id)

        # --- conversation consistency gate (§10) ---------------------------
        gated = list(stage(
            "metadata_filter",
            lambda: apply_gate(candidates, self.index.conversations, query, raw_query, self.settings),
            candidates,
        ))

        # --- reranker (§11, §12) -------------------------------------------
        def _fusion_fallback() -> list:
            peak = max((c.fusion_score for c in gated), default=0.0) or 1.0
            for candidate in gated:
                candidate.rerank_score = round(min(1.0, candidate.fusion_score / peak), 6)
            return sorted(gated, key=lambda c: (-c.rerank_score, c.transcript_id))[
                : self.settings.rerank_k
            ]

        ranked = list(stage(
            "reranker",
            lambda: self.reranker.rerank(query, raw_query, gated, self.index.conversations, self.settings),
            _fusion_fallback(),
        ))
        if not ranked:
            return self._abstain(task, stamps, start, "reranker produced no candidates", request_id=request_id)
        if ranked[0].rerank_score < self.settings.min_rerank_score:
            return self._abstain(task, stamps, start, "no sufficiently relevant conversation", request_id=request_id)

        # --- hard anti-contamination gate (§23) ----------------------------
        selected = list(stage(
            "contamination_gate",
            lambda: ranked if (cross_conversation or self.settings.allow_cross_conversation) else ranked[:1],
            ranked[:1],
        ))

        # --- turn selection (§13) ------------------------------------------
        evidence: list[Evidence] = list(stage(
            "turn_selection",
            lambda: [
                item
                for candidate in selected
                for item in select(query, self.index.conversations[candidate.transcript_id], self.settings)
            ],
            [],
        ))
        if not evidence:
            return self._abstain(task, stamps, start, "no relevant turns selected", request_id=request_id)

        budget = (
            self.settings.evidence_token_budget_task1
            if task == "task1"
            else self.settings.evidence_token_budget_task2
        )

        # --- Laya decision pass 1 (§17–§19) --------------------------------
        features = compute_features(evidence, ranked, self.settings)
        reject = Decision(
            task_route=TaskRoute.TASK_1 if task == "task1" else TaskRoute.TASK_2,
            evidence_action=EvidenceAction.REJECT,
            compression_level=CompressionLevel.NONE,
            answer_gate=AnswerGate.ABSTAIN,
            reason="decision layer unavailable",
            policy="fail-closed",
        )
        decision = cast_decision(stage("laya", lambda: self.laya.decide(task, features, budget, self.settings), reject))
        laya_log: list[dict] = [decision.as_dict()]

        if decision.evidence_action is EvidenceAction.EXPAND:
            expanded = list(stage(
                "expansion",
                lambda: [
                    item
                    for candidate in selected
                    for item in expand(
                        [e for e in evidence if e.turn.transcript_id == candidate.transcript_id],
                        self.index.conversations[candidate.transcript_id],
                        self.settings,
                    )
                ],
                evidence,
            ) or [])
            evidence = expanded if expanded else evidence
            features = compute_features(evidence, ranked, self.settings)
            decision = cast_decision(stage("laya_redecide", lambda: self.laya.decide(task, features, budget, self.settings), decision))
            laya_log.append(decision.as_dict())
        else:
            stamps["expansion_ms"] = 0.0

        if decision.answer_gate is AnswerGate.ABSTAIN:
            return self._abstain(
                task, stamps, start, decision.reason, evidence,
                request_id=request_id, laya_log=laya_log,
            )

        # --- compression (§21) then evidence pack (§22) ---------------------
        compressed, compression_stats = stage(
            "compression",
            lambda: compress(evidence, decision.compression_level, budget, self.settings),
            (evidence, {"level": decision.compression_level.value, "error": "compression skipped"}),
        )
        packed, evidence = stage(
            "evidence_pack",
            lambda: pack(compressed, budget, self.settings),
            ("", compressed),
        )
        if not evidence:
            return self._abstain(task, stamps, start, "evidence pack empty after budgeting", request_id=request_id)

        # --- prompt construction (§26, §42) --------------------------------
        source_count = len({item.turn.transcript_id for item in evidence})
        generation_prompt = str(stage("prompt_build", lambda: build_prompt(task, packed, detail, source_count=source_count), "") or "")

        # --- LLM generation (§25) with deterministic fallback (§43, §44) ----
        raw: object = None
        generation_mode = "llm"
        if self.llm is not None:
            result = stage(
                "llm_generation",
                lambda: measure(self.llm, generation_prompt, timeout=self.settings.llm_timeout_s),
                None,
            )
            if result is not None:
                raw = result.text
                stamps["llm_ttft_ms"] = float(result.ttft_ms)
                stamps["llm_generation_ms"] = float(result.generation_ms)
                stamps["input_tokens"] = float(result.input_tokens)
                stamps["output_tokens"] = float(result.output_tokens)
        if raw is None:
            generation_mode = "deterministic_fallback"
            point = time.perf_counter()
            try:
                raw = self._fallback(task, evidence, cross_conversation)
            except Exception as exc:
                errors.append({"stage": "deterministic_fallback", "error": f"{type(exc).__name__}: {exc}"})
                raw = {"task": task, "insufficient_evidence": True}
            stamps["llm_ttft_ms"] = 0.0
            stamps["llm_generation_ms"] = round((time.perf_counter() - point) * 1000, 2)
        stamps.setdefault("llm_ttft_ms", 0.0)
        stamps.setdefault("input_tokens", 0.0)
        stamps.setdefault("output_tokens", 0.0)

        # --- structured parse (§27, §28) -----------------------------------
        parsed, valid = stage("output_parse", lambda: parse_answer(raw, task), ({}, False))
        if not valid:
            errors.append({"stage": "output_parse", "error": "model output was not valid JSON; used deterministic fallback"})
            generation_mode = "parse_fallback"
            answer = self._fallback(task, evidence, cross_conversation)
        else:
            answer = parsed

        # --- grounding + causal + consistency validation (§29) --------------
        grounding = stage(
            "grounding_validation",
            lambda: validate_grounding(answer, evidence, self.settings.min_grounding_ratio),
            ("UNSUPPORTED", 0.0),
        )
        # NOTE: an empty error list means success — never `or`-default these.
        causal_errors = list(stage(
            "causal_validation",
            lambda: validate_causal_shape(answer, task, evidence, cross_conversation),
            ["CAUSAL_VALIDATION_ERROR"],
        ))
        consistency_errors = list(stage(
            "consistency_validation",
            lambda: transcript_consistency(answer, evidence, cross_conversation),
            ["CONSISTENCY_VALIDATION_ERROR"],
        ))

        # --- final answer gate (§30) ---------------------------------------
        outcome = stage(
            "answer_gate",
            lambda: final_gate(answer, evidence, decision, grounding, causal_errors, consistency_errors, self.settings),
            GateOutcome(AnswerGate.ABSTAIN, "answer gate failure", "UNSUPPORTED", 0.0),
        )
        if outcome.gate is AnswerGate.ABSTAIN:
            return self._abstain(
                task, stamps, start, outcome.reason, evidence,
                status=outcome.grounding_status, request_id=request_id,
                laya_log=laya_log,
            )

        answer["grounding_status"] = outcome.grounding_status
        answer["grounding_ratio"] = outcome.grounding_ratio
        answer["insufficient_evidence"] = outcome.gate != AnswerGate.ANSWER
        answer["source_transcripts"] = sorted({item.turn.transcript_id for item in evidence})
        answer["generation_mode"] = generation_mode

        trace = self._trace(
            task=task, detail=detail, cross_conversation=cross_conversation,
            candidates=candidates, gated=gated, ranked=ranked, evidence=evidence,
            decision=decision, laya_log=laya_log, features=features,
            compression_stats=compression_stats, stamps=stamps, errors=errors,
            start=start, request_id=request_id, outcome=outcome,
            generation_mode=generation_mode, debug=debug,
        )
        if self.cache is not None and cache_key is not None and not errors:
            self.cache.set(cache_key, (copy.deepcopy(answer), copy.deepcopy(evidence), trace))
        self.traces.emit(trace)
        return Result(answer, evidence, trace)

    # ------------------------------------------------------------- helpers --
    def _trace(
        self, *, task, detail, cross_conversation, candidates, gated, ranked, evidence,
        decision, laya_log, features, compression_stats, stamps, errors, start,
        request_id, outcome, generation_mode, debug,
    ) -> dict:
        context_tokens = estimate_tokens(
            "\n".join(item.turn.text for item in evidence), self.settings
        )
        llm_generation_ms = float(stamps.get("llm_generation_ms", 0.0) or 0.0)
        output_tokens = int(float(stamps.get("output_tokens", 0.0) or 0.0))
        trace: dict = {
            "request_id": request_id,
            "system": SYSTEM,
            "system_version": SYSTEM_VERSION,
            "task": task,
            "detail": detail,
            "cross_conversation": cross_conversation,
            "candidate_conversations": len(candidates),
            "gated_conversations": len(gated),
            "reranked_conversations": len(ranked),
            "selected_transcript_ids": sorted({item.turn.transcript_id for item in evidence}),
            "selected_turns": [item.turn.turn_id for item in evidence],
            "laya_decisions": [entry["evidence_action"] for entry in laya_log],
            "laya_answer_gates": [entry["answer_gate"] for entry in laya_log],
            "laya_reason": decision.reason,
            "decision_policy": decision.policy,
            "compression_level": decision.compression_level.value,
            "compression": compression_stats,
            "context_tokens": context_tokens,
            "grounding_status": outcome.grounding_status,
            "grounding_ratio": outcome.grounding_ratio,
            "answer_gate": outcome.gate.value,
            "answer_gate_reason": outcome.reason,
            "pruned_claims": outcome.pruned_claims,
            "generation_mode": generation_mode,
            "reranker": {
                "name": getattr(self.reranker, "name", "unknown"),
                "version": getattr(self.reranker, "version", "unknown"),
            },
            "errors": errors,
            **{k: v for k, v in stamps.items() if k.endswith("_ms")},
        }
        extras = {k: v for k, v in stamps.items() if not k.endswith("_ms")}
        if extras:
            trace["stage_extras"] = extras
        trace["generation_tokens_per_second"] = (
            round(output_tokens / (llm_generation_ms / 1000), 2)
            if llm_generation_ms > 0 and output_tokens
            else 0.0
        )
        trace["total_ms"] = round((time.perf_counter() - start) * 1000, 2)
        if debug:
            trace["debug"] = {
                "candidates": [
                    {
                        "transcript_id": c.transcript_id,
                        "dense_score": c.dense_score,
                        "lexical_score": c.lexical_score,
                        "fusion_score": round(c.fusion_score, 6),
                        "consistency_score": c.consistency_score,
                        "rerank_score": c.rerank_score,
                        "source_retrievers": list(c.source_retrievers),
                    }
                    for c in ranked
                ],
                "consistency_breakdown": {
                    c.transcript_id: c.consistency_breakdown for c in gated
                },
                "features": features.as_dict(),
                "laya": laya_log,
                "selection": [
                    {
                        "transcript_id": item.turn.transcript_id,
                        "turn_id": item.turn.turn_id,
                        "speaker": item.turn.speaker,
                        "score": item.score,
                        "roles": list(item.roles),
                        "reason": item.reason,
                        "stage": item.retrieval_stage,
                    }
                    for item in evidence
                ],
            }
        return trace

    def _fallback(
        self, task: str, evidence: list[Evidence], cross_conversation: bool = False
    ) -> dict:
        """Deterministic evidence-cited answer (§44): never invents content."""
        if not evidence:
            empty: dict = {"task": task, "insufficient_evidence": True, "reason": "no evidence"}
            if task == "task1":
                empty["causal_rationales"] = []
                empty["confidence"] = 0.0
            else:
                empty.update({"summary": "", "key_causes": [], "effects": [], "mechanisms": [], "uncertainties": ["no evidence"], "evidence": []})
            return empty
        # Claims are always built from the strongest conversation only; the
        # citation list may span sources for explicit comparative requests (§23).
        primary = evidence[0].turn.transcript_id
        scoped = [item for item in evidence if item.turn.transcript_id == primary]
        roles = {role for item in scoped for role in item.roles}
        causal = (
            int(bool(roles & {r.value for r in CHAIN_CAUSE_SIDE}))
            + int(bool(roles & {r.value for r in CHAIN_ACTION_SIDE}))
            + int(bool(roles & {r.value for r in CHAIN_OUTCOME_SIDE}))
        ) / 3
        multi = len({item.turn.transcript_id for item in evidence}) > 1
        cited: list[object] = [
            f"{item.turn.transcript_id}:{item.turn.turn_id}" if multi else item.turn.turn_id
            for item in evidence
        ]
        if task == "task1":
            cause_item = next(
                (item for item in scoped if roles_of(item) & {r.value for r in CHAIN_CAUSE_SIDE}),
                None,
            )
            effect_item = next(
                (
                    item
                    for item in scoped
                    if roles_of(item) & {r.value for r in CHAIN_OUTCOME_SIDE}
                    and (cause_item is None or item.turn.turn_id != cause_item.turn.turn_id)
                ),
                None,
            )
            cause = cause_item.turn.text if cause_item else ""
            effect = effect_item.turn.text if effect_item else ""
            supported = bool(cause) and bool(effect) and cause.strip() != effect.strip()
            return {
                "task": "task1",
                "causal_rationales": (
                    [{
                        "cause": cause,
                        "effect": effect,
                        "mechanism": None,  # never invented (§15)
                        "evidence_turns": cited,
                        "strength": "explicit",
                    }]
                    if supported
                    else []
                ),
                "confidence": round(min(0.85, 0.35 + causal / 2), 2),
                "insufficient_evidence": not supported,
                "uncertainties": (
                    [] if supported else ["The retrieved evidence does not support a full cause-and-effect pair."]
                ),
            }
        return {
            "task": "task2",
            "summary": " ".join(item.turn.text for item in scoped[:3]),
            "key_causes": [item.turn.text for item in scoped if roles_of(item) & {r.value for r in CHAIN_CAUSE_SIDE}],
            "effects": [item.turn.text for item in scoped if roles_of(item) & {r.value for r in CHAIN_OUTCOME_SIDE}],
            "mechanisms": [],
            "uncertainties": (
                [] if causal >= 1 else ["The transcript does not explicitly establish the full mechanism or outcome."]
            ),
            "evidence": cited,
            "confidence": round(min(0.85, 0.35 + causal / 2), 2),
            "insufficient_evidence": causal < 0.67,
        }

    def _abstain(
        self,
        task: str,
        stamps: dict,
        start: float,
        reason: str,
        evidence: list[Evidence] | None = None,
        status: str = "UNSUPPORTED",
        request_id: str | None = None,
        laya_log: list[dict] | None = None,
    ) -> Result:
        """Abstention is a valid high-quality result (§30)."""
        if task == "task1":
            answer: dict = {
                "task": "task1",
                "causal_rationales": [],
                "confidence": 0.0,
                "insufficient_evidence": True,
            }
        else:
            answer = {
                "task": "task2",
                "summary": "",
                "key_causes": [],
                "effects": [],
                "mechanisms": [],
                "uncertainties": [reason],
                "evidence": [],
                "insufficient_evidence": True,
            }
        answer["reason"] = reason
        answer["grounding_status"] = status
        evidence = evidence or []
        trace = {
            "request_id": request_id or str(uuid.uuid4()),
            "system": SYSTEM,
            "system_version": SYSTEM_VERSION,
            "task": task,
            "candidate_conversations": 0,
            "reranked_conversations": 0,
            "selected_transcript_ids": sorted({item.turn.transcript_id for item in evidence}),
            "selected_turns": [item.turn.turn_id for item in evidence],
            "laya_decisions": [entry["evidence_action"] for entry in (laya_log or [])],
            "laya_answer_gates": [entry["answer_gate"] for entry in (laya_log or [])],
            "laya_reason": reason,
            "grounding_status": status,
            "answer_gate": AnswerGate.ABSTAIN.value,
            "answer_gate_reason": reason,
            "generation_mode": "none",
            "errors": [],
            "context_tokens": 0,
            **{k: v for k, v in stamps.items() if k.endswith("_ms")},
            "total_ms": round((time.perf_counter() - start) * 1000, 2),
        }
        extras = {k: v for k, v in stamps.items() if not k.endswith("_ms")}
        if extras:
            trace["stage_extras"] = extras
        return Result(answer, evidence, trace)


def roles_of(item: Evidence) -> set[str]:
    return set(item.roles)


def cast_decision(value: object) -> Decision:
    if isinstance(value, Decision):
        return value
    raise TypeError(f"expected Decision, got {type(value).__name__}")
