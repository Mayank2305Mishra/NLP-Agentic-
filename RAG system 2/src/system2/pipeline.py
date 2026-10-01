from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from .cache import LRUCache
from .config import Config
from .decisions import DecisionLog
from .grounding import validate_grounding
from .llm import LLMClient, build_client
from .logging_utils import StructuredLogger
from .observability import StageTimer, new_request_id
from .parsing import parse_answer
from .prompts import INSUFFICIENT_ANSWER, build_prompt
from .query import QueryProcessor
from .retriever import RetrievalResult, Retriever
from .schema import AnswerResult, GroundingReport, QueryPlan
from .store import LexicalStore

log = logging.getLogger("system2.pipeline")

RETRIEVAL_STAGES = [
    "query_normalization_ms",
    "metadata_filter_ms",
    "fts_retrieval_ms",
    "conversation_ranking_ms",
    "turn_ranking_ms",
    "neighbor_expansion_ms",
    "evidence_pack_ms",
    "evidence_gate_ms",
]


class Pipeline:
    def __init__(
        self,
        config: Config,
        store: LexicalStore | None = None,
        client: LLMClient | None = None,
        cache: LRUCache | None = None,
    ):
        self.config = config
        self.store = store or LexicalStore(config.dataset.db_path)
        self.cache = cache or LRUCache(config.cache.max_entries, config.cache.enabled)
        self.client = client or build_client(config)
        self.processor = QueryProcessor(
            intents=self.store.intents(),
            domains=self._domains(),
        )
        self.retriever = Retriever(self.store, config, self.cache)
        self.logger = StructuredLogger(
            config.logging.log_path, config.logging.level, config.logging.debug_evidence
        )
        self.decision_path = config.logging.log_path.parent / "decisions.jsonl"

    def _domains(self) -> list[str]:
        rows = self.store.conn.execute(
            "SELECT DISTINCT domain FROM conversations WHERE domain IS NOT NULL"
        ).fetchall()
        return [r[0] for r in rows]

    def parse_query(self, raw_query: str, task: str | None = None) -> QueryPlan:
        key = f"plan:{raw_query}:{task}:{self.config.config_hash}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        plan = self.processor.process(raw_query, task=task)
        self.cache.set(key, plan)
        return plan

    def retrieve_only(self, query: str, task: str | None = None) -> dict[str, Any]:
        timer = StageTimer.start()
        request_id = new_request_id()
        decisions = DecisionLog(request_id, self.decision_path)
        with timer.stage("query_normalization_ms"):
            plan = self.parse_query(query, task)
        decisions.record(
            "CLASSIFY_TASK",
            f"task={plan.task} comparison={plan.comparison}",
            query_id=plan.query_id,
        )
        retrieval = self.retriever.retrieve(plan, timer, decisions)
        trace = timer.finish()
        decisions.flush()
        return {
            "request_id": request_id,
            "plan": plan,
            "retrieval": retrieval,
            "trace": trace,
            "decisions": decisions.records,
        }

    def ask(self, query: str, task: str | None = None) -> AnswerResult:
        timer = StageTimer.start()
        request_id = new_request_id()
        decisions = DecisionLog(request_id, self.decision_path)
        errors: list[str] = []

        with timer.stage("query_normalization_ms"):
            plan = self.parse_query(query, task)
        decisions.record(
            "CLASSIFY_TASK",
            f"task={plan.task} comparison={plan.comparison}",
            query_id=plan.query_id,
        )

        retrieval = self.retriever.retrieve(plan, timer, decisions)
        answer, parse_ok, gen_info = self._generate(plan, retrieval, timer, errors)

        grounding_report: GroundingReport | None = None
        with timer.stage("grounding_ms"):
            if answer is None and retrieval.gate.decision == "INSUFFICIENT":
                answer = dict(INSUFFICIENT_ANSWER[plan.task])
                parse_ok = True
            if answer is not None:
                grounding_report = validate_grounding(
                    answer, plan.task, plan, retrieval.pack, self.config, decisions
                )

        with timer.stage("answer_gate_ms"):
            answer_gate = self._answer_gate(decisions, grounding_report, retrieval)

        trace = timer.finish()
        evidence_summary = self._evidence_summary(retrieval)
        if retrieval.pack is not None and retrieval.pack.conversations:
            evidence_summary = self._pack_summary(retrieval)
        result = AnswerResult(
            request_id=request_id,
            task=plan.task,
            query=query,
            query_plan=plan.to_dict(),
            answer=answer,
            raw_output=gen_info.get("raw", ""),
            parse_ok=parse_ok,
            evidence_gate=retrieval.gate.to_dict(),
            grounding=grounding_report.to_dict() if grounding_report else {},
            answer_gate=answer_gate,
            candidates=[
                {
                    "transcript_id": c.transcript_id,
                    "score": c.score,
                    "bm25_raw": round(c.bm25_raw, 4),
                    "components": c.components,
                    "meta": c.conversation_meta,
                }
                for c in retrieval.candidates
            ],
            evidence=evidence_summary,
            trace=trace,
            tokens=gen_info.get("tokens", {}),
            model=self.config.generation.model,
            provider=self.config.generation.provider,
            config_hash=self.config.config_hash,
            errors=errors,
        )
        decisions.flush()
        self.logger.emit(self._log_payload(result))
        return result

    def _generate(
        self, plan: QueryPlan, retrieval: RetrievalResult, timer: StageTimer, errors: list[str]
    ) -> tuple[dict[str, Any] | None, bool, dict[str, Any]]:
        info: dict[str, Any] = {"raw": "", "tokens": {}}
        if retrieval.gate.decision == "INSUFFICIENT" or retrieval.pack is None:
            timer.set("generation_ttft_ms", None)
            timer.set("generation_ms", 0.0)
            timer.set("parse_ms", 0.0)
            return None, True, info

        system, prompt = build_prompt(plan.task, plan.raw_query, retrieval.pack.text)
        gen = self.client.generate(
            system,
            prompt,
            temperature=self.config.generation.temperature,
            num_predict=self.config.generation.num_predict,
            timeout_s=self.config.generation.request_timeout_s,
        )
        parse_begin = time.perf_counter()
        raw = gen.text
        answer, parse_errors = parse_answer(raw)
        retries = 0
        while answer is None and retries < self.config.generation.parse_retries:
            retries += 1
            retry = self.client.generate(
                system,
                prompt + "\n\nYour previous reply was not valid JSON. Return only the JSON object.",
                temperature=self.config.generation.temperature,
                num_predict=self.config.generation.num_predict,
                timeout_s=self.config.generation.request_timeout_s,
            )
            raw = retry.text
            answer, parse_errors = parse_answer(raw)
            gen.generation_ms += retry.generation_ms
            gen.output_tokens = (gen.output_tokens or 0) + (retry.output_tokens or 0)
            if retry.prompt_tokens and not gen.prompt_tokens:
                gen.prompt_tokens = retry.prompt_tokens
        parse_ms = round((time.perf_counter() - parse_begin) * 1000, 3)
        timer.set("parse_ms", parse_ms)
        if answer is None:
            errors.extend(parse_errors)

        timer.set("generation_ttft_ms", gen.ttft_ms)
        timer.set("generation_ms", round(gen.generation_ms, 3))
        info["raw"] = raw
        info["tokens"] = {
            "input_tokens": gen.prompt_tokens,
            "output_tokens": gen.output_tokens,
            "context_tokens": gen.prompt_tokens,
            "output_tokens_per_second": gen.tokens_per_second,
        }
        return answer, answer is not None, info

    def _answer_gate(
        self,
        decisions: DecisionLog,
        grounding_report: GroundingReport | None,
        retrieval: RetrievalResult,
    ) -> dict[str, Any]:
        answer_decisions = [
            d
            for d in decisions.records
            if d.decision in ("ACCEPT_ANSWER", "FLAG_ANSWER", "REJECT_ANSWER")
        ]
        if retrieval.gate.decision == "INSUFFICIENT" and retrieval.pack is None:
            final = "ACCEPT_ANSWER"
            reason = "insufficient evidence: returned insufficient_evidence=true without invoking LLM"
            decisions.record(final, reason)
        elif answer_decisions:
            final = answer_decisions[-1].decision
            reason = answer_decisions[-1].reason
        else:
            final = "REJECT_ANSWER"
            reason = "no answer produced"
            decisions.record(final, reason)
        return {
            "decision": final,
            "reason": reason,
            "grounding_passed": bool(grounding_report.passed) if grounding_report else False,
            "contamination": bool(grounding_report.contamination) if grounding_report else False,
            "decisions": [d.to_dict() for d in decisions.records],
        }

    @staticmethod
    def _pack_summary(retrieval: RetrievalResult) -> list[dict[str, Any]]:
        pack = retrieval.pack
        assert pack is not None
        windows_by_conv = {e.transcript_id: e.windows for e in retrieval.evidence}
        return [
            {
                "transcript_id": conv.transcript_id,
                "conversation_score": conv.score,
                "components": conv.components,
                "windows": [
                    {"start": w.start_turn, "end": w.end_turn, "score": w.score}
                    for w in windows_by_conv.get(conv.transcript_id, [])
                ],
                "turns": [
                    {
                        "turn_id": t.turn_id,
                        "speaker": t.speaker,
                        "stage": t.stage,
                        "seed": t.seed,
                        "score": t.score,
                        "components": t.components,
                        "text": t.text_original,
                    }
                    for t in conv.turns
                ],
            }
            for conv in pack.conversations
        ]

    @staticmethod
    def _evidence_summary(retrieval: RetrievalResult) -> list[dict[str, Any]]:
        out = []
        for conv in retrieval.evidence:
            out.append(
                {
                    "transcript_id": conv.transcript_id,
                    "conversation_score": conv.score,
                    "components": conv.components,
                    "windows": [
                        {"start": w.start_turn, "end": w.end_turn, "score": w.score}
                        for w in conv.windows
                    ],
                    "turns": [
                        {
                            "turn_id": t.turn_id,
                            "speaker": t.speaker,
                            "stage": t.stage,
                            "seed": t.seed,
                            "score": t.score,
                            "components": t.components,
                            "text": t.text_original,
                        }
                        for t in conv.turns
                    ],
                }
            )
        return out

    @staticmethod
    def _log_payload(result: AnswerResult) -> dict[str, Any]:
        trace = result.trace
        retrieval_ms = sum(trace.get(k) or 0.0 for k in RETRIEVAL_STAGES)
        return {
            "request_id": result.request_id,
            "task": result.task,
            "query_hash": hashlib.sha256(result.query.encode()).hexdigest()[:16],
            "candidate_count": len(result.candidates),
            "selected_transcript_ids": [c["transcript_id"] for c in result.evidence],
            "evidence_turn_count": sum(len(c["turns"]) for c in result.evidence),
            "retrieval_ms": round(retrieval_ms, 3),
            "generation_ms": trace.get("generation_ms"),
            "total_ms": trace.get("total_ms"),
            "grounding_pass": result.grounding.get("passed", False),
            "answer_gate": result.answer_gate.get("decision"),
            "evidence_gate": result.evidence_gate.get("decision"),
            "model": result.model,
            "provider": result.provider,
            "config_hash": result.config_hash,
            "trace": trace,
            "tokens": result.tokens,
            "errors": result.errors,
        }
