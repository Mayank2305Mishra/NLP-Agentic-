from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

from .cache import LRUCache, index_signature
from .config import Config
from .decisions import DecisionLog
from .evidence import pack_evidence
from .gate import causal_chain_incomplete, evaluate_evidence_gate
from .observability import StageTimer
from .schema import (
    ConversationCandidate,
    EvidenceConversation,
    EvidenceItem,
    EvidencePack,
    EvidenceWindow,
    GateDecision,
    QueryPlan,
    ScoreComponents,
)
from .scoring import conversation_score, turn_score
from .store import LexicalStore

log = logging.getLogger("system2.retriever")

ABLATIONS: dict[str, dict[str, Any]] = {
    "pure_bm25": {"metadata": False, "causal": False, "window": 0, "gate": False},
    "bm25_metadata": {"metadata": True, "causal": False, "window": 0, "gate": False},
    "bm25_causal": {"metadata": False, "causal": True, "window": 0, "gate": False},
    "bm25_expansion": {"metadata": False, "causal": False, "window": 2, "gate": False},
    "full": {"metadata": True, "causal": True, "window": 2, "gate": True},
}


@dataclass
class RetrievalSettings:
    metadata: bool = True
    causal: bool = True
    neighbor_window: int = 2
    gate: bool = True
    ablation: str = "full"

    @classmethod
    def from_config(cls, config: Config) -> "RetrievalSettings":
        mode = (config.ablation or {}).get("mode", "full")
        settings = cls(
            metadata=config.retrieval.metadata_boost,
            causal=config.retrieval.causal_scoring,
            neighbor_window=config.retrieval.neighbor_window,
            gate=True,
            ablation=mode if mode in ABLATIONS else "full",
        )
        if mode in ABLATIONS and mode != "full":
            preset = ABLATIONS[mode]
            settings.metadata = preset["metadata"]
            settings.causal = preset["causal"]
            settings.neighbor_window = preset["window"]
            settings.gate = preset["gate"]
        return settings


@dataclass
class RetrievalResult:
    candidates: list[ConversationCandidate]
    evidence: list[EvidenceConversation]
    gate: GateDecision
    pack: EvidencePack | None
    match_mode: str
    settings: RetrievalSettings
    expanded: int = 0
    turns_by_conv: dict[str, dict[int, EvidenceItem]] = field(default_factory=dict)


def merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not windows:
        return []
    windows = sorted(windows)
    merged = [list(windows[0])]
    for start, end in windows[1:]:
        last = merged[-1]
        if start <= last[1] + 1:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def scoring_config(config: Config, settings: RetrievalSettings) -> Any:
    clone = copy.copy(config.retrieval)
    clone.metadata_boost = settings.metadata
    clone.causal_scoring = settings.causal
    return clone


class Retriever:
    def __init__(self, store: LexicalStore, config: Config, cache: LRUCache | None = None):
        self.store = store
        self.config = config
        self.cache = cache or LRUCache(config.cache.max_entries, config.cache.enabled)
        self.settings = RetrievalSettings.from_config(config)
        self._index_sig = index_signature(config.dataset.db_path)
        self._plan: QueryPlan | None = None
        self._scored: dict[str, list[tuple[int, float, ScoreComponents]]] = {}
        self._raw_bm: dict[str, dict[int, float]] = {}

    def retrieve(
        self, plan: QueryPlan, timer: StageTimer, decisions: DecisionLog
    ) -> RetrievalResult:
        cfg = self.config.retrieval
        settings = self.settings
        self._plan = plan
        match_mode = "strict"
        match = plan.fts_match or plan.fts_match_loose

        if not match:
            timer.set("fts_retrieval_ms", 0.0)
            gate = evaluate_evidence_gate([], self.config, decisions, causal_gap=False)
            return RetrievalResult([], [], gate, None, "empty", settings)

        intent_filter = (
            plan.explicit_intent
            if cfg.hard_filter_on_explicit_intent and settings.metadata
            else None
        )
        if intent_filter:
            decisions.record(
                "APPLY_METADATA_FILTER",
                f"explicit intent constraint: {intent_filter}",
                hard=True,
            )

        with timer.stage("metadata_filter_ms"):
            pass

        with timer.stage("fts_retrieval_ms"):
            rows = self._conversation_rows(match, intent_filter, match_mode)
            if not rows and match_mode == "strict" and plan.fts_match_loose:
                match_mode = "loose"
                rows = self._conversation_rows(plan.fts_match_loose, intent_filter, match_mode)
            if not rows and intent_filter:
                match_mode = match_mode + "+nofilter"
                rows = self._conversation_rows(match, None, match_mode)

        with timer.stage("conversation_ranking_ms"):
            candidates = self._rank_conversations(plan, rows, decisions)

        top = candidates[: cfg.conversation_top_k]
        if not top:
            gate = evaluate_evidence_gate([], self.config, decisions, causal_gap=False)
            return RetrievalResult([], [], gate, None, match_mode, settings)

        with timer.stage("turn_ranking_ms"):
            turn_rows = self._turn_rows(plan, match_mode, [c.transcript_id for c in top])
            self._score_turns(plan, top, turn_rows)

        window = settings.neighbor_window
        seeds_per_conv = cfg.turns_per_conversation
        with timer.stage("neighbor_expansion_ms"):
            evidence, turns_by_conv = self._build_evidence(top, window, seeds_per_conv)

        gate = self._gate_with_expansion(evidence, turns_by_conv, top, timer, decisions, window, seeds_per_conv)

        pack = None
        if gate.decision != "INSUFFICIENT" and evidence:
            max_tokens = (
                self.config.evidence.task1_max_tokens
                if plan.task == "task1"
                else self.config.evidence.task2_max_tokens
            )
            packable = evidence[:1] if (not plan.comparison and len(evidence) > 1) else evidence
            with timer.stage("evidence_pack_ms"):
                pack = pack_evidence(packable, plan.task, max_tokens, turns_by_conv)
            decisions.record(
                "SUFFICIENT_EVIDENCE" if gate.decision == "SUFFICIENT" else "EXPAND_EVIDENCE",
                f"evidence pack built with {len(pack.conversations)} conversation(s)",
                score=gate.score,
                tokens=pack.token_estimate,
                transcript_ids=pack.allowed_transcript_ids,
            )
        else:
            timer.set("evidence_pack_ms", 0.0)

        return RetrievalResult(
            candidates=candidates,
            evidence=evidence,
            gate=gate,
            pack=pack,
            match_mode=match_mode,
            settings=settings,
            expanded=getattr(self, "_last_expansions", 0),
            turns_by_conv=turns_by_conv,
        )

    def _gate_with_expansion(
        self,
        evidence: list[EvidenceConversation],
        turns_by_conv: dict[str, dict[int, EvidenceItem]],
        top: list[ConversationCandidate],
        timer: StageTimer,
        decisions: DecisionLog,
        window: int,
        seeds_per_conv: int,
    ) -> GateDecision:
        cfg = self.config.retrieval
        expansions = 0
        max_window = cfg.neighbor_window_max
        current = evidence
        current_index = turns_by_conv

        with timer.stage("evidence_gate_ms"):
            while True:
                if not self.settings.gate:
                    best = top[0]
                    decisions.record(
                        "SUFFICIENT_EVIDENCE",
                        "evidence gate disabled by ablation",
                        score=best.score,
                        transcript_id=best.transcript_id,
                    )
                    gate = GateDecision("SUFFICIENT", "ablation gate disabled", best.score, best.transcript_id)
                    break
                gap = causal_chain_incomplete(current)
                gate = evaluate_evidence_gate(current, self.config, decisions, causal_gap=gap)
                if gate.decision != "EXPAND":
                    break
                if expansions >= 2 or window >= max_window:
                    evaluate_evidence_gate(current, self.config, decisions, causal_gap=False)
                    gate = GateDecision(
                        "SUFFICIENT",
                        "causal chain still incomplete after bounded expansion",
                        gate.score,
                        gate.transcript_id,
                        detail={"expansions": expansions, "window": window},
                    )
                    break
                window = min(max_window, window + 1)
                seeds_per_conv += 2
                expansions += 1
                current, current_index = self._build_evidence(top, window, seeds_per_conv)
                decisions.record(
                    "EXPAND_EVIDENCE",
                    f"widened neighbor window to +/-{window}, seeds to {seeds_per_conv}",
                    score=gate.score,
                    transcript_id=gate.transcript_id,
                )

        evidence[:] = current
        snapshot = dict(current_index)
        turns_by_conv.clear()
        turns_by_conv.update(snapshot)
        self._last_expansions = expansions
        return gate

    def _conversation_rows(self, match: str, intent: str | None, mode: str) -> list[dict[str, Any]]:
        if not match:
            return []
        key = f"conv:{self._index_sig}:{self.config.config_hash}:{match}:{intent}:{mode}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        rows = self.store.search_conversations(
            match, self.config.retrieval.conversation_pool, intent=intent
        )
        result = [dict(r) for r in rows]
        self.cache.set(key, result)
        log.debug("conversation fts mode=%s rows=%d", mode, len(result))
        return result

    def _rank_conversations(
        self, plan: QueryPlan, rows: list[dict[str, Any]], decisions: DecisionLog
    ) -> list[ConversationCandidate]:
        cfg = scoring_config(self.config, self.settings)
        scored: list[ConversationCandidate] = []
        for row in rows:
            tid = row["transcript_id"]
            meta = {
                "intent": row.get("intent"),
                "domain": row.get("domain"),
                "reason_for_call": row.get("reason_for_call"),
                "time_of_interaction": row.get("time_of_interaction"),
            }
            score, comps = conversation_score(
                plan, self.store.conversation_text(tid), meta, cfg, row["bm"]
            )
            scored.append(
                ConversationCandidate(
                    transcript_id=tid,
                    score=round(score, 4),
                    bm25_raw=row["bm"],
                    components=comps.to_dict(),
                    conversation_meta=meta,
                )
            )
        scored.sort(key=lambda c: c.score, reverse=True)
        for cand in scored[: self.config.retrieval.conversation_top_k]:
            decisions.record(
                "ACCEPT_CANDIDATE",
                "selected as top conversation candidate",
                score=cand.score,
                transcript_id=cand.transcript_id,
            )
        if len(scored) > self.config.retrieval.conversation_top_k:
            rejected = scored[self.config.retrieval.conversation_top_k]
            decisions.record(
                "REJECT_CANDIDATE",
                "outside conversation_top_k",
                score=rejected.score,
                transcript_id=rejected.transcript_id,
            )
        return scored

    def _turn_rows(
        self, plan: QueryPlan, match_mode: str, transcript_ids: list[str]
    ) -> list[dict[str, Any]]:
        cfg = self.config.retrieval
        match = plan.fts_match_loose if "loose" in match_mode else (plan.fts_match or plan.fts_match_loose)
        limit = cfg.turn_pool_per_conversation * max(1, len(transcript_ids))
        key = (
            f"turn:{self._index_sig}:{self.config.config_hash}:{match}:"
            f"{tuple(sorted(transcript_ids))}:{limit}:{int(cfg.include_post_call)}"
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        rows = self.store.search_turns(
            match, transcript_ids, limit, include_post_call=cfg.include_post_call
        )
        if not rows and plan.fts_match_loose and plan.fts_match_loose != match:
            rows = self.store.search_turns(
                plan.fts_match_loose,
                transcript_ids,
                limit,
                include_post_call=cfg.include_post_call,
            )
        result = [dict(r) for r in rows]
        self.cache.set(key, result)
        return result

    def _score_turns(
        self,
        plan: QueryPlan,
        top: list[ConversationCandidate],
        turn_rows: list[dict[str, Any]],
    ) -> None:
        cfg = scoring_config(self.config, self.settings)
        bm_by_key = {(r["transcript_id"], r["turn_id"]): r["bm"] for r in turn_rows}
        meta_by_conv = {c.transcript_id: c.conversation_meta for c in top}
        all_turns = self.store.fetch_turns(
            list(meta_by_conv), include_post_call=self.config.retrieval.include_post_call
        )
        scored: dict[str, list[tuple[int, float, ScoreComponents]]] = {}
        raw_bm: dict[str, dict[int, float]] = {}
        for tid, turns in all_turns.items():
            meta = meta_by_conv.get(tid, {})
            entries = []
            raw_bm[tid] = {}
            for turn in turns:
                bm = bm_by_key.get((tid, turn.turn_id), 0.0)
                if bm:
                    raw_bm[tid][turn.turn_id] = bm
                score, comps = turn_score(plan, turn.text_normalized, meta, cfg, bm)
                entries.append((turn.turn_id, score, comps))
            entries.sort(key=lambda e: e[1], reverse=True)
            scored[tid] = entries
        self._scored = scored
        self._raw_bm = raw_bm

    def _build_evidence(
        self, top: list[ConversationCandidate], window: int, seeds_per_conv: int
    ) -> tuple[list[EvidenceConversation], dict[str, dict[int, EvidenceItem]]]:
        assert self._plan is not None
        plan = self._plan
        cfg = scoring_config(self.config, self.settings)
        all_turns = self.store.fetch_turns(
            [c.transcript_id for c in top],
            include_post_call=self.config.retrieval.include_post_call,
        )
        evidence: list[EvidenceConversation] = []
        turns_by_conv: dict[str, dict[int, EvidenceItem]] = {}

        for cand in top:
            tid = cand.transcript_id
            conv_turns = all_turns.get(tid, [])
            if not conv_turns:
                continue
            by_id = {t.turn_id: t for t in conv_turns}
            max_turn = max(by_id)
            seed_entries = self._scored.get(tid, [])[:seeds_per_conv]
            if not seed_entries:
                continue
            seed_ids = {sid for sid, _, _ in seed_entries}
            windows = merge_windows(
                [(max(0, sid - window), min(max_turn, sid + window)) for sid, _, _ in seed_entries]
            )
            score_by_seed = {sid: s for sid, s, _ in seed_entries}
            evid_windows = [
                EvidenceWindow(
                    transcript_id=tid,
                    start_turn=start,
                    end_turn=end,
                    score=round(max(score_by_seed[s] for s in seed_ids if start <= s <= end), 4),
                    seed_turn_ids=sorted(s for s in seed_ids if start <= s <= end),
                )
                for start, end in windows
            ]
            turns_map: dict[int, EvidenceItem] = {}
            for w in evid_windows:
                for turn_id in w.turn_ids:
                    turn = by_id.get(turn_id)
                    if turn is None:
                        continue
                    entry = next((e for e in seed_entries if e[0] == turn_id), None)
                    if entry is not None:
                        score, comps = entry[1], entry[2]
                        seed = True
                    else:
                        score, comps = turn_score(
                            plan, turn.text_normalized, cand.conversation_meta, cfg, 0.0
                        )
                        seed = False
                    turns_map[turn_id] = EvidenceItem(
                        transcript_id=tid,
                        turn_id=turn_id,
                        speaker=turn.speaker,
                        speaker_original=turn.speaker_original,
                        text_original=turn.text_original,
                        text_normalized=turn.text_normalized,
                        score=round(score, 4),
                        components=comps.to_dict(),
                        stage="turn_ranking" if seed else "neighbor_expansion",
                        seed=seed,
                        is_post_call_artifact=turn.is_post_call_artifact,
                    )
            evidence.append(
                EvidenceConversation(
                    transcript_id=tid,
                    score=cand.score,
                    meta=cand.conversation_meta,
                    turns=sorted(turns_map.values(), key=lambda i: i.turn_id),
                    windows=evid_windows,
                    components=cand.components,
                )
            )
            turns_by_conv[tid] = turns_map
        return evidence, turns_by_conv
