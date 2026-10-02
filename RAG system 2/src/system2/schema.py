from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

TASK1_STRENGTHS = ("explicit", "strongly_supported", "inferred", "insufficient")


@dataclass
class Turn:
    transcript_id: str
    turn_id: int
    speaker_original: str
    speaker: str
    text_original: str
    text_normalized: str
    domain: str | None = None
    intent: str | None = None
    reason_for_call: str | None = None
    time_of_interaction: str | None = None
    is_post_call_artifact: bool = False
    speaker_inferred: bool = False

    @property
    def key(self) -> tuple[str, int]:
        return (self.transcript_id, self.turn_id)


@dataclass
class Conversation:
    transcript_id: str
    time_of_interaction: str | None
    domain: str | None
    intent: str | None
    reason_for_call: str | None
    turns: list[Turn] = field(default_factory=list)


@dataclass
class QueryPlan:
    raw_query: str
    query_id: str
    normalized_query: str
    terms: list[str]
    phrases: list[str]
    causal_terms: list[str]
    event_terms: list[str]
    outcome_terms: list[str]
    expanded_terms: list[str]
    metadata_hints: dict[str, Any]
    explicit_intent: str | None
    intent_match: str | None = None
    task: str = "task1"
    comparison: bool = False
    fts_match: str = ""
    fts_match_loose: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreComponents:
    bm25_raw: float = 0.0
    bm25: float = 0.0
    metadata: float = 0.0
    phrase: float = 0.0
    causal: float = 0.0
    outcome: float = 0.0
    problem: float = 0.0
    action: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in asdict(self).items()}


@dataclass
class ConversationCandidate:
    transcript_id: str
    score: float
    bm25_raw: float
    components: dict[str, float] = field(default_factory=dict)
    conversation_meta: dict[str, Any] = field(default_factory=dict)
    stage: str = "conversation_ranking"


@dataclass
class EvidenceItem:
    transcript_id: str
    turn_id: int
    speaker: str
    speaker_original: str
    text_original: str
    text_normalized: str
    score: float
    components: dict[str, float] = field(default_factory=dict)
    stage: str = "turn_ranking"
    seed: bool = False
    is_post_call_artifact: bool = False


@dataclass
class EvidenceWindow:
    transcript_id: str
    start_turn: int
    end_turn: int
    score: float
    seed_turn_ids: list[int]
    components: dict[str, float] = field(default_factory=dict)

    @property
    def turn_ids(self) -> list[int]:
        return list(range(self.start_turn, self.end_turn + 1))


@dataclass
class EvidenceConversation:
    transcript_id: str
    score: float
    meta: dict[str, Any]
    turns: list[EvidenceItem]
    windows: list[EvidenceWindow]
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class EvidencePack:
    task: str
    conversations: list[EvidenceConversation]
    token_estimate: int
    allowed_transcript_ids: list[str]
    text: str


@dataclass
class GateDecision:
    decision: str
    reason: str
    score: float
    transcript_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GroundingReport:
    passed: bool
    contamination: bool
    claims: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerResult:
    request_id: str
    task: str
    query: str
    query_plan: dict[str, Any]
    answer: dict[str, Any] | None
    raw_output: str
    parse_ok: bool
    evidence_gate: dict[str, Any]
    grounding: dict[str, Any]
    answer_gate: dict[str, Any]
    candidates: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    trace: dict[str, Any]
    tokens: dict[str, Any]
    model: str
    provider: str
    config_hash: str
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
