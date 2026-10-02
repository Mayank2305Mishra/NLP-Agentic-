"""Core typed contracts for System 4.

Every dataclass here carries provenance: evidence items always know their
source transcript, turn, retrieval stage, score, and selection reason
(AGENTS_SYSTEM_4.md §2, §13, §22).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

SYSTEM = "system4"
SYSTEM_VERSION = "system4-v2"


class CausalRole(str, Enum):
    """Causal evidence roles (AGENTS_SYSTEM_4.md §15)."""

    CAUSE = "cause"
    TRIGGER = "trigger"
    CUSTOMER_MOTIVATION = "customer_motivation"
    PROBLEM = "problem"
    AGENT_ACTION = "agent_action"
    MECHANISM = "mechanism"
    EFFECT = "effect"
    OUTCOME = "outcome"
    UNCERTAINTY = "uncertainty"
    QUERY_MATCH = "query_match"
    CLARIFICATION = "clarification"
    CONTRADICTION = "contradiction"
    CONTEXT = "context"
    CAUSAL_NEIGHBOR = "causal_neighbor"


#: Roles that carry causal weight (never "context"/"query_match" alone).
CAUSAL_ROLES = frozenset(
    {
        CausalRole.CAUSE,
        CausalRole.TRIGGER,
        CausalRole.CUSTOMER_MOTIVATION,
        CausalRole.PROBLEM,
        CausalRole.AGENT_ACTION,
        CausalRole.MECHANISM,
        CausalRole.EFFECT,
        CausalRole.OUTCOME,
    }
)

#: The three chain segments used for causal-coverage scoring (§16, §19).
CHAIN_CAUSE_SIDE = frozenset(
    {CausalRole.CAUSE, CausalRole.TRIGGER, CausalRole.PROBLEM, CausalRole.CUSTOMER_MOTIVATION}
)
CHAIN_ACTION_SIDE = frozenset({CausalRole.AGENT_ACTION, CausalRole.MECHANISM})
CHAIN_OUTCOME_SIDE = frozenset({CausalRole.EFFECT, CausalRole.OUTCOME})


class TaskRoute(str, Enum):
    TASK_1 = "TASK_1"
    TASK_2 = "TASK_2"
    UNKNOWN = "UNKNOWN"


class EvidenceAction(str, Enum):
    ACCEPT = "ACCEPT"
    EXPAND = "EXPAND"
    RERANK_MORE = "RERANK_MORE"
    REJECT = "REJECT"


class CompressionLevel(str, Enum):
    NONE = "NONE"
    LIGHT = "LIGHT"
    AGGRESSIVE = "AGGRESSIVE"


class AnswerGate(str, Enum):
    ANSWER = "ANSWER"
    ANSWER_WITH_UNCERTAINTY = "ANSWER_WITH_UNCERTAINTY"
    ABSTAIN = "ABSTAIN"


class GroundingStatus(str, Enum):
    GROUNDED = "GROUNDED"
    PARTIALLY_GROUNDED = "PARTIALLY_GROUNDED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


@dataclass(frozen=True)
class Turn:
    transcript_id: str
    turn_id: int
    speaker: str
    text: str
    domain: str | None = None
    intent: str | None = None
    reason_for_call: str | None = None
    post_call: bool = False


@dataclass
class Conversation:
    transcript_id: str
    domain: str | None
    intent: str | None
    reason_for_call: str | None
    turns: list[Turn]
    time_of_interaction: str | None = None


@dataclass
class Candidate:
    transcript_id: str
    dense_score: float | None = None
    lexical_score: float | None = None
    metadata_score: float = 0.0
    source_retrievers: list[str] = field(default_factory=list)
    #: Rank-fusion score from broad retrieval (§9); normalized at rerank time.
    fusion_score: float = 0.0
    #: Conversation consistency gate score (§10).
    consistency_score: float = 0.0
    consistency_breakdown: dict[str, float] = field(default_factory=dict)
    #: Final deterministic reranker score in [0, 1] (§12).
    rerank_score: float = 0.0
    #: Named reranker features for observability (§45 debug mode).
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class Evidence:
    turn: Turn
    score: float
    roles: list[str]
    reason: str
    retrieval_stage: str = "turn_selection"
    #: Compressed representation; ``turn.text`` always keeps the original (§21).
    compressed_text: str | None = None

    @property
    def transcript_id(self) -> str:
        return self.turn.transcript_id

    @property
    def turn_id(self) -> int:
        return self.turn.turn_id

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["roles"] = list(self.roles)
        return data


@dataclass(frozen=True)
class Decision:
    task_route: TaskRoute
    evidence_action: EvidenceAction
    compression_level: CompressionLevel
    answer_gate: AnswerGate
    reason: str
    policy: str = "deterministic"

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_route": self.task_route.value,
            "evidence_action": self.evidence_action.value,
            "compression_level": self.compression_level.value,
            "answer_gate": self.answer_gate.value,
            "reason": self.reason,
            "policy": self.policy,
        }


@dataclass
class Result:
    answer: dict[str, Any]
    evidence: list[Evidence]
    trace: dict[str, Any]

    def public(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "evidence": [item.to_dict() for item in self.evidence],
            "trace": self.trace,
        }
