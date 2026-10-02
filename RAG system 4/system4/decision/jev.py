"""Jev-compatible adapter for the decision layer (AGENTS_SYSTEM_4.md §20).

Application logic talks to :class:`DecisionPolicy`; this adapter exposes the
same typed decisions through a Jev System-One-compatible protocol envelope.
Local-first by default: a remote endpoint is only consulted when configured
and any failure falls back to the local deterministic policy. The application
never depends on a remote Jev service.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from system4.config.settings import Settings
from system4.decision.policies import DecisionPolicy, DeterministicPolicy, decide_with_fallback
from system4.evidence.gate import EvidenceFeatures
from system4.schema import AnswerGate, CompressionLevel, Decision, EvidenceAction, TaskRoute

JEV_SCHEMA = "jev.system-one.decision.v1"


def to_protocol(decision: Decision, features: EvidenceFeatures) -> dict[str, Any]:
    """Jev-compatible typed envelope for a decision + its input features."""
    return {
        "schema": JEV_SCHEMA,
        "decision": decision.as_dict(),
        "features": features.as_dict(),
    }


def from_protocol(payload: dict[str, Any]) -> Decision:
    data = payload.get("decision", payload)
    return Decision(
        task_route=TaskRoute(data["task_route"]),
        evidence_action=EvidenceAction(data["evidence_action"]),
        compression_level=CompressionLevel(data["compression_level"]),
        answer_gate=AnswerGate(data["answer_gate"]),
        reason=str(data.get("reason", "")),
        policy=str(data.get("policy", "jev")),
    )


class JevAdapter:
    """Bridges a DecisionPolicy to a Jev-compatible interface (§20)."""

    def __init__(
        self,
        policy: DecisionPolicy | None = None,
        endpoint: str | None = None,
        timeout: float = 1.0,
    ) -> None:
        self.policy = policy or DeterministicPolicy()
        self.endpoint = endpoint if endpoint is not None else os.getenv("JEV_ENDPOINT") or None
        self.timeout = timeout

    def decide(
        self, task: str, features: EvidenceFeatures, budget: int, settings: Settings
    ) -> Decision:
        local = decide_with_fallback(task, features, budget, settings, self.policy)
        if not self.endpoint:
            return local
        try:
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(to_protocol(local, features)).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode() or "{}")
            return from_protocol(payload) if payload else local
        except Exception:
            # §20/§43: remote Jev is optional; local policy always answers.
            return local
