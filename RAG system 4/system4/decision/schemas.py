"""Typed Laya decision contracts (AGENTS_SYSTEM_4.md §18).

Application code depends on these types only; the protocol layer lives in
``system4.decision.jev`` (§20).
"""

from __future__ import annotations

from system4.schema import (  # re-exported for a stable import path (§18)
    AnswerGate,
    CompressionLevel,
    Decision,
    EvidenceAction,
    TaskRoute,
)

__all__ = [
    "AnswerGate",
    "CompressionLevel",
    "Decision",
    "EvidenceAction",
    "TaskRoute",
    "task_route_of",
]


def task_route_of(task: str) -> TaskRoute:
    if task == "task1":
        return TaskRoute.TASK_1
    if task == "task2":
        return TaskRoute.TASK_2
    return TaskRoute.UNKNOWN
