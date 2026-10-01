"""Shared enums."""
from __future__ import annotations

from enum import Enum


class TaskType(str, Enum):
    TASK1 = "task1"
    TASK2 = "task2"


class Strength(str, Enum):
    EXPLICIT = "explicit"
    SUPPORTED = "supported"
    UNCERTAIN = "uncertain"


class Groundedness(str, Enum):
    GROUNDED = "grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    UNSUPPORTED = "unsupported"
