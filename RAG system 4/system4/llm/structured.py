"""Structured output parsing (AGENTS_SYSTEM_4.md §27, §28, §51).

Tolerant extraction: JSON object, fenced JSON, or JSON embedded in prose.
Parsing never invents content — failure returns ``None`` so the pipeline can
degrade to its deterministic fallback instead of fabricating a result.
"""

from __future__ import annotations

import json
from typing import Any


def _first_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json(raw: object) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    # Strip markdown fences if present.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    candidate = _first_object(text)
    if candidate:
        try:
            value = json.loads(candidate)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def parse_answer(raw: object, task: str) -> tuple[dict[str, Any], bool]:
    """Return ``(answer, valid)``; the caller's selected task is authoritative."""
    answer = extract_json(raw)
    if answer is None:
        return {"task": task, "insufficient_evidence": True}, False
    answer["task"] = task
    return answer, True
