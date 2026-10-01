from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger("system2.parsing")

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | None:
    if not text or not text.strip():
        return None
    candidates = [text.strip()]
    for match in _FENCE_RE.findall(text):
        candidates.append(match.strip())
    brace = text.find("{")
    if brace != -1:
        candidates.append(text[brace:])
    decoder = json.JSONDecoder()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            obj, _ = decoder.raw_decode(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
        for idx, ch in enumerate(candidate):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(candidate[idx:])
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def parse_answer(raw: str) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    answer = extract_json(raw)
    if answer is None:
        errors.append("no valid JSON object found in model output")
        return None, errors
    return answer, errors
