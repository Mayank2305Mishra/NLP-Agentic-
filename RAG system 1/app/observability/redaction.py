"""PII redaction for logs/UI diagnostics (never mutates the retrieval corpus)."""
from __future__ import annotations

import re

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Alphanumeric lookarounds keep digit runs inside UUIDs / chunk IDs / timestamps
# (e.g. "...af92-7399046a59f5:chunk:1") out of the phone pattern.
PHONE = re.compile(
    r"(?<![0-9A-Za-z])(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}(?![0-9A-Za-z])")
LONG_ID = re.compile(r"\b(?:booking|policy|reference|confirmation|reservation)[\s_#-]*[A-Za-z0-9-]{4,}\b", re.IGNORECASE)
GENERIC_ID = re.compile(r"\b[A-Z]{2,}-?\d{5,}\b")
API_KEY = re.compile(r"(?i)(api[_-]?key|bearer|cookie)\s*[:=]\s*\S+")

# Source/citation identifiers are never PII: AGENTS.md 27 says log source IDs
# (not conversation text), and redacting them destroys traceability.
ID_KEYS = {"source_id", "chunk_id", "transcript_id", "supporting_evidence",
           "dropped_chunk_ids", "allowed_ids", "valid_ids", "bad_ids",
           "request_id", "correlation_id", "run_id", "chunking_config_hash",
           "index_manifest_hash", "source_sha256"}


def redact(text: str) -> str:
    if not text:
        return text
    text = EMAIL.sub("[redacted-email]", text)
    text = PHONE.sub("[redacted-phone]", text)
    text = LONG_ID.sub("[redacted-id]", text)
    text = GENERIC_ID.sub("[redacted-id]", text)
    text = API_KEY.sub("[redacted-secret]", text)
    return text


def redact_dict(d: dict) -> dict:
    out: dict = {}
    for k, v in d.items():
        if k in ID_KEYS:
            out[k] = v
        elif isinstance(v, str) and any(s in k.lower() for s in ("authoriz", "key", "cookie", "token", "secret")):
            out[k] = "[redacted]"
        elif isinstance(v, str):
            out[k] = redact(v)
        elif isinstance(v, dict):
            out[k] = redact_dict(v)
        elif isinstance(v, list):
            out[k] = [redact_dict(x) if isinstance(x, dict)
                      else (x if k in ID_KEYS else (redact(x) if isinstance(x, str) else x))
                      for x in v]
        else:
            out[k] = v
    return out
