"""Normalization for retrieval/indexing only.

Allowed: unicode NFKC, whitespace cleanup, removal of clearly non-semantic
transcription markup like "[pause]", "[inaudible]".
Forbidden: paraphrase, dropping negations/numbers/entities/emotion.
Raw verbatim text is always preserved alongside the retrieval representation.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

from app.domain.models import NormalizedTranscript, NormalizedTurn

_TRANSCRIPTION_MARKUP = re.compile(r"\[(?:pause|pauses|inaudible|silence|laughter|coughs?|noise|music)\]", re.IGNORECASE)
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _TRANSCRIPTION_MARKUP.sub(" ", text)
    text = text.replace("\u00a0", " ")
    text = _WS.sub(" ", text).strip()
    return text


def normalize_record(rec: dict, *, source_hash: str = "") -> NormalizedTranscript:
    tid = str(rec.get("transcript_id")).strip()
    turns: list[NormalizedTurn] = []
    skipped = 0
    speakers: set[str] = set()
    idx = 0
    for t in rec.get("conversation") or []:
        if not isinstance(t, dict):
            skipped += 1
            continue
        speaker = t.get("speaker")
        text = t.get("text")
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue
        label = speaker.strip() if isinstance(speaker, str) and speaker.strip() else "Unknown"
        speakers.add(label)
        # Preserve verbatim text (stripped of surrounding ws only); normalize a
        # separate retrieval copy. Never merge turns across speakers here.
        verbatim = text.strip()
        turns.append(
            NormalizedTurn(index=idx, speaker=label, text=verbatim, retrieval_text=normalize_text(verbatim))
        )
        idx += 1
    # Speaker anomaly: single speaker only, or any non Agent/Customer/Unknown label.
    known = {"agent", "customer", "unknown"}
    anomaly = len(speakers) <= 1 or any(s.lower() not in known for s in speakers)
    return NormalizedTranscript(
        transcript_id=tid,
        domain=rec.get("domain"),
        intent=rec.get("intent"),
        reason_for_call=rec.get("reason_for_call"),
        turns=turns,
        skipped_empty_turns=skipped,
        speaker_anomaly=anomaly,
        source_hash=source_hash,
    )


def record_fingerprint(rec: dict) -> str:
    return hashlib.sha256(str(rec.get("transcript_id")).encode()).hexdigest()[:16]
