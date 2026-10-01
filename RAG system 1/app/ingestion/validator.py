"""Validation: fail closed on malformed data, skip only the bad record when safe.

Rules (AGENTS.md section 31):
- duplicate transcript_id -> fail unless an explicit dedupe policy is set.
- empty conversation -> excluded from index, reason recorded.
- unusable turn (missing speaker/text) -> turn skipped; record preserved if any
  usable turn remains; otherwise the record is excluded.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RecordReport:
    transcript_id: str | None
    ok: bool
    reason: str = ""
    skipped_empty_turns: int = 0


@dataclass
class ValidationSummary:
    total: int = 0
    usable: int = 0
    excluded: int = 0
    duplicate_ids: list[str] = field(default_factory=list)
    reports: list[RecordReport] = field(default_factory=list)


def validate_records(
    records: list[dict],
    *,
    on_duplicate: str = "fail",  # fail | keep_first
) -> tuple[list[dict], ValidationSummary]:
    summary = ValidationSummary(total=len(records))
    seen: dict[str, int] = {}
    usable: list[dict] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            summary.excluded += 1
            summary.reports.append(RecordReport(None, False, f"record {i} is not an object"))
            continue
        tid = rec.get("transcript_id")
        if not isinstance(tid, str) or not tid.strip():
            summary.excluded += 1
            summary.reports.append(RecordReport(None, False, f"record {i} missing transcript_id"))
            continue
        tid = tid.strip()
        if tid in seen:
            summary.duplicate_ids.append(tid)
            if on_duplicate == "fail":
                # Fail closed: duplicates are a source-identity violation.
                from app.domain.errors import DataValidationError

                raise DataValidationError(
                    f"Duplicate transcript_id: {tid}",
                    context={"transcript_id": tid, "first_index": seen[tid], "second_index": i},
                )
            summary.excluded += 1
            summary.reports.append(RecordReport(tid, False, "duplicate transcript_id (keep_first)"))
            continue
        seen[tid] = i
        conv = rec.get("conversation")
        if not isinstance(conv, list) or len(conv) == 0:
            summary.excluded += 1
            summary.reports.append(RecordReport(tid, False, "empty or missing conversation"))
            continue
        good_turns = 0
        skipped = 0
        for t in conv:
            if not isinstance(t, dict):
                skipped += 1
                continue
            speaker = t.get("speaker")
            text = t.get("text")
            if not isinstance(text, str) or not text.strip():
                skipped += 1
                continue
            if not isinstance(speaker, str) or not speaker.strip():
                skipped += 1  # missing speaker: skip turn, keep record if others usable
                continue
            good_turns += 1
        if good_turns == 0:
            summary.excluded += 1
            summary.reports.append(RecordReport(tid, False, "no usable turns", skipped))
            continue
        summary.usable += 1
        summary.reports.append(RecordReport(tid, True, "", skipped))
        usable.append(rec)
    return usable, summary
