from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from ..config import Config
from ..textnorm import normalize_text

log = logging.getLogger("system2.fixture")

CATEGORIES: list[dict[str, Any]] = [
    {
        "category": "refund",
        "anchors": ["refund", "reimburse", "money back"],
        "question": "Why did the customer ask for a refund, what did the agent do, and what was the outcome?",
    },
    {
        "category": "cancellation",
        "anchors": ["cancel", "cancellation"],
        "question": "Why did the customer want to cancel, and how did the agent respond?",
    },
    {
        "category": "booking_error",
        "anchors": ["wrong booking", "booking error", "incorrect reservation", "rebook", "wrong reservation"],
        "question": "What booking problem did the customer report and how did the agent resolve it?",
    },
    {
        "category": "pricing_dissatisfaction",
        "anchors": ["overcharged", "charged too much", "too expensive", "price is too high", "unexpected charge"],
        "question": "Why was the customer unhappy about pricing and what did the agent offer?",
    },
    {
        "category": "plan_comparison",
        "anchors": ["compare", "other plan", "different plan", "versus", "better plan"],
        "question": "The customer compared options — what influenced their decision?",
    },
    {
        "category": "upsell",
        "anchors": ["upgrade", "upsell", "add-on", "premium plan"],
        "question": "Did the agent propose an upgrade, and how did the customer react?",
    },
    {
        "category": "product_defect",
        "anchors": ["not working", "broken", "defect", "damaged", "malfunction"],
        "question": "What defect did the customer report, and what resolution was offered?",
    },
    {
        "category": "service_complaint",
        "anchors": ["complaint", "unhappy", "frustrated", "dissatisfied", "poor service"],
        "question": "What service issue did the customer complain about, and how was it handled?",
    },
]


def build_fixture(config: Config, count: int = 64, seed: int = 7) -> Path:
    turns_path = config.dataset.turns_jsonl
    if not turns_path.exists():
        raise FileNotFoundError("processed turns missing — run build-data first")

    allowed = _population(config)
    per_category = max(1, count // len(CATEGORIES))
    matches: dict[str, dict[str, list[int]]] = {c["category"]: {} for c in CATEGORIES}
    anchors_by_cat = {c["category"]: [normalize_text(a) for a in c["anchors"]] for c in CATEGORIES}
    reasons: dict[str, str] = {}

    with open(turns_path, "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("is_post_call_artifact"):
                continue
            if allowed is not None and row["transcript_id"] not in allowed:
                continue
            text = row["text_normalized"]
            tid = row["transcript_id"]
            if not reasons.get(tid) and row.get("reason_for_call"):
                reasons[tid] = row["reason_for_call"]
            for cat, anchors in anchors_by_cat.items():
                bucket = matches[cat]
                if tid in bucket and row["turn_id"] in bucket[tid]:
                    continue
                if any(a in text for a in anchors):
                    bucket.setdefault(tid, []).append(row["turn_id"])

    rng = random.Random(seed)
    questions: list[dict[str, Any]] = []
    q_index = 1
    for category in CATEGORIES:
        cat = category["category"]
        transcript_ids = sorted(matches[cat].keys())
        rng.shuffle(transcript_ids)
        taken = 0
        for tid in transcript_ids:
            if taken >= per_category:
                break
            expected_turns = sorted(matches[cat][tid])[:6]
            if not expected_turns:
                continue
            reason = (reasons.get(tid) or "").strip()
            context = f" Caller context: {reason}" if reason else ""
            questions.append(
                {
                    "question_id": f"q{q_index:03d}",
                    "category": cat,
                    "question": category["question"] + context,
                    "transcript_id": tid,
                    "task": "task1",
                    "expected_evidence_turns": expected_turns,
                    "anchor_terms": category["anchors"],
                    "expected_cause": category["anchors"][0],
                    "expected_effect": None,
                }
            )
            q_index += 1
            taken += 1
        if taken < per_category:
            log.warning("category %s only produced %d/%d questions", cat, taken, per_category)

    if len(questions) < 50:
        raise RuntimeError(f"fixture too small: {len(questions)} questions (minimum 50)")

    out_dir = Path(__file__).resolve().parents[3] / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fixture_questions.json"
    payload = {
        "version": 1,
        "seed": seed,
        "count": len(questions),
        "source_hash": _source_hash(config),
        "questions": questions,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    log.info("wrote fixture with %d questions to %s", len(questions), out_path)
    return out_path


def load_fixture(config: Config, path: str | None = None) -> dict[str, Any]:
    fixture_path = Path(path) if path else Path(__file__).resolve().parents[3] / "benchmarks" / "fixture_questions.json"
    with open(fixture_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _population(config: Config) -> set[str] | None:
    manifest_path = config.dataset.sampling_manifest
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        return set(manifest.get("selected_transcript_ids") or [])
    return None


def _source_hash(config: Config) -> str:
    meta = config.dataset.processed_path / "source_meta.json"
    if meta.exists():
        with open(meta, "r", encoding="utf-8") as fh:
            return json.load(fh).get("source_hash", "")
    return ""
