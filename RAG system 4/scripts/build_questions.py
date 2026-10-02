"""Deterministically build the benchmark question set (AGENTS_SYSTEM_4.md §35).

Samples the project's transcript subset into a fixed, reproducible question
file covering: direct causal questions, multi-turn causal questions, questions
requiring agent action, questions requiring outcomes, conversations with
similar terminology (shared intents), adversarial cross-conversation
distractors, and unanswerable queries for abstention precision.

Usage:
    python scripts/build_questions.py --source final_transcripts.json \\
        --output benchmarks/questions.json --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_CAUSAL_CUE = re.compile(
    r"\b(because|due to|caused|reason|therefore|so that|resulted|led to|unable|"
    r"issue|problem|resolved|approved|denied|refund|delay)\b",
    re.I,
)
_STOP = set(
    "the a an and or is are was were to of in on for with at by from that this it its "
    "i you your we our they their he she as so if not no about into his her was".split()
)


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if len(w) > 2 and w not in _STOP}


def _hash_query(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def _gold_turns(record: dict, query: str) -> list[int]:
    """Best-effort gold turns: causal cues with query overlap (§31 turn recall)."""
    query_terms = _terms(query)
    hits: list[int] = []
    for index, turn in enumerate(record.get("conversation") or [], 1):
        text = str(turn.get("text") or "")
        if _CAUSAL_CUE.search(text) and len(query_terms & _terms(text)) >= 2:
            hits.append(index)
    return hits[:6]


def _question(qid: str, category: str, task: str, query: str, record: dict | None,
              answerable: bool = True) -> dict:
    return {
        "id": qid,
        "category": category,
        "task": task,
        "query": query,
        "query_hash": _hash_query(query),
        "transcript_id": record["transcript_id"] if record else None,
        "answerable": answerable,
        "expected_turns": _gold_turns(record, query) if record else [],
    }


def _distinct(records: list[dict], count: int, rng: random.Random) -> list[dict]:
    """Sample records across distinct intents for coverage (§35)."""
    by_intent: dict[str, list[dict]] = {}
    for record in records:
        by_intent.setdefault(record.get("intent") or "unknown", []).append(record)
    intents = list(by_intent)
    rng.shuffle(intents)
    picked: list[dict] = []
    while len(picked) < count and intents:
        intent = intents.pop()
        pool = by_intent[intent]
        picked.append(rng.choice(pool))
    if len(picked) < count:
        remaining = [r for r in records if r not in picked]
        picked.extend(rng.sample(remaining, count - len(picked)))
    return picked


def _unanswerable_topics(rng: random.Random, corpus_terms: set[str], count: int) -> list[str]:
    candidates = [
        "volcano eruption travel claim",
        "ski equipment rental dispute",
        "pet grooming appointment refund",
        "guitar lesson cancellation fee",
        "space tourism insurance upgrade",
        "tractor lease deposit dispute",
        "wedding photographer no-show fee",
        "haunted house ticket refund",
    ]
    # Generic support vocabulary can legitimately appear in the corpus; only the
    # topic-specific terms must be verified absent for the query to be unanswerable.
    generic = {
        "refund", "claim", "dispute", "cancellation", "cancel", "fee", "deposit",
        "insurance", "upgrade", "appointment", "rental", "lesson", "ticket", "customer",
        "called", "contact", "support", "why", "did", "get", "their",
    }
    topics: list[str] = []
    for topic in candidates:
        distinctive = _terms(topic) - generic
        if distinctive and not (distinctive & corpus_terms):
            topics.append(topic)
    if len(topics) < count:
        raise SystemExit("not enough verified-unanswerable topics; extend the candidate list")
    rng.shuffle(topics)
    return topics[:count]


def build(records: list[dict], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    corpus_terms: set[str] = set()
    for record in records:
        corpus_terms |= _terms(str(record.get("reason_for_call") or ""))

    questions: list[dict] = []
    pool = _distinct(records, 40, rng)

    # 1. Direct causal questions (§35 "easy direct causal").
    for index in range(8):
        record = pool[index]
        questions.append(_question(
            f"direct-{index + 1:02d}", "direct_causal", "task1",
            str(record["reason_for_call"]), record,
        ))

    # 2. Multi-turn causal questions end to end (§35).
    for index in range(5):
        record = pool[8 + index]
        questions.append(_question(
            f"multiturn-{index + 1:02d}", "multi_turn_causal", "task2",
            f"Summarize the causal chain of this interaction end to end: {record['reason_for_call']}",
            record,
        ))

    # 3. Questions requiring agent action.
    for index in range(5):
        record = pool[13 + index]
        questions.append(_question(
            f"action-{index + 1:02d}", "agent_action", "task1",
            f"What did the agent do to resolve this and why: {record['reason_for_call']}",
            record,
        ))

    # 4. Questions requiring outcomes.
    for index in range(5):
        record = pool[18 + index]
        questions.append(_question(
            f"outcome-{index + 1:02d}", "outcome", "task1",
            f"What was the outcome of this call: {record['reason_for_call']}",
            record,
        ))

    # 5. Similar-terminology conversations: intent phrase only (many siblings share it).
    for index in range(4):
        record = pool[23 + index]
        intent = (record.get("intent") or "support").lower()
        questions.append(_question(
            f"similar-{index + 1:02d}", "similar_terminology", "task1",
            f"Why did the customer contact support about {intent}?",
            record,
        ))

    # 6. Adversarial cross-conversation distractors: gold reason + distractor phrase.
    for index in range(5):
        record = pool[27 + index]
        distractor = pool[32 + (index * 2) % 8]
        if distractor is record:
            distractor = pool[-1]
        distractor_phrase = " ".join(sorted(_terms(str(distractor["reason_for_call"])) - _terms(str(record["reason_for_call"]))))
        query = str(record["reason_for_call"])
        if distractor_phrase:
            query = f"{query} Also determine whether this involved: {distractor_phrase}."
        questions.append(_question(
            f"adversarial-{index + 1:02d}", "adversarial_distractor", "task1",
            query, record,
        ))

    # 7. Unanswerable / incomplete-evidence queries (abstention precision §31).
    for index, topic in enumerate(_unanswerable_topics(rng, corpus_terms, 4), 1):
        questions.append(_question(
            f"unanswerable-{index:02d}", "unanswerable", "task1",
            f"Why did the customer's {topic} get cancelled?",
            None, answerable=False,
        ))

    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fixed benchmark question set")
    parser.add_argument("--source", default="final_transcripts.json")
    parser.add_argument("--output", default="benchmarks/questions.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = json.loads(Path(args.source).read_text(encoding="utf-8"))
    questions = build(records, seed=args.seed)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(questions, indent=2), encoding="utf-8")
    categories: dict[str, int] = {}
    for question in questions:
        categories[question["category"]] = categories.get(question["category"], 0) + 1
    print(f"wrote {len(questions)} questions -> {target}")
    print(json.dumps(categories, indent=2))


if __name__ == "__main__":
    main()
