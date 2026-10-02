from system3.evidence import MULTI_HEADER, pack_evidence, turns_index
from system3.parsing import extract_json, parse_answer
from system3.grounding import support_ratio, validate_grounding, validate_schema
from system3.decisions import DecisionLog
from system3.schema import EvidenceConversation, EvidenceItem, EvidencePack, EvidenceWindow, QueryPlan


def _plan(task="task1", comparison=False) -> QueryPlan:
    return QueryPlan(
        raw_query="q",
        query_id="id",
        normalized_query="q",
        terms=["refund"],
        phrases=[],
        causal_terms=[],
        event_terms=[],
        outcome_terms=[],
        expanded_terms=[],
        metadata_hints={},
        explicit_intent=None,
        task=task,
        comparison=comparison,
    )


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_surrounding_prose():
    raw = 'Sure! Here is the answer:\n{"task": "task1", "confidence": 0.5}\nHope that helps.'
    assert extract_json(raw)["task"] == "task1"


def test_extract_json_failure():
    assert extract_json("no json here") is None
    answer, errors = parse_answer("")
    assert answer is None and errors


def test_schema_validation_rejects_bad_strength():
    bad = {"task": "task1", "causal_rationales": [{"cause": "a", "effect": "b", "strength": "guess", "evidence_turns": [1]}], "confidence": 0.5, "insufficient_evidence": False}
    issues = validate_schema(bad, "task1")
    assert any("strength" in i for i in issues)


def test_support_ratio():
    assert support_ratio("refund because duplicate charge", "i want a refund due to the duplicate charge") >= 0.5
    assert support_ratio("quantum entanglement", "hello there") < 0.3


def _conv(turn_texts, transcript_id="t1"):
    items = [
        EvidenceItem(
            transcript_id=transcript_id,
            turn_id=i,
            speaker="customer",
            speaker_original="Customer",
            text_original=text,
            text_normalized=text,
            score=0.5,
        )
        for i, text in enumerate(turn_texts)
    ]
    return EvidenceConversation(
        transcript_id=transcript_id,
        score=0.7,
        meta={"intent": "x", "domain": "Insurance", "reason_for_call": "r"},
        turns=items,
        windows=[EvidenceWindow(transcript_id, 0, len(turn_texts) - 1, 0.5, [0])],
        components={"bm25": 0.7},
    )


def test_pack_format_and_budget_drops_low_score_windows():
    items = []
    for i, text in enumerate(["lead in problem statement", "agent asks a question", "customer explains cause"]):
        items.append(
            EvidenceItem(
                transcript_id="t1",
                turn_id=i,
                speaker="customer",
                speaker_original="Customer",
                text_original=text,
                text_normalized=text,
                score=0.9 - 0.1 * i,
            )
        )
    for i, text in enumerate(["unrelated later chatter about weather", "small talk continues"], start=10):
        items.append(
            EvidenceItem(
                transcript_id="t1",
                turn_id=i,
                speaker="agent",
                speaker_original="Agent",
                text_original=text,
                text_normalized=text,
                score=0.2,
            )
        )
    conv = EvidenceConversation(
        transcript_id="t1",
        score=0.7,
        meta={"intent": "x", "domain": "Insurance", "reason_for_call": "r"},
        turns=items,
        windows=[
            EvidenceWindow("t1", 0, 2, 0.9, [0]),
            EvidenceWindow("t1", 10, 11, 0.2, [10]),
        ],
        components={"bm25": 0.7},
    )
    idx = turns_index([conv])
    pack = pack_evidence([conv], "task1", max_tokens=60, turns_by_conv=idx)
    assert "CONVERSATION: t1" in pack.text
    assert "TURN 000 | CUSTOMER" in pack.text
    assert "lead in problem statement" in pack.text
    assert "unrelated later chatter about weather" not in pack.text
    assert pack.token_estimate <= 60
    assert pack.conversations[0].windows[0].score >= 0.9


def test_pack_multi_conversation_separator():
    a = _conv(["alpha refund text"], "ta")
    b = _conv(["beta cancellation text"], "tb")
    idx = turns_index([a, b])
    pack = pack_evidence([a, b], "task2", max_tokens=500, turns_by_conv=idx)
    assert MULTI_HEADER.strip().split("\n")[0] in pack.text
    assert "CONVERSATION: ta" in pack.text and "CONVERSATION: tb" in pack.text
    assert pack.allowed_transcript_ids == ["ta", "tb"]


def test_pack_single_conversation_by_default_ordering():
    a = _conv(["alpha"], "ta")
    b = _conv(["beta"], "tb")
    idx = turns_index([a, b])
    pack = pack_evidence([a, b], "task1", max_tokens=1000, turns_by_conv=idx)
    assert pack.allowed_transcript_ids[0] == "ta"


def test_grounding_accepts_valid_answer():
    conv = _conv(["the customer wanted a refund because of the duplicate charge", "agent processed refund"])
    pack = EvidencePack("task1", [conv], 50, ["t1"], "text")
    answer = {
        "task": "task1",
        "causal_rationales": [
            {"cause": "duplicate charge", "effect": "refund processed", "mechanism": None, "evidence_turns": [0, 1], "strength": "explicit"}
        ],
        "confidence": 0.8,
        "insufficient_evidence": False,
    }
    report = validate_grounding(answer, "task1", _plan(), pack, _cfg(), DecisionLog("g1"))
    assert report.passed and not report.contamination


def test_grounding_flags_fabricated_turns():
    conv = _conv(["refund text"])
    pack = EvidencePack("task1", [conv], 50, ["t1"], "text")
    answer = {
        "task": "task1",
        "causal_rationales": [{"cause": "x", "effect": "y", "mechanism": None, "evidence_turns": [99], "strength": "explicit"}],
        "confidence": 0.5,
        "insufficient_evidence": False,
    }
    report = validate_grounding(answer, "task1", _plan(), pack, _cfg(), DecisionLog("g2"))
    assert not report.passed
    assert any("not present" in i for i in report.issues)


def test_grounding_detects_contamination():
    a = _conv(["alpha refund", "more alpha"], "ta")
    b_items = [
        EvidenceItem(
            transcript_id="tb",
            turn_id=5,
            speaker="agent",
            speaker_original="Agent",
            text_original="beta cancel outcome",
            text_normalized="beta cancel outcome",
            score=0.5,
        ),
        EvidenceItem(
            transcript_id="tb",
            turn_id=6,
            speaker="agent",
            speaker_original="Agent",
            text_original="second beta line",
            text_normalized="second beta line",
            score=0.5,
        ),
    ]
    b = EvidenceConversation(
        transcript_id="tb",
        score=0.7,
        meta={},
        turns=b_items,
        windows=[EvidenceWindow("tb", 5, 6, 0.5, [5])],
        components={},
    )
    pack = EvidencePack("task1", [a, b], 50, ["ta", "tb"], "text")

    mixed = {
        "task": "task1",
        "causal_rationales": [
            {"cause": "alpha refund", "effect": "beta cancel outcome", "mechanism": None, "evidence_turns": [0, 5], "strength": "explicit"}
        ],
        "confidence": 0.5,
        "insufficient_evidence": False,
    }
    report = validate_grounding(mixed, "task1", _plan(), pack, _cfg(), DecisionLog("g4"))
    assert report.contamination

    single = {
        "task": "task1",
        "causal_rationales": [
            {"cause": "alpha refund", "effect": "alpha effect", "mechanism": None, "evidence_turns": [0], "strength": "explicit"}
        ],
        "confidence": 0.5,
        "insufficient_evidence": False,
    }
    plan = _plan()
    plan.comparison = True
    report2 = validate_grounding(single, "task1", plan, pack, _cfg(), DecisionLog("g5"))
    assert not report2.contamination


def _cfg():
    from system3.config import Config

    return Config()
