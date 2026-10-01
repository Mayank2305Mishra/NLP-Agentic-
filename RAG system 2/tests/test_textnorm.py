from system2.textnorm import (
    estimate_tokens,
    infer_speaker_by_position,
    is_post_call_artifact,
    normalize_speaker,
    normalize_text,
)


def test_normalize_text_lowercase_and_whitespace():
    assert normalize_text("  Hello   WORLD\t") == "hello world"


def test_normalize_text_preserves_filler_words():
    assert "um" in normalize_text("hi um i have a question")


def test_normalize_contraction():
    assert normalize_text("i don't know") == "i do not know"


def test_speaker_canonical_roles():
    assert normalize_speaker("Customer") == "customer"
    assert normalize_speaker("Agent") == "agent"
    assert normalize_speaker("Supervisor") == "agent"
    assert normalize_speaker("Fraud Specialist") == "agent"
    assert normalize_speaker("[System Note") == "unknown"
    assert normalize_speaker("") == "unknown"


def test_speaker_position_inference():
    assert infer_speaker_by_position("Isabella Nguyen", "customer") == ("agent", True)
    assert infer_speaker_by_position("Lily Kelly", "agent") == ("customer", True)
    assert infer_speaker_by_position("Agent", None) == ("agent", False)
    assert infer_speaker_by_position("??? weird label ???", None) == ("unknown", False)


def test_post_call_artifact_detection():
    assert is_post_call_artifact("Note", "internal observation")
    assert is_post_call_artifact("Customer", "[Key Events] summary of the call")
    assert not is_post_call_artifact("Customer", "i want a refund please")


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 40) == 10
