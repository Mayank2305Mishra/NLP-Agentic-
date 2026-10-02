"""Unit tests: causal roles, turn selection, expansion, compression, packing (§13–§16, §21, §22)."""

from __future__ import annotations

from system4.config.settings import Settings
from system4.evidence.causal_scoring import contradiction_count, detect_roles
from system4.evidence.compressor import compress, is_filler
from system4.evidence.packer import estimate_tokens, pack
from system4.evidence.selection import expand, select
from system4.schema import CompressionLevel, Evidence, Turn
from tests.conftest import conv

SETTINGS = Settings(turns_per_conversation=6, neighbor_window=2)


def _evidence(turn: Turn, roles=None) -> Evidence:
    return Evidence(turn=turn, score=1.0, roles=roles or ["cause"], reason="test")


def test_detect_roles_maps_cause_problem_action_outcome():
    conversation = conv("t", [
        (1, "Customer", "I cannot access my account because the password reset failed"),
        (2, "Agent", "I can reset your account and will process it now"),
        (3, "Customer", "That fixed it, thank you"),
    ])
    roles0 = [r for r, _ in detect_roles(conversation.turns[0], {"password"}, SETTINGS)]
    roles1 = [r for r, _ in detect_roles(conversation.turns[1], set(), SETTINGS)]
    roles2 = [r for r, _ in detect_roles(conversation.turns[2], set(), SETTINGS)]
    assert "problem" in roles0 and "cause" in roles0
    assert "agent_action" in roles1
    assert "outcome" in roles2


def test_mechanism_never_inferred_from_adjacency():
    conversation = conv("t", [
        (1, "Customer", "because of the failed charge I want help"),
        (2, "Customer", "so the money disappeared"),
    ])
    # A customer turn adjacent to cause/effect must not become a mechanism (§15).
    for turn in conversation.turns:
        roles = [r for r, _ in detect_roles(turn, set(), SETTINGS)]
        assert "mechanism" not in roles
    # Even an agent turn without explicit explanatory markers has no mechanism.
    agent = conv("t2", [(1, "Agent", "okay sure I will do that")]).turns[0]
    assert "mechanism" not in [r for r, _ in detect_roles(agent, set(), SETTINGS)]


def test_selection_retains_provenance_and_excludes_post_call():
    conversation = conv("t", [
        (1, "Customer", "I need a refund because the order was cancelled"),
        (2, "Agent", "I can approve the refund"),
        (3, "Customer", "Internal note: refund approved by supervisor"),  # post-call-ish
        (4, "Customer", "okay thanks"),
    ])
    conversation.turns[2] = Turn("t", 3, "Customer", "Internal note: refund approved by supervisor",
                                 post_call=True)
    selected = select("refund cancelled", conversation, SETTINGS)
    assert all(not item.turn.post_call for item in selected)
    assert all(item.reason and item.roles for item in selected)
    assert all(item.turn.transcript_id == "t" for item in selected)


def test_expansion_uses_local_window_only():
    turns = [(i, "Customer", f"turn {i} about the refund") for i in range(1, 31)]
    conversation = conv("t", turns)
    seed = [Evidence(turn=conversation.turns[19], score=1.0, roles=["cause"], reason="seed")]
    expanded = expand(seed, conversation, SETTINGS)
    ids = [item.turn.turn_id for item in expanded]
    assert ids == [18, 19, 20, 21, 22]  # window ±2 around turn 20, never the whole conversation (§14)
    by_id = {item.turn.turn_id: item for item in expanded}
    assert by_id[20].retrieval_stage == "turn_selection"  # seed keeps its stage
    assert all(by_id[n].retrieval_stage == "expansion" for n in (18, 19, 21, 22))


def test_contradiction_detection():
    a = conv("a", [(1, "Customer", "was my refund approved")])
    b = conv("b", [(1, "Agent", "your refund was denied")])
    items = [_evidence(a.turns[0], ["cause"]), _evidence(b.turns[0], ["outcome"])]
    assert contradiction_count(items) == 1
    clean = [_evidence(conv("c", [(1, "Agent", "your refund was approved")]).turns[0], ["outcome"])]
    assert contradiction_count(clean) == 0


def test_compressor_drops_filler_keeps_causal():
    filler = _evidence(conv("t", [(1, "Agent", "thank you for calling")]).turns[0], ["context"])
    causal = _evidence(conv("t", [(2, "Customer", "the refund failed because of the system")]).turns[0],
                       ["cause"])
    assert is_filler(filler)
    kept, stats = compress([filler, causal], CompressionLevel.LIGHT, 1000, SETTINGS)
    assert kept == [causal]
    assert stats["dropped_filler"] == 1
    assert stats["input_tokens"] >= stats["output_tokens"]


def test_compressor_budget_and_none_level():
    many = [
        _evidence(conv("t", [(i, "Customer", f"issue number {i} because of delay " * 5)]).turns[0],
                  ["cause"])
        for i in range(1, 30)
    ]
    kept_none, stats_none = compress(many, CompressionLevel.NONE, 50, SETTINGS)
    assert len(kept_none) == len(many)
    kept, stats = compress(many, CompressionLevel.AGGRESSIVE, 80, SETTINGS)
    assert len(kept) < len(many)
    assert stats["output_tokens"] <= stats["input_tokens"]
    assert stats["dropped_turns"], "dropped turn ids recorded for audit (§21)"


def test_compressor_never_empties_evidence():
    only = [_evidence(conv("t", [(1, "Agent", "hello")]).turns[0], ["context"])]
    kept, _ = compress(only, CompressionLevel.AGGRESSIVE, 1, SETTINGS)
    assert kept, "compression must keep the strongest item"


def test_pack_format_budget_and_chronology():
    conversation = conv("t", [
        (1, "Customer", "why was the refund denied"),
        (2, "Agent", "I approved it after review"),
        (3, "Customer", "that fixed the issue"),
    ])
    items = select("refund denied fixed", conversation, Settings(turns_per_conversation=6))
    packed, chosen = pack(items, 10000, SETTINGS)
    assert packed.startswith("SOURCE: t")
    assert "DOMAIN:" in packed and "INTENT:" in packed
    assert "[Turn 001 | Customer]" in packed
    turns = [item.turn.turn_id for item in chosen]
    assert turns == sorted(turns), "chronology is a correctness property (§22)"
    # Budget cut: tiny budget still returns at least one turn.
    packed_small, chosen_small = pack(items, 1, SETTINGS)
    assert chosen_small and packed_small


def test_pack_multi_source_blocks_have_stable_ids():
    a = _evidence(conv("a", [(1, "Customer", "refund because damaged")]).turns[0])
    b = _evidence(conv("b", [(1, "Agent", "fixed it")]).turns[0])
    packed, chosen = pack([a, b], 10000, SETTINGS)
    assert "SOURCE: a" in packed and "SOURCE: b" in packed
    assert {item.turn.transcript_id for item in chosen} == {"a", "b"}


def test_estimate_tokens_modes():
    text = "hello world this is a test"
    assert estimate_tokens(text, Settings(token_estimator="words")) == 6
    assert estimate_tokens(text, Settings(token_estimator="chars4")) == 7
