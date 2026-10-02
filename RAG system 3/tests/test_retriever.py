from system3.config import RetrievalConfig, EvidenceConfig, Config, DatasetConfig, LoggingConfig
from system3.decisions import DecisionLog
from system3.gate import causal_chain_incomplete, evaluate_evidence_gate
from system3.observability import StageTimer
from system3.query import QueryProcessor
from system3.retriever import merge_windows, RetrievalSettings, Retriever, scoring_config
from system3.schema import ConversationCandidate, EvidenceConversation, EvidenceItem


def _retriever(store, config) -> Retriever:
    return Retriever(store, config)


def _plan(q: str, task=None):
    return QueryProcessor(intents=["Billing & Payments", "Bookings", "Cancellations"]).process(q, task)


def test_merge_windows_overlapping_and_adjacent():
    assert merge_windows([(1, 3), (3, 5), (8, 9)]) == [(1, 5), (8, 9)]
    assert merge_windows([]) == []


def test_settings_ablation_presets():
    cfg = Config(retrieval=RetrievalConfig())
    cfg.ablation = {"mode": "pure_bm25"}
    settings = RetrievalSettings.from_config(cfg)
    assert settings.metadata is False and settings.causal is False
    assert settings.neighbor_window == 0 and settings.gate is False
    cfg.ablation = {"mode": "full"}
    full = RetrievalSettings.from_config(cfg)
    assert full.metadata and full.causal and full.gate


def test_hierarchical_retrieval_selects_expected_conversation(store, config):
    retriever = _retriever(store, config)
    timer = StageTimer.start()
    decisions = DecisionLog("t1")
    plan = _plan("why was i charged twice and i want my money back")
    result = retriever.retrieve(plan, timer, decisions)
    assert result.candidates
    assert result.candidates[0].transcript_id == "conv-refund"
    assert [e.transcript_id for e in result.evidence] == ["conv-refund"]
    assert result.pack is not None
    assert result.pack.allowed_transcript_ids == ["conv-refund"]
    trace = timer.finish()
    for key in (
        "fts_retrieval_ms",
        "conversation_ranking_ms",
        "turn_ranking_ms",
        "neighbor_expansion_ms",
        "evidence_gate_ms",
        "evidence_pack_ms",
        "total_ms",
    ):
        assert trace[key] is not None


def test_never_crosses_conversation_boundary(store, config):
    retriever = _retriever(store, config)
    plan = _plan("i want a refund for the duplicate charge")
    result = retriever.retrieve(plan, StageTimer.start(), DecisionLog("t2"))
    ids = {t.transcript_id for e in result.evidence for t in e.turns}
    assert len(ids) == 1


def test_neighbor_expansion_produces_contiguous_windows(store, config):
    retriever = _retriever(store, config)
    plan = _plan("duplicate charge refund")
    result = retriever.retrieve(plan, StageTimer.start(), DecisionLog("t3"))
    evidence = result.evidence[0]
    for window in evidence.windows:
        assert window.start_turn <= window.end_turn
    turn_ids = [t.turn_id for t in evidence.turns]
    assert turn_ids == sorted(turn_ids)
    assert len(turn_ids) >= 2


def test_causal_scoring_components_observable(store, config):
    retriever = _retriever(store, config)
    plan = _plan("why was the customer charged twice")
    result = retriever.retrieve(plan, StageTimer.start(), DecisionLog("t4"))
    top = result.candidates[0]
    assert set(top.components) >= {"bm25", "metadata", "phrase", "causal", "outcome"}


def test_gate_insufficient_when_no_candidates(config):
    decisions = DecisionLog("t5")
    gate = evaluate_evidence_gate([], config, decisions, causal_gap=False)
    assert gate.decision == "INSUFFICIENT"


def _conv(score=0.5, turn_score=0.5, turns_text="because of the charge"):
    item = EvidenceItem(
        transcript_id="x",
        turn_id=1,
        speaker="customer",
        speaker_original="Customer",
        text_original=turns_text,
        text_normalized=turns_text,
        score=turn_score,
    )
    return EvidenceConversation(
        transcript_id="x",
        score=score,
        meta={},
        turns=[item],
        windows=[],
        components={},
    )


def test_gate_sufficient_and_expand(config):
    decisions = DecisionLog("t6")
    gate = evaluate_evidence_gate([_conv()], config, decisions, causal_gap=False)
    assert gate.decision == "SUFFICIENT"
    decisions = DecisionLog("t7")
    gate = evaluate_evidence_gate([_conv()], config, decisions, causal_gap=True)
    assert gate.decision == "EXPAND"


def test_gate_insufficient_on_low_scores(config):
    low = Config(
        retrieval=RetrievalConfig(),
        evidence=EvidenceConfig(min_score=0.9, min_conversation_score=0.9),
    )
    decisions = DecisionLog("t8")
    gate = evaluate_evidence_gate([_conv(score=0.1, turn_score=0.1)], low, decisions, False)
    assert gate.decision == "INSUFFICIENT"


def test_causal_chain_incomplete():
    assert causal_chain_incomplete([])
    complete = _conv(turns_text="because the charge was wrong i will process a refund and resolved it")
    assert causal_chain_incomplete([complete]) is False
    incomplete = _conv(turns_text="hello there")
    assert causal_chain_incomplete([incomplete]) is True


def test_scoring_config_overrides():
    cfg = Config(retrieval=RetrievalConfig(metadata_boost=True, causal_scoring=True))
    settings = RetrievalSettings(metadata=False, causal=False)
    view = scoring_config(cfg, settings)
    assert view.metadata_boost is False and view.causal_scoring is False
    assert cfg.retrieval.metadata_boost is True


def test_retriever_is_reentrant_across_threads(store, config):
    """One retriever instance must serve concurrent requests without
    per-request state leaking between them (Streamlit, parallel benchmark)."""
    from concurrent.futures import ThreadPoolExecutor

    retriever = _retriever(store, config)
    queries = [
        "why was i charged twice and i want my money back",
        "i want to cancel my policy because it is too expensive",
        "the reservation shows the wrong date after the booking error",
    ]

    def run(query: str):
        plan = _plan(query)
        result = retriever.retrieve(plan, StageTimer.start(), DecisionLog("conc"))
        return (
            tuple(c.transcript_id for c in result.candidates),
            tuple(e.transcript_id for e in result.evidence),
            result.gate.decision,
            tuple(t.turn_id for e in result.evidence for t in e.turns),
        )

    sequential = [run(q) for q in queries]
    with ThreadPoolExecutor(max_workers=6) as pool:
        concurrent = list(pool.map(run, queries * 4))
    for idx, outcome in enumerate(concurrent):
        assert outcome == sequential[idx % len(queries)]


def test_retrieval_fetches_turns_once_per_request(store, config, monkeypatch):
    """The turn fetch must be hoisted: evidence building and gate expansion
    reuse it instead of re-querying identical rows."""
    from system3.store import LexicalStore

    calls: list[int] = []
    original = LexicalStore.fetch_turns

    def counting(self, transcript_ids, include_post_call=False):
        calls.append(len(transcript_ids))
        return original(self, transcript_ids, include_post_call=include_post_call)

    monkeypatch.setattr(LexicalStore, "fetch_turns", counting)
    retriever = _retriever(store, config)
    plan = _plan("duplicate charge refund")
    result = retriever.retrieve(plan, StageTimer.start(), DecisionLog("fetch1"))
    assert result.gate.decision != "INSUFFICIENT"
    assert len(calls) == 1, f"expected a single turn fetch, got {len(calls)}"

    calls.clear()
    texts: list[int] = []
    original_texts = LexicalStore.fetch_conversation_texts

    def counting_texts(self, transcript_ids):
        texts.append(len(transcript_ids))
        return original_texts(self, transcript_ids)

    monkeypatch.setattr(LexicalStore, "fetch_conversation_texts", counting_texts)
    retriever.retrieve(plan, StageTimer.start(), DecisionLog("fetch2"))
    assert len(texts) == 1, f"expected one batched text fetch, got {len(texts)}"
