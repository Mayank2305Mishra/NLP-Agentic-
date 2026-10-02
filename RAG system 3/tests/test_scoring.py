from system3.config import RetrievalConfig
from system3.scoring import conversation_score, min_max, squash_bm25, turn_score
from system3.query import QueryProcessor


def _plan(q: str):
    return QueryProcessor(intents=["Cancellations"]).process(q)


def test_squash_bm25_monotonic():
    assert squash_bm25(-30) > squash_bm25(-10) > squash_bm25(-1)
    assert squash_bm25(0.0) == 0.0


def test_min_max():
    assert min_max([1.0, 1.0]) == [1.0, 1.0]
    assert min_max([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]


def test_turn_score_prefers_causal_language():
    cfg = RetrievalConfig()
    plan = _plan("why did the customer cancel the policy")
    high, _ = turn_score(plan, "the customer cancelled because the price was too high", {}, cfg, -12.0)
    low, _ = turn_score(plan, "hello how can i help you today", {}, cfg, -12.0)
    assert high > low


def test_turn_score_components_are_reported():
    cfg = RetrievalConfig()
    plan = _plan("why did the customer cancel")
    score, comps = turn_score(plan, "i want to cancel because of the charge", {}, cfg, -8.0)
    assert 0.0 <= score <= 1.0
    assert comps.bm25 > 0
    assert comps.causal > 0
    assert comps.to_dict()["bm25_raw"] == -8.0


def test_conversation_score_metadata_boost():
    cfg = RetrievalConfig(metadata_boost=True)
    plan = _plan("what happens during a cancellation")
    meta_match = {"intent": "Cancellations"}
    meta_none = {"intent": "Bookings"}
    score_match, comps_m = conversation_score(plan, "cancellation policy", meta_match, cfg, -10.0)
    score_none, comps_n = conversation_score(plan, "cancellation policy", meta_none, cfg, -10.0)
    assert comps_m.metadata > comps_n.metadata
    assert score_match > score_none
