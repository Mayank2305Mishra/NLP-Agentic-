from system2.store import build_match, quote_term


def test_quote_term_strips_dquotes():
    assert quote_term('hello "world"') == '"hello world"'
    assert quote_term("   ") == ""


def test_build_match_and_or():
    assert build_match(["refund", "policy"], "AND") == '"refund" AND "policy"'
    assert build_match(["refund", "policy"], "OR") == '"refund" OR "policy"'
    assert build_match([], "AND") == ""


def test_fts_returns_matching_conversation(store):
    rows = store.search_conversations('"duplicate" AND "charge"', limit=5)
    assert rows
    assert rows[0]["transcript_id"] == "conv-refund"


def test_fts_turn_search_restricted_to_conversations(store):
    rows = store.search_turns('"refund"', ["conv-refund"], limit=10)
    assert rows
    assert all(r["transcript_id"] == "conv-refund" for r in rows)
    assert store.search_turns('"refund"', ["conv-booking"], limit=10) == []


def test_metadata_tables(store):
    assert "Cancellations" in store.intents()
    meta = store.fetch_meta(["conv-refund"])
    assert meta["conv-refund"]["intent"] == "Billing & Payments"
    turns = store.fetch_turns(["conv-refund"])
    assert len(turns["conv-refund"]) == 8


def test_bm25_ranks_relevant_conversation_first(store):
    rows = store.search_conversations('"reservation" AND "wrong" AND "date"', limit=5)
    assert rows[0]["transcript_id"] == "conv-booking"
