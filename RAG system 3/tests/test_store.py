from concurrent.futures import ThreadPoolExecutor

from system3.store import build_match, quote_term


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


def test_fetch_conversation_texts_batch(store):
    texts = store.fetch_conversation_texts(["conv-refund", "conv-booking", "missing-id"])
    assert set(texts) == {"conv-refund", "conv-booking"}
    assert "duplicate charge" in texts["conv-refund"]
    assert store.fetch_conversation_texts([]) == {}


def test_thread_local_connections_serve_concurrent_queries(store):
    import threading

    results: list[tuple[int, int]] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        barrier.wait(timeout=5)
        conn = store.conn
        count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        results.append((id(conn), count))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(results) == 4
    assert all(count == 6 for _, count in results)
    # Each thread must have received its own connection object.
    assert len({conn_id for conn_id, _ in results}) == 4
    store.close()
    # After close() the store reopens cleanly on next access.
    assert store.conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 6


def test_parallel_searches_are_consistent(store):
    def search(_: int) -> list[str]:
        rows = store.search_conversations('"duplicate" AND "charge"', limit=5)
        return [r["transcript_id"] for r in rows]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(search, range(16)))
    assert all(r == results[0] for r in results)
    assert results[0] and results[0][0] == "conv-refund"
