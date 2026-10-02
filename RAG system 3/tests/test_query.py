from system3.query import QueryProcessor, classify_task, detect_comparison


def make_processor() -> QueryProcessor:
    return QueryProcessor(intents=["Billing & Payments", "Cancellations"], domains=["Insurance"])


def test_task_classification():
    assert classify_task("Why was the customer unhappy?") == "task1"
    assert classify_task("Summarize the entire call end to end") == "task2"
    assert classify_task("Compare the two plans the customers discussed") == "task2"
    assert detect_comparison("compare the conversations about refunds") is True
    assert detect_comparison("The customer compared options during the call") is False


def test_query_plan_structure():
    plan = make_processor().process("Why did the customer want a refund?")
    assert plan.terms
    assert plan.causal_terms
    assert "refund" in plan.terms
    assert plan.fts_match and '"' in plan.fts_match
    assert plan.fts_match_loose
    assert plan.task == "task1"


def test_inferred_intent_is_a_hint_not_a_hard_filter():
    plan = make_processor().process("what happens during a cancellation request")
    assert plan.explicit_intent is None
    assert plan.metadata_hints["intent"] == "Cancellations"


def test_role_word_does_not_infer_customer_trust_intent():
    plan = QueryProcessor(intents=["Customer trust"]).process(
        "Why was the customer unhappy?"
    )
    assert plan.explicit_intent is None
    assert "intent" not in plan.metadata_hints


def test_expansion_is_deterministic():
    first = make_processor().process("i want a refund")
    second = make_processor().process("i want a refund")
    assert first.expanded_terms == second.expanded_terms
    assert "reimbursement" in first.expanded_terms


def test_quoted_phrase_kept():
    plan = make_processor().process('the "money back guarantee" question')
    assert any("money back guarantee" in p for p in plan.phrases)


def test_stopwords_excluded_from_retrieval_terms():
    plan = make_processor().process("why did the customer cancel the policy")
    assert "why" not in plan.fts_match
    assert "cancel" in plan.fts_match
