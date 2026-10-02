from system2.query import QueryProcessor, classify_task, detect_comparison


def make_processor() -> QueryProcessor:
    return QueryProcessor(intents=["Billing & Payments", "Cancellations"], domains=["Insurance"])


def test_task_classification():
    assert classify_task("Why was the customer unhappy?") == "task1"
    assert classify_task("Summarize the entire call end to end") == "task2"
    assert classify_task("Compare the two plans the customers discussed") == "task2"
    assert detect_comparison("compare the conversations about refunds") is True
    assert detect_comparison("compare the two conversations side by side") is True
    # single-customer "compare options" questions stay conversation-local (§21)
    assert detect_comparison(
        "The customer compared options — what influenced their decision?"
    ) is False
    assert detect_comparison(
        "Customer wanted to compare home insurance plans"
    ) is False


def test_query_plan_structure():
    plan = make_processor().process("Why did the customer want a refund?")
    assert plan.terms
    assert plan.causal_terms
    assert "refund" in plan.terms
    assert plan.fts_match and '"' in plan.fts_match
    assert plan.fts_match_loose
    assert plan.task == "task1"


def test_explicit_intent_detection():
    plan = make_processor().process("what happens during a cancellation request")
    assert plan.explicit_intent == "Cancellations"
    assert plan.metadata_hints["intent"] == "Cancellations"
    # token-level match: soft boost only, never a hard filter (spec §9.1)
    assert plan.intent_match == "token"


def test_verbatim_intent_is_hard_constraint():
    plan = make_processor().process("list all Cancellations calls from Monday")
    assert plan.explicit_intent == "Cancellations"
    assert plan.intent_match == "verbatim"


def test_generic_role_word_never_sets_intent():
    # "customer" must not map to "Customer trust": turning a guessed hint into a
    # hard filter caused catastrophic recall loss (spec §9.1).
    proc = QueryProcessor(intents=["Customer trust", "Billing & Payments"])
    plan = proc.process("why did the customer ask for a refund")
    assert plan.explicit_intent is None
    assert plan.intent_match is None


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
