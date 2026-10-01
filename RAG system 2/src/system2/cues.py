from __future__ import annotations

CAUSE_CUES = [
    "because",
    "reason",
    "cause",
    "caused by",
    "due to",
    "result of",
    "why",
    "what led to",
    "trigger",
    "stemming from",
    "since",
    "owing to",
    "as a result",
    "root cause",
]

EFFECT_CUES = [
    "therefore",
    "so",
    "resulted",
    "led to",
    "resolved",
    "fixed",
    "confirmed",
    "decided",
    "agreed",
    "accepted",
    "cancelled",
    "canceled",
    "refunded",
    "thereby",
    "outcome",
    "in the end",
    "eventually",
]

ACTION_CUES = [
    "i can",
    "i will",
    "let me",
    "i'll",
    "we can",
    "i've updated",
    "i've processed",
    "i can help",
    "i'll arrange",
    "i will process",
    "i'll process",
    "i can offer",
    "i will escalate",
    "let me check",
    "i can waive",
    "i've applied",
]

PROBLEM_CUES = [
    "problem",
    "issue",
    "complaint",
    "unhappy",
    "dissatisfied",
    "frustrated",
    "angry",
    "not happy",
    "doesn't work",
    "not working",
    "wrong",
    "error",
    "failed",
    "failure",
    "charged",
    "overcharged",
    "confused",
    "concerned",
    "worried",
    "disappointed",
]

TEMPORAL_CUES = ["after", "before", "then", "when", "once", "since", "while", "until"]

SYNONYM_MAP: dict[str, list[str]] = {
    "refund": ["refund", "reimbursement", "money back", "credit back", "return"],
    "cancel": ["cancellation", "cancel", "canceled", "cancelled", "terminate"],
    "cancellation": ["cancel", "cancellation", "terminated"],
    "booking": ["booking", "reservation", "itinerary"],
    "reservation": ["reservation", "booking"],
    "discount": ["discount", "promo", "promotion", "coupon", "offer"],
    "promo": ["promotion", "promo code", "coupon", "discount"],
    "price": ["price", "pricing", "cost", "charge", "rate"],
    "pricing": ["price", "cost", "charge", "rate"],
    "complaint": ["complaint", "dissatisfied", "unhappy", "frustrated"],
    "defect": ["defect", "broken", "damaged", "not working", "malfunction"],
    "upgrade": ["upgrade", "upsell", "upselling", "plan change"],
    "plan": ["plan", "package", "tier", "coverage"],
    "coverage": ["coverage", "covered", "policy"],
    "claim": ["claim", "reimbursement", "approval"],
    "delay": ["delay", "delayed", "late", "postponed"],
    "error": ["error", "mistake", "incorrect", "wrong"],
    "issue": ["issue", "problem", "trouble"],
    "resolve": ["resolve", "resolved", "fix", "fixed", "solution"],
    "resolution": ["resolved", "resolution", "fixed", "solved"],
    "cause": ["cause", "reason", "why", "trigger", "led to", "resulted in"],
    "because": ["because", "due to", "since", "reason"],
    "unhappy": ["unhappy", "dissatisfied", "frustrated", "complaint"],
    "agent": ["agent", "representative", "advisor", "specialist"],
    "compensation": ["compensation", "goodwill", "credit", "voucher"],
    "fee": ["fee", "charge", "penalty", "fine"],
    "duplicate": ["duplicate", "double", "twice", "two times"],
    "overcharge": ["overcharge", "charged twice", "extra charge", "wrong amount"],
    "policy": ["policy", "coverage", "plan", "terms"],
    "renewal": ["renewal", "renew", "extend"],
    "compare": ["comparison", "compare", "versus", "vs", "difference"],
    "comparison": ["compare", "versus", "vs", "difference", "alternative"],
}

CAUSE_EXPANSION = ["cause", "reason", "why", "trigger", "led to", "resulted in", "due to"]
EFFECT_EXPANSION = ["effect", "result", "outcome", "consequence", "resolved", "led to"]

STOPWORDS_FOR_CUES = frozenset(
    """a an and are as at be been but by for from has have how i if in into is it its
    me my of on or our so that the their then there these they this to was we were what
    when where which who why will with you your""".split()
)

CONTENT_STOPWORDS = STOPWORDS_FOR_CUES | frozenset(
    """uh um uh-huh uh huh mm-hmm okay ok right yeah yes no well actually just like
    really know let s t don t""".split()
)


def cue_hits(text: str, cues: list[str]) -> list[str]:
    low = text or ""
    hits = []
    for cue in cues:
        if " " in cue:
            if cue in low:
                hits.append(cue)
        else:
            if f" {cue} " in f" {low} ":
                hits.append(cue)
    return hits


def content_words(text: str) -> list[str]:
    return [w for w in (text or "").split() if w not in CONTENT_STOPWORDS and len(w) > 1]
