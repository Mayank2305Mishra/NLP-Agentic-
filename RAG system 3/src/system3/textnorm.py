from __future__ import annotations

import re
import unicodedata

_CONTRACTIONS = {
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "what's": "what is",
    "who's": "who is",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "wasn't": "was not",
    "aren't": "are not",
    "weren't": "were not",
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "i've": "i have",
    "we've": "we have",
    "you've": "you have",
    "i'll": "i will",
    "we'll": "we will",
    "you'll": "you will",
    "i'd": "i would",
    "we'd": "we would",
    "you'd": "you would",
    "let's": "let us",
    "kinda": "kind of",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
}

_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_CONTRACTIONS, key=len, reverse=True)) + r")\b"
)
_WS_RE = re.compile(r"\s+")
_KEEP_RE = re.compile(r"[^a-z0-9\s'’-]")

CUSTOMER_MARKERS = ("customer", "caller", "client", "caller-side", "policyholder")
AGENT_MARKERS = (
    "agent",
    "representative",
    "rep",
    "advisor",
    "adviser",
    "specialist",
    "supervisor",
    "manager",
    "consultant",
    "operator",
    "associate",
    "agent-",
)
NOTE_MARKERS = (
    "note",
    "system",
    "summary",
    "key event",
    "tracker",
    "disposition",
    "outcome",
    "survey",
    "domain category",
    "call notes",
    "after a moment",
    "asr noise",
    "please advise",
    "patterns noted",
    "additional note",
    "wrap-up",
    "wrap up",
    "end of call",
    "escalation",
    "student",
    "teacher",
    "listener",
    "ai",
)

_NOTE_TEXT_RE = re.compile(
    r"^\s*([\[*#>-]{1,3}\s*)?("
    r"key events|call summary|summary of|internal note|qa note|churn note|post-call|"
    r"agent notes?|disposition|call notes|wrap[- ]?up notes?|outcome[:\-]|notes?:"
    r")",
    re.IGNORECASE,
)

_BYE_RE = re.compile(
    r"\b(goodbye|good bye|bye bye|bye|have a (good|great|nice) (day|evening)|"
    r"thanks for calling|thank you for calling|talk to you (later|soon))\b",
    re.IGNORECASE,
)

_NAMEISH_RE = re.compile(r"^[A-Z][a-zA-Z'’.-]+(?: [A-Z][a-zA-Z'’.-]+){0,2}$")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("​", "").replace("\t", " ")
    text = text.lower()
    text = _CONTRACTION_RE.sub(lambda m: _CONTRACTIONS[m.group(1)], text)
    text = _KEEP_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_speaker(speaker_original: str) -> str:
    raw = (speaker_original or "").strip()
    low = raw.lower().strip(" -*#[].()")
    if not low:
        return "unknown"
    if any(marker in low for marker in NOTE_MARKERS):
        return "unknown"
    if any(marker in low for marker in CUSTOMER_MARKERS):
        return "customer"
    if any(marker in low for marker in AGENT_MARKERS):
        return "agent"
    return "unknown"


def infer_speaker_by_position(
    speaker_original: str, previous_canonical: str | None
) -> tuple[str, bool]:
    canonical = normalize_speaker(speaker_original)
    if canonical != "unknown":
        return canonical, False
    raw = (speaker_original or "").strip()
    if not raw or not _NAMEISH_RE.match(raw):
        return "unknown", False
    if previous_canonical == "customer":
        return "agent", True
    if previous_canonical == "agent":
        return "customer", True
    return "unknown", False


def looks_like_note_speaker(speaker_original: str) -> bool:
    low = (speaker_original or "").strip().lower()
    if not low:
        return False
    if any(marker in low for marker in NOTE_MARKERS):
        return True
    if low.startswith(("[", "(*", "*", "#")):
        return True
    if low.startswith("- ") and " " in low[2:]:
        return True
    return False


def is_post_call_artifact(
    speaker_original: str,
    text_original: str,
    previous_text: str | None = None,
) -> bool:
    if looks_like_note_speaker(speaker_original):
        return True
    text = (text_original or "").strip()
    if not text:
        return False
    if _NOTE_TEXT_RE.match(text):
        return True
    if previous_text and _BYE_RE.search(previous_text) and _NOTE_TEXT_RE.match(text):
        return True
    return False


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4))
