"""Parse + validate generator output: JSON -> Pydantic -> evidence-ID grounding.

Pipeline (AGENTS.md 17-18): HTTP (client) -> JSON parse -> schema -> required
fields -> evidence-ID check -> quote grounding -> optional Laya answer gate.
One bounded repair retry; never loop indefinitely.
"""
from __future__ import annotations

import json
import re
import unicodedata

from app.domain.errors import GenerationParseError, GroundingValidationError
from app.domain.models import EvidencePack, Task1Result, Task2Result


def _insert_missing_commas(s: str) -> str:
    """Structural repair for the most common LLM JSON slip: omitted commas.

    Strict-first: this only ever runs after json.loads already failed, and the
    result must still parse and pass the Pydantic schema to be accepted.
    """
    out: list[str] = []
    stack: list[str] = []          # "obj" | "arr"
    pending_value = False          # a complete value was just emitted
    i, n = 0, len(s)
    literal = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null")
    while i < n:
        ch = s[i]
        if ch.isspace():
            out.append(ch)
            i += 1
            continue
        # A complete value must be followed by ',' (or the closer); if another
        # token starts instead, the comma was omitted by the model.
        if pending_value and (ch == '"' or ch in "{[" or literal.match(s, i)):
            out.append(",")
            pending_value = False
        if ch == '"':
            j, esc = i + 1, False
            while j < n:
                c = s[j]
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    break
                j += 1
            token = s[i:min(j + 1, n)]
            out.append(token)
            i = j + 1
            k = i
            while k < n and s[k].isspace():
                k += 1
            # A string followed by ':' inside an object is a key, not a value.
            pending_value = not (k < n and s[k] == ":" and (not stack or stack[-1] == "obj"))
            continue
        if ch in "{[":
            out.append(ch)
            stack.append("obj" if ch == "{" else "arr")
            pending_value = False
            i += 1
            continue
        if ch in "}]":
            out.append(ch)
            if stack:
                stack.pop()
            pending_value = True   # the container is now a value in its parent
            i += 1
            continue
        if ch in ":,":
            out.append(ch)
            pending_value = False
            i += 1
            continue
        m = literal.match(s, i)
        if m:
            out.append(m.group(0))
            pending_value = True
            i = m.end()
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _repair_json(s: str) -> str:
    fixed = re.sub(r",\s*([}\]])", r"\1", s)          # trailing commas
    if fixed != s:
        try:
            json.loads(fixed)
            return fixed
        except json.JSONDecodeError:
            pass
    return _insert_missing_commas(s)


def extract_json_object(text: str) -> dict:
    """Tolerate code fences / surrounding prose / minor JSON slippage.

    Never cut objects halfway: every candidate is a complete {...} span and
    must parse cleanly (strict attempt first, bounded repair second).
    """
    t = text.strip()
    candidates: list[str] = []
    for fence in ("```json", "```"):
        if fence in t:
            try:
                inner = t.split(fence, 1)[1]
                inner = inner.rsplit("```", 1)[0] if "```" in inner else inner
                candidates.append(inner.strip())
                break
            except IndexError:
                pass
    candidates.append(t)
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        candidates.append(t[start:end + 1])
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_repair_json(cand))
        except json.JSONDecodeError:
            pass
    raise GenerationParseError("Generator output is not valid JSON",
                               context={"preview": t[:300]})



def parse_task1(text: str) -> Task1Result:
    data = extract_json_object(text)
    try:
        return Task1Result.model_validate(data)
    except Exception as e:
        raise GenerationParseError(f"Task1 schema validation failed: {e}",
                                   context={"preview": text[:300]}) from e


def parse_task2(text: str) -> Task2Result:
    data = extract_json_object(text)
    try:
        return Task2Result.model_validate(data)
    except Exception as e:
        raise GenerationParseError(f"Task2 schema validation failed: {e}",
                                   context={"preview": text[:300]}) from e


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


# The corpus uses typographic punctuation (“ ” ‘ ’ – —); models copy quotes
# with ASCII equivalents. Compare canonically — corpus text is never mutated.
_PUNCT = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u02bc": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u00a0": " ", "\u2026": "...",
})


def _canon(s: str) -> str:
    return _norm_ws(unicodedata.normalize("NFC", s).translate(_PUNCT))


def _quote_grounded(quote: str, source: str, threshold: float = 0.85) -> bool:
    """Tolerant grounding: exact substring -> verbatim prefix -> aligned fuzzy.

    LLMs routinely normalise punctuation, truncate ("..."), or cut mid-word when
    copying evidence. Strict substring matching turns those into hard failures.
    Contiguity is still required: the quote must match ONE window of the source,
    so stitched quotes joining distant turns stay rejected.
    """
    import difflib

    q = _canon(quote)
    src = _canon(source)
    if not q:
        return True
    if q in src:
        return True
    # Truncation: long verbatim prefix counts (model cut off mid-quote).
    if len(q) >= 40 and q[:40] in src:
        return True
    if len(q) < 24:
        return False
    # Align on the longest common block, then score the source window that
    # starts where the quote starts (offset-accurate; no fixed-step guessing).
    sm = difflib.SequenceMatcher(None, q, src, autojunk=False)
    m = sm.find_longest_match(0, len(q), 0, len(src))
    if m.size <= 0:
        return False
    base = m.b - m.a
    for delta in (0, -1, 1, -2, 2):
        lo = max(0, base + delta)
        hi = min(len(src), lo + len(q))
        if hi - lo < max(24, int(len(q) * 0.5)):
            continue
        window = src[lo:hi]
        if difflib.SequenceMatcher(None, q, window, autojunk=False).ratio() >= threshold:
            return True
    return False


def validate_task1_grounding(result: Task1Result, pack: EvidencePack) -> dict:
    """Deterministic checks: IDs exist, quotes present (ws-normalized), fields set."""
    valid_ids = {c.chunk_id for c in pack.chunks}
    by_id = {c.chunk_id: c for c in pack.chunks}
    cited = 0
    valid = 0
    if result.insufficient_evidence:
        return {"cited": 0, "valid_ids": 0, "grounded": True, "insufficient": True}
    for r in result.causal_rationales:
        if not r.cause.strip() or not r.effect.strip():
            raise GroundingValidationError("Rationale has empty cause/effect",
                                           context={"cause": r.cause[:80]})
        if not r.evidence:
            raise GroundingValidationError("Rationale has no evidence citations",
                                           context={"cause": r.cause[:80]})
        for ev in r.evidence:
            cited += 1
            if ev.source_id not in valid_ids:
                raise GroundingValidationError(
                    f"Hallucinated citation ID: {ev.source_id}",
                    context={"source_id": ev.source_id, "valid": sorted(valid_ids)[:5]})
            valid += 1
            if ev.quote.strip():
                if "..." in ev.quote or "…" in ev.quote:
                    raise GroundingValidationError(
                        "Evidence quote must be verbatim: split '...' into separate evidence entries",
                        context={"source_id": ev.source_id, "quote": ev.quote[:120]})
                src = by_id[ev.source_id].text
                if not _quote_grounded(ev.quote, src):
                    raise GroundingValidationError(
                        f"Evidence quote not found in source {ev.source_id}",
                        context={"source_id": ev.source_id, "quote": ev.quote[:120]})
    return {"cited": cited, "valid_ids": valid, "grounded": True, "insufficient": False}


def validate_task2_grounding(result: Task2Result, pack: EvidencePack) -> dict:
    valid_ids = {c.chunk_id for c in pack.chunks}
    cited = len(result.supporting_evidence)
    valid = sum(1 for sid in result.supporting_evidence if sid in valid_ids)
    if cited and valid < cited:
        bad = [s for s in result.supporting_evidence if s not in valid_ids]
        raise GroundingValidationError(f"Task2 cites unknown evidence IDs: {bad[:3]}",
                                       context={"bad_ids": bad[:3]})
    if not result.synthesis.strip():
        raise GroundingValidationError("Task2 synthesis is empty")
    return {"cited": cited, "valid_ids": valid, "grounded": True}
