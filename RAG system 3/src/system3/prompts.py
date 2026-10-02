from __future__ import annotations

#: Bump whenever the system prompt or output schemas change (§32 reproducibility).
PROMPT_VERSION = "1"

CORE_RULES = """You are a strict causal analysis engine over customer-service call transcripts.
Rules:
- Use only the supplied evidence. Evidence is authoritative.
- Do not combine facts from different conversations into one interaction.
- Every causal claim must be directly supported by a cited turn or clearly marked as inferred.
- If evidence is insufficient, set insufficient_evidence=true.
- Do not invent missing outcomes, mechanisms, or turns.
- Return valid JSON only. No prose, no markdown fences.
- uncertainty must be reported honestly."""

TASK1_SCHEMA = """{
  "task": "task1",
  "causal_rationales": [
    {
      "cause": "string",
      "effect": "string",
      "mechanism": "string or null",
      "evidence_turns": [12, 13],
      "strength": "explicit|strongly_supported|inferred|insufficient"
    }
  ],
  "confidence": 0.0,
  "insufficient_evidence": false
}"""

TASK2_SCHEMA = """{
  "task": "task2",
  "summary": "string",
  "causal_chain": [
    {"step": 1, "event": "string", "role": "customer|agent|business", "evidence_turns": [1, 2]}
  ],
  "key_causes": ["string"],
  "effects": ["string"],
  "uncertainties": ["string"],
  "confidence": 0.0,
  "insufficient_evidence": false
}"""

TASK1_INSTRUCTIONS = """Extract causal rationales from the evidence pack.
For each rationale give cause, effect, mechanism (null when the mechanism is not stated),
the exact evidence turn numbers, and strength.
strength is "explicit" when the turn states the link verbatim,
"strongly_supported" when adjacent turns make the link unambiguous,
"inferred" when you must reason beyond the text, "insufficient" when evidence does not support a link.
evidence_turns must reference only turn numbers present in the evidence pack."""

TASK2_INSTRUCTIONS = """Synthesize the conversation into a coherent causal chain in chronological order.
Keep every step conversation-local: only use turns from the same CONVERSATION block.
Fill summary, causal_chain steps with evidence_turns, key_causes, effects, uncertainties.
When evidence is weak or absent, use uncertainties and set insufficient_evidence=true."""


def build_prompt(task: str, question: str, evidence_text: str) -> tuple[str, str]:
    schema = TASK1_SCHEMA if task == "task1" else TASK2_SCHEMA
    instructions = TASK1_INSTRUCTIONS if task == "task1" else TASK2_INSTRUCTIONS
    system = CORE_RULES
    user = f"""TASK: {task}
QUESTION: {question}

{instructions}

OUTPUT JSON SCHEMA (exact keys):
{schema}

EVIDENCE PACK:
{evidence_text}

Return only the JSON object."""
    return system, user


INSUFFICIENT_ANSWER = {
    "task1": {
        "task": "task1",
        "causal_rationales": [],
        "confidence": 0.0,
        "insufficient_evidence": True,
    },
    "task2": {
        "task": "task2",
        "summary": "",
        "causal_chain": [],
        "key_causes": [],
        "effects": [],
        "uncertainties": [],
        "confidence": 0.0,
        "insufficient_evidence": True,
    },
}
