# AGENTS.md — System 4: Agentic / Quality RAG

> **Document control**
> Status: Production · Version: 2.0 · Supersedes: 1.x (same section numbering, §0–§52 preserved)
> Scope: architecture, contracts, operational rules, and acceptance criteria for System 4
> Change rule: any material deviation from §52 must be documented in Appendix A with its comparability impact
> Review cadence: re-benchmark thresholds (§12, §19, §35) at every benchmark cycle; never tune only on the final evaluation set

---

## 0. Purpose

This repository implements System 4 of the conversational causal-rationale benchmark.

System 4 is the highest-quality, multi-stage RAG architecture. Its purpose is to maximize evidence quality and causal correctness by performing broad candidate retrieval followed by reranking, evidence selection, context compression, Laya/System-1 decision control, LLM synthesis, and grounding validation.

The architecture must remain measurably distinct from Systems 1–3.

System 4 is NOT merely System 1 with a larger top-k. It is a deliberate multi-stage retrieval-and-evidence pipeline.

**Operating principle**

    evidence quality > recall after sufficient candidate generation > context efficiency > latency

---

## 1. System Lineup

The benchmark contains four systems:

1. **System 1 — Dense Vector RAG**
   - Embeddings
   - Vector search / FAISS
   - Conversation-aware retrieval and expansion

2. **System 2 — Hybrid RAG**
   - Dense semantic retrieval
   - Lexical/BM25 retrieval
   - Candidate/rank fusion

3. **System 3 — Vectorless RAG**
   - No embeddings
   - No vector database
   - BM25 / inverted index / lexical and metadata signals

4. **System 4 — Agentic / Quality RAG**
   - Broad candidate retrieval
   - Multi-signal retrieval
   - Learned or cross-encoder reranking where available
   - Conversation and turn selection
   - Causal evidence scoring
   - Context compression
   - Laya-controlled decisions
   - LLM synthesis
   - Grounding and answer validation

The fourth system is expected to have higher retrieval complexity and potentially higher latency. That trade-off must be measured rather than assumed.

---

## 2. Core Design Principle

System 4 optimizes for:

    evidence quality > recall after sufficient candidate generation > context efficiency > latency

subject to a hard requirement that every generated rationale is grounded in retrieved evidence.

The system must prefer abstention over unsupported causal claims.

The system must never combine unrelated conversations merely because their text is semantically similar.

Every evidence item must retain its `transcript_id`, turn range, retrieval stage, score, and selection reason.

---

## 3. Critical Lessons Carried Forward From System 1

The following observed failure modes are architectural requirements for System 4.

### 3.1 Never retrieve arbitrary global chunks for causal tasks

A previous System 1 run retrieved eight chunks from unrelated conversations. The resulting Task 2 answer synthesized facts about multiple customers in one answer.

System 4 must therefore enforce:

    query
      -> candidate conversations
      -> conversation-aware evidence selection
      -> final evidence pack

A final evidence pack must not contain unrelated conversations unless the user explicitly asks for cross-conversation analysis.

### 3.2 Similarity score is not evidence quality

Scores around 0.39–0.42 were previously sufficient to retrieve unrelated insurance/refund conversations.

System 4 must not interpret raw retrieval similarity as causal confidence.

Use independent signals:

- conversation relevance
- query term coverage
- entity consistency
- turn proximity
- causal cue presence
- action/outcome presence
- reranker score
- evidence completeness
- contradiction signals
- metadata consistency

### 3.3 Generation latency must not hide retrieval latency

Previous traces showed retrieval in milliseconds but generation taking tens or hundreds of seconds.

Every stage must be timed independently:

- normalization
- candidate retrieval
- reranking
- evidence selection
- Laya decisions
- compression
- prompt construction
- LLM time-to-first-token
- LLM generation
- JSON parsing
- grounding validation
- total request time

Do not report only total latency.

### 3.4 Context must be compact

Previous runs sent roughly 4,000 tokens to the LLM.

System 4 should normally target:

- Task 1: 800–1,500 evidence tokens
- Task 2: 1,500–2,500 evidence tokens

These are initial operating targets, not immutable limits. Benchmark them and tune empirically.

---

## 4. System 4 High-Level Architecture

```text
User Query
    |
    v
Query Normalization
    |
    v
Task / Intent Routing
    |
    +-----------------------------+
    |                             |
    v                             v
Broad Dense Retrieval       Broad Lexical Retrieval
    |                             |
    +-------------+---------------+
                  |
                  v
        Candidate Conversation Pool
                  |
                  v
        Candidate Deduplication
                  |
                  v
        Conversation Consistency Gate
                  |
                  v
             Reranker
                  |
                  v
        Relevant Turn Selection
                  |
                  v
        Neighbor / Causal Expansion
                  |
                  v
       Causal Evidence Scoring
                  |
                  v
           Laya Decision Layer
          /        |          \
      reject     select       expand
                  |
                  v
        Context Compression
                  |
                  v
          Evidence Pack
                  |
                  v
          LLM Generation
                  |
                  v
       Structured Output Parser
                  |
                  v
       Grounding Validator
                  |
                  v
          Final Answer Gate
                  |
             +----+----+
             |         |
           answer    abstain
```

---

## 5. Repository Layout

Canonical structure (the implementation may add modules, but every module below must exist with the stated responsibility):

```text
system4/
├── AGENTS_SYSTEM_4.md
├── final_transcripts.json
├── README.md
├── pyproject.toml
├── .env.example
│
├── app/
│   └── streamlit_app.py
│
├── config/
│   ├── settings.py          # every threshold, fingerprinted for cache keys
│   └── prompts.py           # prompt contracts, injection defense, prompt version
│
├── ingestion/
│   ├── loader.py            # source reading + skip statistics
│   ├── normalize.py         # normalization, content hash, post-call detection
│   └── artifacts.py         # versioned persistence + validation
│
├── retrieval/
│   ├── index.py             # deterministic inverted index + hashed vectors
│   ├── dense.py
│   ├── lexical.py
│   ├── broad_retrieval.py   # RRF fusion, per-stream observability, fallbacks
│   └── metadata.py          # conversation consistency gate (§10)
│
├── reranking/
│   ├── features.py          # feature extraction + normalization
│   ├── reranker.py          # protocol + feature/passthrough + factory
│   └── cross_encoder.py     # optional learned reranker (lazy import)
│
├── evidence/
│   ├── causal_scoring.py    # typed causal roles (§15)
│   ├── selection.py         # turn selection + neighbor expansion
│   ├── compressor.py        # NONE/LIGHT/AGGRESSIVE compression
│   ├── packer.py            # evidence pack format + token budgets
│   └── gate.py              # structured evidence features for Laya
│
├── decision/
│   ├── schemas.py           # typed decision contracts (§18)
│   ├── policies.py          # deterministic policy + task routing + fallback
│   ├── laya.py              # Laya facade
│   └── jev.py               # Jev-compatible adapter, local-first (§20)
│
├── llm/
│   ├── base.py              # provider-independent protocol + measured generation
│   ├── structured.py        # tolerant JSON extraction, never fabricates
│   ├── ollama.py            # local, streaming TTFT + token usage
│   ├── bonsai.py            # local endpoint adapter
│   ├── cloud.py             # approved remote adapter (explicit opt-in)
│   └── __init__.py          # local-first factory
│
├── validation/
│   ├── grounding.py         # claim support, contradiction detection (§29)
│   ├── citation.py          # transcript-scoped citation checks
│   ├── consistency.py       # anti-contamination identity checks (§23)
│   ├── causal.py            # causal shape guardrails
│   └── answer_gate.py       # final gate with pruning/downgrading (§30)
│
├── evaluation/
│   ├── retrieval.py         # Recall@k, Precision@k, MRR, nDCG, contamination
│   ├── quality.py           # grounding, abstention precision, completeness
│   ├── latency.py           # per-stage mean/p50/p95 aggregation
│   ├── benchmark.py         # orchestration + variant matrix
│   └── reports.py           # JSON + Markdown reports
│
├── pipeline.py              # canonical stage wiring, timings, fallbacks
├── cache.py                 # versioned cache keys (§39)
├── observability.py         # structured traces with redaction (§40)
│
├── tests/                   # unit, regression, end-to-end (§47)
│
├── benchmarks/
│   └── questions.json       # fixed seeded question set (§35)
│
└── scripts/
    ├── build_index.py
    ├── build_questions.py
    ├── run_benchmark.py
    └── inspect_trace.py
```

---

## 6. Data Contract

The source dataset contains conversation records with fields such as:

- `transcript_id`
- `time_of_interaction`
- `domain`
- `intent`
- `reason_for_call`
- `conversation`

Conversation turns may contain:

- speaker
- text
- turn number or equivalent ordering

Do not assume speaker names are always literally `Agent` and `Customer`. Preserve the original speaker identity and normalize only when the schema makes the mapping safe.

The source data may contain conversational noise, incomplete punctuation, hesitations, malformed sentences, and metadata nulls.

Never fabricate missing metadata.

---

## 7. Ingestion

Normalize records once and persist a deterministic representation.

Required normalized fields:

```python
{
    "transcript_id": str,
    "domain": str | None,
    "intent": str | None,
    "reason_for_call": str | None,
    "turn_id": int,
    "speaker": str,
    "text": str,
    "post_call": bool,
}
```

Preserve:

- original turn number (when present and sane)
- original text (whitespace collapsed only; meaning unchanged)
- original speaker
- source transcript ID

Never modify source data in place.

Create a content hash for reproducibility. Malformed records are skipped and counted — never repaired by invention.

---

## 8. Broad Retrieval

System 4 begins with high-recall retrieval.

The purpose of broad retrieval is candidate generation, not final evidence selection.

Use both:

1. dense retrieval
2. lexical retrieval

The two candidate streams should remain independently observable in traces.

Recommended initial candidate sizes:

```text
Dense candidates:       20–50 conversations
Lexical candidates:     20–50 conversations
Union:                  30–80 conversations
```

Tune these values empirically.

Do not send this broad candidate set to the LLM.

---

## 9. Candidate Pool Construction

Merge candidates by `transcript_id`.

For each conversation store:

```python
{
    "transcript_id": ...,
    "dense_score": ...,
    "lexical_score": ...,
    "metadata_score": ...,
    "source_retrievers": [...],
}
```

Do not average scores from incompatible retrieval systems without normalization.

Prefer rank-based fusion or calibrated score normalization.

Every candidate must remain traceable to its original retrieval source.

---

## 10. Conversation Consistency Gate

Before reranking turns, determine whether a candidate conversation is internally consistent with the query.

Signals include:

- domain compatibility
- intent compatibility
- reason-for-call compatibility
- named entity overlap
- customer/product/service overlap
- query topic coverage

Metadata is supporting evidence, not a replacement for transcript evidence.

A metadata-only match must never become final causal evidence.

The gate must keep a configurable floor of candidates so that sparse metadata can never empty the pool before transcript evidence is scored.

---

## 11. Reranking

Reranking is the defining quality stage of System 4.

The reranker should score query/evidence relevance more precisely than the initial retrievers.

Possible implementations:

1. cross-encoder reranker
2. lightweight local reranker
3. hosted reranker when permitted
4. deterministic feature-based reranker as fallback

The implementation must be swappable behind one protocol and must degrade gracefully: unavailable learned rerankers fall back to the deterministic feature reranker.

Recommended initial flow:

```text
80 candidates
    ↓
conversation consistency
    ↓
30 candidates
    ↓
reranker
    ↓
5–10 conversations
```

Do not assume these values are optimal. Benchmark them.

---

## 12. Reranker Features

If a learned reranker is unavailable, use deterministic features.

Example feature family:

```text
query-term coverage
entity overlap
phrase overlap
semantic relevance
metadata compatibility
reason-for-call similarity
causal keyword proximity
customer problem proximity
agent action proximity
outcome proximity
turn density
```

Potential weighted score:

```text
rerank_score =
    w1 * semantic_relevance
  + w2 * lexical_relevance
  + w3 * entity_match
  + w4 * metadata_match
  + w5 * causal_density
  + w6 * action_density
  + w7 * outcome_density
```

Weights must be configurable. Normalize each feature across the candidate set before weighting; never feed raw cross-system scores into one sum.

Do not claim these weights are optimal without benchmark evidence.

---

## 13. Relevant Turn Selection

Once the best conversations are identified, select the turns that directly support the query.

Do not simply include the entire conversation.

For each selected turn, retain:

- transcript ID
- turn number
- speaker
- exact text
- selection score
- selection reason

Selection reasons may include:

- query match
- cause evidence
- effect evidence
- agent action
- customer motivation
- outcome
- contradiction
- clarification

---

## 14. Causal Evidence Expansion

Causal evidence often spans adjacent turns.

If a relevant turn is found:

```text
T20
```

inspect a local window such as:

```text
T18–T22
```

or adjacent chunks.

The expansion window must be configurable. Do not blindly expand entire conversations. Prefer local causal neighborhoods. Never expand into post-call artifacts (§24).

---

## 15. Causal Evidence Model

System 4 should explicitly distinguish:

```text
CAUSE
TRIGGER
CUSTOMER MOTIVATION
PROBLEM
AGENT ACTION
MECHANISM
EFFECT
OUTCOME
UNCERTAINTY
```

Evidence can support one or more roles.

Example:

```json
{
  "turn": 24,
  "role": "customer_motivation",
  "reason": "Customer explicitly asks why the refund option is popular"
}
```

Do not infer a mechanism merely because cause and effect appear in adjacent turns. A mechanism role requires an explicit explanatory marker in an agent utterance.

---

## 16. Evidence Completeness

For Task 1, prefer evidence containing:

```text
trigger/problem
    +
customer response/motivation
    +
agent response or mechanism
    +
outcome when available
```

For Task 2, prefer a complete interaction-level causal chain.

If only part of the chain is supported, represent the missing part as uncertainty rather than hallucinating it.

---

## 17. Laya Integration

Laya is the fast System-1 decision/control layer around the expensive retrieval and generation stages.

Laya must NOT replace retrieval.

Laya must NOT silently invent evidence.

Laya should make typed, deterministic decisions such as:

- task route
- candidate acceptance
- evidence sufficiency
- expansion decision
- compression policy
- answer gate

Conceptually:

```text
retrieval candidates
        ↓
      Laya
   /    |    \
 keep  expand reject
        |
        v
 evidence pipeline
```

Use Laya where a low-latency decision can prevent unnecessary expensive work. Laya may run a second pass after expansion to re-decide sufficiency.

---

## 18. Laya Decision Contracts

Recommended decisions:

### `task_route`

```text
TASK_1
TASK_2
UNKNOWN
```

### `evidence_action`

```text
ACCEPT
EXPAND
RERANK_MORE
REJECT
```

### `compression_level`

```text
NONE
LIGHT
AGGRESSIVE
```

### `answer_gate`

```text
ANSWER
ANSWER_WITH_UNCERTAINTY
ABSTAIN
```

All Laya outputs must be typed and logged.

---

## 19. Laya Evidence Policy

Laya can consume structured evidence features such as:

```python
{
    "top_rerank_score": float,
    "conversation_count": int,
    "entity_consistency": float,
    "causal_coverage": float,
    "outcome_coverage": float,
    "contradiction_count": int,
    "evidence_tokens": int,
}
```

Example policy:

```text
if contradiction_count > 0:
    inspect_or_expand

if causal_coverage is low:
    expand

if evidence_tokens are excessive:
    compress

if evidence is incomplete:
    abstain or answer with uncertainty
```

Exact thresholds must be benchmark-calibrated.

---

## 20. Jev Compatibility

Where the installed Laya/Jev integration supports a System-One decision schema, expose the decision layer through the compatible typed interface.

Keep the application logic independent from the protocol layer.

Recommended abstraction:

```text
Application
   ↓
DecisionPolicy
   ↓
LayaAdapter
   ↓
Jev-compatible interface when enabled
```

Do not make the whole application depend directly on a remote Jev service. Local-first execution is preferred for confidential hackathon transcripts; any remote decision endpoint must fail open to the local deterministic policy, never to silence.

---

## 21. Context Compression

Compression is performed AFTER reranking and evidence selection.

Never compress before identifying the relevant conversation and causal turns.

Compression should:

- remove irrelevant greetings
- remove repetitive filler
- remove unrelated product discussion
- preserve causal transitions
- preserve exact evidence turns
- preserve uncertainty
- preserve speaker identity
- preserve source locations

Compression must not rewrite evidence in a way that changes its meaning.

When possible, store both:

1. original evidence
2. compressed representation

for auditing. Dropped turn IDs must be recorded for audit.

---

## 22. Evidence Pack

The final evidence pack should be compact and explicit.

Recommended format:

```text
SOURCE: transcript_id
DOMAIN: ...
INTENT: ...

[Turn 012 | Customer]
...

[Turn 013 | Agent]
...

[Turn 014 | Customer]
...

Evidence roles:
- customer_problem
- agent_action
- outcome
```

Every source must have a stable identifier. Turns inside a source stay chronological; sources stay grouped (a single-interaction answer contains exactly one SOURCE block).

---

## 23. Hard Anti-Contamination Rule

A single answer must not merge customers from unrelated conversations.

Before LLM generation:

```python
unique_transcript_ids = set(evidence.transcript_id)
```

If the task is single-interaction analysis:

```text
len(unique_transcript_ids) > 1
```

must trigger a validation step.

Default behavior:

- keep the highest-quality conversation
- discard unrelated candidates
- or abstain if ambiguity remains

Only allow multiple transcript IDs when the user explicitly requests comparative or aggregate analysis. In that mode every citation must be transcript-scoped (`TRANSCRIPT_ID:TURN`).

---

## 24. Post-Call Artifact Handling

Some conversations may contain post-call notes, internal QA text, or artifacts after the apparent conversational ending.

These should not automatically become customer causal evidence.

Detect likely post-call segments using:

- structural markers
- speaker patterns
- sudden style changes
- internal-note phrases
- conversation termination markers

Classify them separately (`post_call: true`).

Use them only when the task explicitly requests internal notes or post-call analysis.

---

## 25. LLM Abstraction

The generator must be provider-independent.

Required interface:

```python
class LLM:
    def generate(self, prompt: str, **kwargs) -> str:
        ...
```

Supported adapters may include:

```text
Bonsai local endpoint
Ollama / Qwen3.5 9B MLX
Groq
Gemini
Other approved providers
```

Adapters should optionally expose measured generation (TTFT, token usage) for §32 instrumentation.

System 4 benchmarks must keep the LLM fixed when comparing retrieval architectures.

Do not change the LLM and retrieval architecture simultaneously in the main experiment.

---

## 26. LLM Generation Settings

For deterministic evaluation:

```text
temperature = 0
structured JSON output = enabled when supported
reasonable context limit
explicit grounding instructions
```

Avoid excessive context merely because the model supports a large context window.

---

## 27. Task 1 Prompt Contract

Task 1 extracts causal rationales.

Required conceptual output:

```json
{
  "task": "task1",
  "causal_rationales": [
    {
      "cause": "...",
      "effect": "...",
      "mechanism": "...",
      "evidence_turns": [1, 2],
      "strength": "explicit|inferred"
    }
  ],
  "confidence": 0.0,
  "insufficient_evidence": false
}
```

The model must not invent a mechanism when the evidence only supports cause and effect.

---

## 28. Task 2 Prompt Contract

Task 2 synthesizes the interaction-level causal rationale.

Required conceptual output:

```json
{
  "task": "task2",
  "summary": "...",
  "key_causes": [],
  "effects": [],
  "mechanisms": [],
  "uncertainties": [],
  "evidence": []
}
```

The schema can evolve, but downstream validators must remain explicit. Abstention responses must retain the full schema shape.

---

## 29. Grounding Validator

After generation, validate every causal claim.

Validation should check:

1. claim exists in output
2. claim has supporting evidence
3. referenced turn exists
4. referenced turn belongs to the retrieved conversation
5. claim does not exceed evidence
6. unsupported certainty is downgraded

Potential statuses:

```text
GROUNDED
PARTIALLY_GROUNDED
UNSUPPORTED
CONTRADICTED
```

Citation resolution must be transcript-scoped; ambiguous integer citations across multiple transcripts are rejected, never guessed.

---

## 30. Final Answer Gate

Final output should only pass when:

- schema is valid
- transcript identity is consistent
- evidence exists
- major causal claims are grounded
- unsupported claims are removed or marked uncertain

If evidence is insufficient:

```json
{
  "insufficient_evidence": true,
  "causal_rationales": []
}
```

Abstention is a valid high-quality result.

---

## 31. Retrieval and Quality Metrics

Measure retrieval independently from generation.

Recommended retrieval metrics:

- Recall@k
- Precision@k
- MRR
- nDCG
- conversation-level recall
- relevant-turn recall
- causal-evidence recall
- cross-conversation contamination rate

Generation metrics:

- causal correctness
- evidence grounding
- unsupported claim rate
- JSON validity
- answer completeness
- abstention precision

---

## 32. Latency Metrics

Record:

```text
query_normalization_ms
metadata_filter_ms
dense_retrieval_ms
lexical_retrieval_ms
candidate_merge_ms
reranker_ms
turn_selection_ms
expansion_ms
laya_ms
compression_ms
evidence_pack_ms
prompt_build_ms
llm_ttft_ms
llm_generation_ms
output_parse_ms
grounding_validation_ms
answer_gate_ms
total_ms
```

Also record:

```text
input_tokens
output_tokens
context_tokens
generation_tokens_per_second
```

A stage whose timing is missing from a trace is a defect, not an omission.

---

## 33. Quality/Latency Trade-off

System 4 is expected to be slower than simpler systems.

That is acceptable only if quality improves sufficiently.

The benchmark must therefore report:

```text
quality vs total latency
quality vs retrieval latency
quality vs context tokens
quality vs compute cost
```

Do not label System 4 as better simply because it is more complex.

---

## 34. Benchmark Isolation

For fair comparison across Systems 1–4, keep fixed:

- dataset
- query set
- gold annotations
- LLM
- prompt contract
- temperature
- output schema
- evaluation code

Change only the retrieval/evidence architecture.

Run at least:

- warm-cache benchmark
- cold-start benchmark where practical
- repeated queries for variance

---

## 35. Benchmark Dataset

Use the same sampled benchmark subset selected for the project.

The benchmark should include:

- easy direct causal questions
- multi-turn causal questions
- questions requiring agent action
- questions requiring outcomes
- noisy conversations
- incomplete evidence
- ambiguous queries
- conversations with similar terminology
- adversarial cross-conversation distractors
- unanswerable queries (for abstention precision)

Never tune thresholds only on the final evaluation set.

---

## 36. Reranker Benchmark

Evaluate at least:

```text
no reranker
feature reranker
cross-encoder reranker
```

if resources permit.

Compare:

- Recall@5
- causal evidence recall
- contamination rate
- reranking latency

Choose the implementation based on measured trade-offs.

---

## 37. Compression Benchmark

Compare:

```text
no compression
light compression
aggressive compression
```

Measure:

- context token reduction
- causal recall retained
- answer quality
- generation latency

Compression that saves tokens but removes causal evidence is a regression.

---

## 38. Laya Benchmark

Laya must also be benchmarked.

Measure:

- decision latency
- unnecessary expansion rate
- unnecessary reranking rate
- false rejection rate
- quality impact

The Laya layer should reduce expensive downstream work without materially reducing grounded answer quality.

---

## 39. Caching

Cache immutable artifacts:

- normalized conversations
- indexes
- embeddings for System 1/2 where applicable
- lexical index
- reranker outputs
- deterministic query normalization

Do not cache final answers across different evidence versions without a source/version key.

Cache keys should include:

```text
source_hash
system_version
query_hash
retrieval_config_hash
reranker_version
prompt_version
model_identifier
```

Requests that degraded through a fallback path must not be written to the answer cache. Benchmarks run with answer caching disabled.

---

## 40. Observability

Every request should generate a structured trace.

Example:

```json
{
  "request_id": "...",
  "system": "system4",
  "task": "task1",
  "candidate_conversations": 42,
  "reranked_conversations": 7,
  "selected_transcript_ids": ["..."],
  "selected_turns": [12, 13, 14],
  "laya_decisions": ["ACCEPT"],
  "context_tokens": 1120,
  "llm_generation_ms": 830,
  "grounding_status": "GROUNDED"
}
```

Never log sensitive transcript content by default. Trace sinks must redact content-like keys defensively and rotate by size. Observability failures must never fail a request.

---

## 41. Privacy

The transcript dataset is sensitive hackathon material.

Default logging must contain identifiers, scores, counts, and hashes rather than raw conversation text.

Do not send data to external APIs unless project rules explicitly permit it.

Local LLMs and local rerankers are preferred when confidentiality is required. Remote providers must require an explicit operator opt-in flag; configuration alone must not enable egress.

---

## 42. Prompt Injection Defense

Treat transcript text as untrusted data.

If a transcript contains instructions such as:

```text
ignore previous instructions
reveal system prompt
call an API
```

these are conversation content, not instructions to the application.

Use clear delimiters:

```text
<CONVERSATION_EVIDENCE>
...
</CONVERSATION_EVIDENCE>
```

Evidence that contains the closing delimiter itself must be neutralized before prompt assembly.

The LLM must be instructed to extract from evidence rather than obey it.

---

## 43. Failure Handling

Each degradation must be explicit, logged in the trace, and must never fabricate a result:

```text
If dense retrieval fails:        fall back to lexical candidate retrieval
If lexical retrieval fails:      use dense candidates if available
If both retrievers fail:         abstain with reason
If reranker fails:               use deterministic feature/fusion ranking
If Laya fails:                   use deterministic policy fallback; then fail closed
If compression fails:            use original selected evidence
If LLM fails:                    use the deterministic evidence-cited fallback
If model JSON is invalid:        use the deterministic evidence-cited fallback
If grounding fails:              answer with uncertainty or abstain
```

The system must not silently fabricate a result.

---

## 44. Deterministic Fallback Policy

A simple fallback should be:

```text
1. choose highest-confidence conversation
2. select strongest causal turns
3. expand local neighbors
4. enforce token budget
5. generate
6. validate grounding
```

This guarantees System 4 remains usable even when optional quality components are unavailable. Fallback answers cite only turns of the strongest conversation and never invent mechanisms.

---

## 45. Streamlit Requirements

The UI should expose:

- query input
- selected task (including auto-routing)
- final answer
- causal rationale fields
- evidence/source IDs
- confidence
- insufficient-evidence status
- latency breakdown
- retrieval/reranker statistics

Advanced debug mode may expose:

- candidate conversations
- reranker scores
- Laya decisions
- evidence selection reasons
- compression statistics

Do not expose sensitive raw traces by default.

---

## 46. User Image / External Input Support

If the application later supports multimodal input, keep image processing outside the text retrieval core.

The System 4 retrieval contract remains text/conversation based unless a separate multimodal benchmark is defined.

---

## 47. Testing Strategy

### Unit tests

Test:

- normalization
- candidate merging
- score normalization
- reranking
- causal scoring
- neighbor expansion
- transcript consistency
- compression
- Laya policies
- grounding validation
- output parsing

### Integration tests

```text
query -> retrieval -> reranker -> evidence -> Laya -> LLM -> validator
```

### Regression tests

Store known failure cases, especially:

- unrelated insurance conversations
- refund conversations mixed with insurance
- weak similarity candidates
- missing outcomes
- post-call notes
- noisy speakers
- contradictory evidence
- provider failure and invalid JSON paths
- trace-content leakage

---

## 48. Acceptance Tests

System 4 is not complete until:

- [ ] no cross-conversation contamination occurs on the regression suite (rate = 0)
- [ ] every answer has traceable evidence (non-empty `source_transcripts` + turn ids)
- [ ] unsupported mechanisms are not invented (`MECHANISM_UNSUPPORTED` gate green)
- [ ] insufficient evidence produces abstention/uncertainty
- [ ] reranker improves or preserves retrieval quality relative to the broad candidate pool
- [ ] compression does not materially reduce causal evidence recall
- [ ] Laya decisions are typed and observable
- [ ] all major latency stages (§32) are instrumented on every non-abstained request
- [ ] the benchmark is reproducible (same seed → same questions → same rows)
- [ ] the LLM can be swapped without modifying retrieval code
- [ ] traces contain no transcript text by default

---

## 49. Definition of Done

System 4 is complete when:

1. Broad dense + lexical candidate generation works.
2. Candidates are merged by transcript identity.
3. Conversation consistency is enforced.
4. Reranking works and is benchmarked.
5. Relevant turns are selected.
6. Neighboring causal context is expanded.
7. Evidence is scored by causal role.
8. Laya controls expensive decisions.
9. Context is compressed without losing causal evidence.
10. A structured evidence pack is produced.
11. The LLM generates schema-valid output.
12. Grounding validation runs after generation.
13. The final answer gate can abstain.
14. Latency is measured stage-by-stage.
15. Quality and latency are benchmarked against Systems 1–3.
16. Streamlit exposes the final answer and useful diagnostics.

---

## 50. Recommended Implementation Order

Implement in this order:

### Phase 1 — Foundation
1. Load and normalize transcripts.
2. Build deterministic indexes.
3. Implement dense and lexical candidate retrieval.
4. Add conversation-level deduplication.

### Phase 2 — Quality Retrieval
5. Implement candidate scoring.
6. Add reranker abstraction.
7. Implement turn selection.
8. Implement neighbor expansion.
9. Implement causal evidence scoring.

### Phase 3 — Decision Layer
10. Integrate Laya.
11. Add typed evidence decisions.
12. Add deterministic fallback policies.
13. Add Jev-compatible adapter where required.

### Phase 4 — Evidence Optimization
14. Implement evidence packing.
15. Implement compression.
16. Benchmark compression levels.

### Phase 5 — Generation
17. Integrate Bonsai adapter.
18. Integrate Ollama/Qwen adapter.
19. Integrate approved cloud adapters.
20. Enforce structured output.

### Phase 6 — Validation
21. Implement grounding validator.
22. Implement transcript consistency validation.
23. Implement answer gate.
24. Add regression tests.

### Phase 7 — Benchmark
25. Run retrieval benchmark.
26. Run end-to-end quality benchmark.
27. Run latency benchmark.
28. Compare Systems 1–4.
29. Produce plots and tables.

---

## 51. Engineering Rules

1. Prefer simple deterministic components when they provide the same quality.
2. Do not introduce an agent loop merely for appearance.
3. Every additional stage must have a measurable purpose.
4. Preserve source provenance at every stage.
5. Never use raw retrieval score as final confidence.
6. Never mix unrelated conversations.
7. Never hallucinate missing causal mechanisms.
8. Never silently discard contradictory evidence.
9. Keep retrieval independent from generation.
10. Keep Laya independent from retrieval implementation.
11. Keep LLM providers behind an adapter.
12. Keep benchmarks reproducible.
13. Measure latency before optimizing it.
14. Optimize the largest measured bottleneck.
15. Treat abstention as a valid outcome.

---

## 52. Final Architecture Contract

The canonical System 4 pipeline is:

```text
                    QUERY
                      |
                      v
              Query Normalizer
                      |
                      v
                Task Router
                      |
          +-----------+-----------+
          |                       |
          v                       v
    Dense Retrieval          Lexical Retrieval
          |                       |
          +-----------+-----------+
                      |
                      v
             Candidate Pool
                      |
                      v
         Conversation Consistency
                      |
                      v
                  Reranker
                      |
                      v
             Relevant Turns
                      |
                      v
             Causal Expansion
                      |
                      v
             Evidence Scorer
                      |
                      v
                   Laya
             /       |       \
         reject    expand    accept
                      |
                      v
               Compression
                      |
                      v
               Evidence Pack
                      |
                      v
                    LLM
                      |
                      v
             Structured Parser
                      |
                      v
            Grounding Validator
                      |
                      v
               Answer Gate
                  /       \
              ANSWER     ABSTAIN
```

This architecture is the canonical definition of System 4 for the benchmark.

Any implementation that materially deviates from this design must document the deviation and explain how it affects comparability with Systems 1–3.

---

# Part C — Operational Readiness (production requirements)

The sections below are additive production requirements. They do not alter §0–§52; they make those sections enforceable in operations.

## 53. Configuration & Change Management

- All thresholds, weights, budgets, and lexicons live in `config/settings.py` — never as magic numbers in stage code.
- Every settings object exposes a deterministic fingerprint used in cache keys (§39).
- Prompt text carries a `PROMPT_VERSION`; rerankers carry a `version`; artifacts carry `system_version`. A benchmark result is invalid unless all three are recorded.
- Changes to prompts, weights, or budgets require a benchmark delta (§33) before merge: quality must not regress and contamination must stay at 0.
- Environment configuration is documented in `.env.example`; unknown variables are ignored, and privacy-relevant flags default to the safe value (§41, §56).

## 54. Deployment & Runtime Operations

- **Local-first**: the core has no mandatory third-party runtime dependency; optional components (Streamlit, cross-encoder, remote LLM) are additive and fail soft.
- **Entry points**: `system4-build-index`, `system4-benchmark`, `system4-inspect-trace`, `system4-build-questions`, plus `streamlit run system4/app/streamlit_app.py`.
- **Index artifacts** are version-checked at load; a version mismatch refuses to serve and prints the rebuild command (§57).
- **Startup validation**: the UI reports the exact build command when the index is missing or stale; scripts exit non-zero with actionable messages.
- **Resource profile**: index builds are batch-one-pass; request-path memory is bounded by evidence budgets (§3.4) and the answer cache is LRU-bounded (§39).

## 55. Monitoring & Alerting (SLOs)

Traces (§40) are the monitoring source of truth. Alert on:

| Signal | Condition | Rationale |
|---|---|---|
| `errors[]` non-empty | any request | a stage degraded (§43) — must be visible |
| `answer_gate` | ABSTAIN rate outside calibrated band | retrieval or grounding regression |
| `grounding_status` | UNSUPPORTED/CONTRADICTED among answered | §29 failure |
| `contamination` | any multi-source answer without cross-conversation opt-in | §23 violation — P0 |
| `total_ms` p95 | above calibrated budget | §33 trade-off regression |
| `llm_ttft_ms` / `llm_generation_ms` | generation dominates (>80% of total) for sustained runs | §3.3 signal |
| trace sink | write failures | observability outage (must never fail requests) |

Benchmark-calibrate every threshold in this table; record the calibration with the release.

## 56. Security & Privacy Controls

- **Data egress**: disabled by default. Remote LLM adapters require an explicit operator flag (`SYSTEM4_ALLOW_REMOTE=1`); configuration alone must not enable egress (§41).
- **Secrets**: API keys come from environment variables only; never in code, traces, artifacts, or reports.
- **Logging**: traces carry ids/scores/counts/hashes; a redaction pass strips content-like keys defensively (§40).
- **Prompt injection**: transcript text is untrusted; delimiters plus a neutralization pass (§42), and the model is told to extract, not obey.
- **Artifact integrity**: source content hashes are recorded at index build and reported by every benchmark (§7, §39).
- **Supply chain**: core is stdlib-only; optional dependencies are pinned in `pyproject.toml` extras.

## 57. Artifact & Schema Versioning

- `system_version` (e.g., `system4-v2`) is stamped into index artifacts and every trace.
- Loading validates version equality and fails with a rebuild command on mismatch.
- Question sets are regenerated only with an explicit seed; committed question files are immutable during a benchmark cycle (§34).
- Answer-cache keys include source hash + system version + config fingerprint + reranker/prompt/model identity (§39).
- Schema evolution of answers (§27, §28) is allowed only when validators are updated in the same change.

## 58. CI / Quality Gates

A change merges only when all gates pass:

1. `pytest` — unit, regression, end-to-end suites (§47) green.
2. **Stage-instrumentation gate** — every §32 stage present in a live trace.
3. **Contamination gate** — regression suite contamination rate = 0 (§48).
4. **Determinism gate** — same seed + same settings → identical question set and identical retrieval rows.
5. **Privacy gate** — no transcript text in traces or reports (automated content scan).
6. **Benchmark gate** — when prompts/weights/budgets change, a benchmark delta is attached (§53).

## 59. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Thresholds overfit to eval set | invalid comparison across systems | §35: calibrate on non-final sets; record calibration |
| LLM swap confounds retrieval comparison | unattributable quality change | §25/§34: LLM fixed per experiment |
| Learned reranker unavailable locally | quality drop | §11/§43: feature-reranker fallback, benchmarked (§36) |
| Compression drops causal links | silent quality regression | §37 compression benchmark; drop audit in traces (§21) |
| Cache serves stale answers | wrong answer for new evidence | §39 full key includes source+config+prompt+model |
| Trace leakage of transcript content | privacy breach | §40/§56 redaction + CI privacy gate |
| Remote endpoint misconfiguration | unintended egress | §41/§56 explicit opt-in flag, warn-and-stay-local |
| Post-call notes treated as customer evidence | wrong causal attribution | §24 classification + default exclusion |
| Laya policy drift | unexplainable decisions | §18/§40 typed decisions + logged reasons |

## 60. Release Process — Definition of Production-Ready

A release may ship when:

1. §48 acceptance checklist is fully checked.
2. CI gates (§58) pass on the release commit.
3. Benchmark report (JSON + Markdown) for the release is attached, including variant matrix (§36–§38).
4. Trace sample from a live run shows all §32 stages and zero errors.
5. `AGENTS.md`, `README.md`, and `.env.example` match the shipped behavior (Appendix A updated).
6. Rollback is trivial: artifacts are immutable and version-stamped; reverting the commit restores the previous version gate.

---

# Appendix A — Implementation Traceability Map

Section → module mapping for the current implementation, including documented deviations (§52 requires this):

| Spec section(s) | Implementation |
|---|---|
| §5 layout | As specified; project files live at repo root (`README`, `pyproject`, `AGENTS_SYSTEM_4.md`) instead of inside `system4/` — packaging convention only, no architectural impact |
| §6–§7 ingestion | `ingestion/loader.py`, `ingestion/normalize.py`, `ingestion/artifacts.py` |
| §8–§9 broad retrieval | `retrieval/dense.py`, `retrieval/lexical.py`, `retrieval/broad_retrieval.py` (RRF; per-stream scores preserved for normalization) |
| §10 consistency gate | `retrieval/metadata.py` (floor keeps sparse-metadata candidates observable) |
| §11–§12 reranking | `reranking/reranker.py` (protocol + factory), `reranking/features.py`, `reranking/cross_encoder.py` (optional) |
| §13–§14 selection/expansion | `evidence/selection.py` |
| §15–§16 causal model | `evidence/causal_scoring.py` (mechanism requires explicit agent-side marker) |
| §17–§19 Laya | `decision/policies.py`, `decision/laya.py`, `evidence/gate.py` (feature computation) |
| §18 contracts | `decision/schemas.py` → `schema.py` enums |
| §20 Jev | `decision/jev.py` (local-first; remote endpoint optional, fails to local policy) |
| §21 compression | `evidence/compressor.py` (drops/reorders never rewrite; drop audit ids recorded) |
| §22 evidence pack | `evidence/packer.py` (grouped SOURCE blocks, chronological, budgeted) |
| §23 anti-contamination | pipeline `contamination_gate` + `validation/consistency.py` + transcript-scoped citations |
| §24 post-call | `ingestion/normalize.py::mark_post_call` + selection exclusion |
| §25–§26 LLM | `llm/base.py`, `llm/__init__.py` factory, `llm/ollama.py`, `llm/bonsai.py`, `llm/cloud.py` |
| §27–§28 prompt contracts | `config/prompts.py` (+ `PROMPT_VERSION`, delimiter neutralization) |
| §29 grounding | `validation/grounding.py`, `validation/citation.py` |
| §30 answer gate | `validation/answer_gate.py` (pruning + downgrade + abstain) |
| §31–§33 metrics/trade-off | `evaluation/` (retrieval, quality, latency, reports) |
| §34–§38 benchmarks | `evaluation/benchmark.py` + `scripts/run_benchmark.py` (`--matrix`, `--repeats`) |
| §35 dataset | `scripts/build_questions.py` → `benchmarks/questions.json` (seeded, includes unanswerable + adversarial sets) |
| §39 caching | `cache.py` (full §39 key; degraded requests never cached) |
| §40–§41 observability/privacy | `observability.py` (redaction, rotation, fail-silent) |
| §42 injection defense | `config/prompts.py::sanitize_evidence` + system prompt |
| §43–§44 failure handling | stage guards in `pipeline.py`, fallbacks in `decision/policies.py`, `llm` factory |
| §45 Streamlit | `app/streamlit_app.py` (answer, evidence, latency, stats, debug mode) |
| §47 tests | `tests/` — 92 tests incl. `test_regression.py` (§47 regression catalog) |
| §48 acceptance | enforced by `tests/test_regression.py` + CI gates (§58) |
| §52 contract | `pipeline.py::QualityRAG.ask` implements the canonical order; deviations are timing/observability additions only (`task_route_ms`, `contamination_gate_ms`, `causal_validation_ms`, `consistency_validation_ms`, `laya_redecide_ms`) and are supersets of §32 |

**Documented deviations (intentional, comparability-neutral):**

1. Extra stage timings beyond §32 (listed above) — observability supersets, no pipeline-order change.
2. Laya runs a second pass after expansion (`laya_redecide_ms`) — implements §17 "expansion decision" feedback without altering the stage order of §52.
3. `Candidate.fusion_score` is stored alongside `rerank_score` — required by §9 (never average incompatible scores); fusion is rank-based and normalized at rerank time.
4. Project metadata files live at repository root rather than inside `system4/` — packaging convention only.
