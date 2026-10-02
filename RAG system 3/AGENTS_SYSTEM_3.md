# AGENTS.md — System 3: Vectorless RAG / Structured Lexical Causal Retrieval

> **Project:** Causal Rationale Extraction and Synthesis from Conversational Data on Business Events  
> **System:** System 3 — Vectorless RAG / Structured Lexical Retrieval  
> **Dataset:** `final_transcripts(2).json` (source of truth; preserve raw data)  
> **Target environment:** Local-first development on Apple Silicon macOS, including 16 GB unified-memory machines  
> **Primary LLM:** configurable; must support local Ollama and the existing local Bonsai endpoint  
> **Scope:** Task 1 causal rationale extraction + Task 2 causal rationale synthesis

---

## 0. Mission

Build a production-grade **vectorless RAG system** that answers causal questions over business conversations without using dense embeddings or a vector database.

System 3 is intentionally a pure vectorless architecture and must remain distinct from Systems 1 and 2. Its retrieval signal must come primarily from:

- lexical matching;
- BM25/FTS ranking;
- exact and normalized phrase matching;
- conversation metadata filtering;
- causal/event vocabulary;
- turn-level evidence scoring;
- deterministic neighboring-turn expansion;
- conversation-level candidate selection.

The LLM is the reasoning and synthesis layer, **not the retrieval engine**.

The system must be benchmarkable against the other RAG systems under the same questions, dataset split, output schema, and quality metrics.

---

# 1. Non-Negotiable Design Principles

## 1.1 No dense retrieval

System 3 must not use:

- FAISS;
- Chroma;
- Milvus;
- Qdrant;
- Pinecone;
- pgvector;
- dense embedding similarity;
- semantic vector search.

A dependency that silently creates embeddings is a violation of the System 3 architecture.

## 1.2 Lexical retrieval is the experimental variable

The purpose of System 3 is to measure how far a carefully engineered lexical/structured retriever can go without dense embeddings.

Therefore do not add a dense fallback merely because a query performs poorly.

If lexical retrieval fails, record the failure. That failure is part of the benchmark.

## 1.3 Conversation identity is mandatory

A central lesson from System 1 testing is that globally retrieving eight individually similar chunks can combine unrelated conversations.

System 3 must therefore retrieve **conversation candidates first**, then retrieve evidence within those candidates.

Never treat unrelated chunks from different conversations as one conversation.

Every evidence item must carry:

- `transcript_id`;
- turn start;
- turn end;
- original speaker;
- normalized speaker;
- original text;
- normalized text;
- retrieval score;
- scoring components;
- retrieval stage.

## 1.4 Preserve evidence locality

Causal explanations often span multiple adjacent turns.

After a lexical hit is found, expand locally to neighboring turns rather than retrieving isolated snippets only.

Default starting configuration:

- base lexical hit: one or more turns;
- neighboring expansion: ±2 turns;
- maximum evidence window: task-specific;
- never cross a conversation boundary.

These are defaults, not immutable constants. Benchmark them.

## 1.5 No silent hallucination

The final answer must distinguish:

- explicitly supported cause;
- explicitly supported effect;
- supported mechanism;
- inferred relationship;
- unresolved uncertainty;
- insufficient evidence.

If evidence does not support a causal link, the system must not manufacture one.

## 1.6 Latency must be measured by stage

System 1 demonstrated that retrieval can be only a few milliseconds while generation can dominate the request by orders of magnitude.

System 3 must separately measure:

- query normalization;
- metadata filtering;
- FTS/BM25 retrieval;
- lexical scoring;
- candidate conversation selection;
- evidence expansion;
- evidence packing;
- LLM time-to-first-token if available;
- generation time;
- parsing;
- grounding validation;
- total request time.

Never optimize retrieval based only on total request latency.

---

# 2. Dataset Contract

The source dataset contains approximately 19,621 English two-party conversations with Agent/Customer turns and metadata such as:

- `transcript_id`;
- `time_of_interaction`;
- `domain`;
- `intent`;
- `reason_for_call`;
- conversation turns.

Metadata may be null except for the transcript identity. The actual source schema must be discovered from the file and handled defensively.

## 2.1 Raw data is immutable

Never overwrite or mutate `final_transcripts(2).json`.

Create derived artifacts separately, for example:

```text
system2/
├── data/
│   ├── raw/
│   │   └── final_transcripts(2).json
│   ├── processed/
│   │   ├── conversations.jsonl
│   │   ├── turns.jsonl
│   │   └── sampling_manifest.json
│   └── index/
│       ├── lexical.sqlite
│       ├── lexical.sqlite-shm
│       └── lexical.sqlite-wal
├── src/
├── tests/
├── benchmarks/
├── configs/
├── logs/
└── app/
```

Do not commit the raw hackathon dataset unless project rules explicitly permit it.

## 2.2 Preserve original text

Normalization must never replace the original transcript.

Store both:

```text
text_original
text_normalized
```

Original punctuation, hesitations, filler words, speaker anomalies, and conversational noise may contain useful evidence and must remain recoverable.

## 2.3 Canonical turn schema

Normalize each turn into a canonical internal representation:

```json
{
  "transcript_id": "...",
  "turn_id": 24,
  "speaker_original": "Customer",
  "speaker": "customer",
  "text_original": "...",
  "text_normalized": "...",
  "domain": "...",
  "intent": "...",
  "reason_for_call": "...",
  "time_of_interaction": "..."
}
```

Retain the original speaker label even when a canonical speaker mapping is possible.

## 2.4 Speaker normalization

Expected canonical values:

- `customer`;
- `agent`;
- `unknown`.

Do not assume every transcript uses exactly these labels. Some data may contain names or unusual labels.

If a mapping cannot be established confidently, retain `unknown` rather than inventing a role.

---

# 3. Sampling and Evaluation Population

The benchmark uses approximately 8,000 conversations sampled at the **conversation level** using reproducible stratification by `intent`.

Requirements:

- preserve the full 19,621-source dataset untouched;
- use a fixed random seed;
- save the sampling manifest;
- prevent transcript leakage between benchmark partitions;
- stratify by intent where possible;
- record strata that cannot be sampled exactly because of insufficient population.

Example manifest fields:

```json
{
  "seed": 42,
  "source_hash": "...",
  "source_count": 19621,
  "sample_count": 8000,
  "stratification_key": "intent",
  "created_at": "..."
}
```

The benchmark query set must be reproducible.

---

# 4. System 3 Architecture

The canonical pipeline is:

```text
                         USER QUESTION
                               │
                               ▼
                       Query Normalizer
                               │
                               ▼
                     Query Signal Extractor
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
             keywords       phrases       metadata
                 │             │             │
                 └─────────────┼─────────────┘
                               ▼
                    Metadata / intent filter
                               │
                               ▼
                     SQLite FTS5 / BM25
                               │
                               ▼
                 Conversation-level candidates
                               │
                               ▼
                    Turn-level lexical scoring
                               │
                               ▼
                     Causal cue prioritization
                               │
                               ▼
                     Neighbor turn expansion
                               │
                               ▼
                       Evidence deduplication
                               │
                               ▼
                         Evidence packing
                               │
                               ▼
                          Evidence gate
                               │
                               ▼
                             LLM
                               │
                               ▼
                         JSON validation
                               │
                               ▼
                       Grounding validator
                               │
                               ▼
                           Answer gate
                               │
                               ▼
                          Streamlit UI
```

The architectural identity of System 3 is:

> **Query → lexical/metadata candidate retrieval → conversation-aware evidence selection → causal scoring → LLM synthesis.**

---

# 5. Retrieval Philosophy

## 5.1 Why BM25/FTS instead of embeddings?

The dataset contains explicit lexical signals:

- business terms;
- product names;
- intent vocabulary;
- complaint terms;
- refund/cancellation/booking terms;
- policy terms;
- causal verbs;
- outcome phrases.

BM25 is useful when the query shares important terms with the evidence.

System 3 should exploit these signals deliberately rather than pretending lexical retrieval is semantic retrieval.

## 5.2 Retrieval must be hierarchical

Do not implement:

```text
query → top 8 chunks globally
```

Prefer:

```text
query
 ↓
metadata constraints
 ↓
conversation candidates
 ↓
turn candidates inside conversations
 ↓
neighbor expansion
 ↓
evidence pack
```

This directly addresses the cross-conversation contamination observed in System 1.

---

# 6. SQLite FTS5 / BM25 Index

SQLite FTS5 is the preferred baseline because it is:

- local;
- deterministic;
- lightweight;
- easy to inspect;
- suitable for tens of thousands of conversations;
- easy to package with the project;
- compatible with lexical ranking;
- free from external infrastructure.

Do not add a server database unless profiling proves SQLite inadequate.

## 6.1 Recommended logical tables

### conversations

```sql
CREATE TABLE conversations (
    transcript_id TEXT PRIMARY KEY,
    time_of_interaction TEXT,
    domain TEXT,
    intent TEXT,
    reason_for_call TEXT
);
```

### turns

```sql
CREATE TABLE turns (
    turn_pk INTEGER PRIMARY KEY,
    transcript_id TEXT NOT NULL,
    turn_id INTEGER NOT NULL,
    speaker TEXT,
    speaker_original TEXT,
    text_original TEXT NOT NULL,
    text_normalized TEXT NOT NULL,
    FOREIGN KEY(transcript_id) REFERENCES conversations(transcript_id)
);
```

### FTS table

```sql
CREATE VIRTUAL TABLE turns_fts USING fts5(
    transcript_id UNINDEXED,
    turn_id UNINDEXED,
    speaker,
    text_normalized,
    content='turns',
    content_rowid='turn_pk'
);
```

The exact implementation may differ if an external BM25 library is faster or more controllable, but the retrieval remains lexical.

## 6.2 Conversation-level index

System 3 should maintain a conversation-level lexical representation as well as turn-level text.

A useful representation is:

```text
conversation_text =
    domain
    + intent
    + reason_for_call
    + important turn text
```

This permits candidate conversation selection before turn retrieval.

Do not blindly concatenate every turn if this creates an unnecessarily huge index. Benchmark title/metadata + full text representations.

---

# 7. Query Processing

The query processor must produce a structured object rather than just a string.

Example:

```json
{
  "raw_query": "Why was the customer unhappy and what did the agent do to resolve it?",
  "terms": [
    "customer",
    "unhappy",
    "agent",
    "resolve"
  ],
  "phrases": [],
  "causal_terms": [
    "why",
    "resolve"
  ],
  "event_terms": [
    "unhappy",
    "issue"
  ],
  "outcome_terms": [
    "resolve"
  ],
  "metadata_hints": {},
  "task": "task1"
}
```

## 7.1 Normalization

Normalization may include:

- lowercase copy;
- Unicode normalization;
- whitespace normalization;
- conservative punctuation handling;
- contraction normalization where safe;
- common spelling normalization only when deterministic.

Do not stem or lemmatize blindly. Benchmark whether stemming improves or harms conversational retrieval.

## 7.2 Stopwords

Use stopword handling carefully.

Words such as `why`, `how`, `did`, `cause`, `reason`, `result`, `after`, `before`, and `because` can be useful for task classification even if they have low lexical retrieval value.

Do not remove causal cues before query classification.

---

# 8. Query Expansion Without Embeddings

System 3 may use deterministic lexical expansion.

Allowed:

- synonym dictionaries;
- domain vocabulary maps;
- causal cue dictionaries;
- spelling normalization;
- phrase variants;
- abbreviation expansion;
- metadata-aware term expansion.

Example:

```text
refund
→ refund, reimbursement, money back, return
```

Example:

```text
cause
→ cause, reason, why, trigger, led to, resulted in
```

Expansion must be deterministic and logged.

Do not use an LLM to generate query expansion during the retrieval benchmark unless explicitly defined as part of a separate ablation. Otherwise LLM query expansion contaminates the System 3 architecture and makes latency comparison difficult.

---

# 9. Metadata Filtering

Metadata is a major advantage of this system.

When the query provides or strongly implies a known field, filter or boost accordingly.

Potential fields:

- domain;
- intent;
- reason_for_call;
- date/time;
- business category.

Use filtering only when evidence supports the constraint.

Do not infer a domain merely because a word appears in the query.

## 9.1 Hard filters vs boosts

Default:

- exact explicit metadata constraint → hard filter;
- inferred metadata hint → soft boost.

This avoids catastrophic recall loss from incorrect metadata inference.

---

# 10. BM25 Ranking

Use BM25 as the primary lexical score.

Conceptually:

```text
BM25(query, document)
```

should be combined with deterministic auxiliary signals.

Recommended starting score:

```text
final_score =
    0.70 * normalized_bm25
  + 0.10 * metadata_score
  + 0.08 * phrase_score
  + 0.07 * causal_cue_score
  + 0.05 * outcome_cue_score
```

These weights are **starting values only**.

They must be calibrated on the benchmark rather than treated as scientifically optimal.

Every score component must be logged so that ranking decisions are explainable.

---

# 11. Causal Lexical Signals

Maintain deterministic cue dictionaries.

## 11.1 Cause cues

Examples:

```text
because
reason
cause
caused by
due to
result of
why
what led to
trigger
stemming from
```

## 11.2 Effect/outcome cues

Examples:

```text
therefore
so
resulted
led to
resolved
fixed
confirmed
decided
agreed
accepted
cancelled
refunded
```

## 11.3 Agent-action cues

Examples:

```text
I can
I will
let me
I'll
we can
I've updated
I've processed
I can help
I'll arrange
```

These are not proof of causality. They are retrieval prioritization signals only.

---

# 12. Turn-Level Evidence Selection

After candidate conversations are identified, score their turns.

A turn should receive additional priority when it has:

- strong query-term overlap;
- exact phrase overlap;
- causal cue overlap;
- customer problem statements;
- agent actions;
- explicit outcome statements;
- temporal relationship markers.

A useful conceptual score is:

```text
turn_score =
    lexical_score
  + causal_bonus
  + problem_bonus
  + action_bonus
  + outcome_bonus
```

Keep the raw components.

---

# 13. Neighbor Expansion

Causal evidence frequently exists around the lexical hit rather than inside the exact hit.

Example:

```text
Turn 18: Customer reports problem
Turn 19: Agent asks clarification
Turn 20: Customer explains cause
Turn 21: Agent proposes action
Turn 22: Customer agrees
Turn 23: Agent confirms resolution
```

If Turn 20 is retrieved, System 3 should consider:

```text
Turns 18–22
```

rather than sending Turn 20 alone.

Default:

```text
neighbor_window = 2
```

For Task 2, a wider window may be justified, but must remain bounded.

Never expand into a different `transcript_id`.

Never expand indefinitely.

---

# 14. Conversation Candidate Selection

Default candidate policy:

```text
conversation_top_k = 3
```

For each candidate:

```text
retrieve top lexical turns
→ merge overlapping windows
→ expand neighbors
→ score evidence
```

Do not automatically pass all three conversations to the LLM.

Apply an evidence gate.

Example:

```text
candidate score < threshold
→ discard

candidate score sufficient
→ retain
```

The thresholds must be calibrated using benchmark data.

---

# 15. Evidence Gate

The evidence gate should answer:

> Is there enough directly relevant lexical evidence to ask the LLM to synthesize an answer?

Possible outcomes:

```text
SUFFICIENT
EXPAND
INSUFFICIENT
```

### SUFFICIENT

Relevant conversation and causal evidence are present.

### EXPAND

Relevant terms were found, but the causal chain is incomplete. Increase neighboring context or retrieve another candidate from the same conversation.

### INSUFFICIENT

No sufficiently relevant conversation was found.

Do not send obviously irrelevant evidence to the LLM merely to force an answer.

---

# 16. Evidence Pack Format

Use a compact, structured format.

Example:

```text
CONVERSATION: 52ac...
DOMAIN: Insurance
INTENT: ...
REASON: ...

TURN 024 | CUSTOMER
that's good to know so uh how often do these upsell attempts typically succeed

TURN 025 | AGENT
in my experience upsell attempts like this one generally succeed quite well especially when we highlight the added value customers receive

TURN 026 | CUSTOMER
and do you think um the phrasing affects success rates
```

Keep evidence contiguous where possible.

Do not dump metadata that is irrelevant to the question.

---

# 17. Task 1 — Causal Rationale Extraction

Task 1 should answer questions such as:

> What caused the customer's dissatisfaction, what did the agent do, and what was the resulting effect?

Output schema:

```json
{
  "task": "task1",
  "causal_rationales": [
    {
      "cause": "...",
      "effect": "...",
      "mechanism": "...",
      "evidence_turns": [12, 13, 14],
      "strength": "explicit"
    }
  ],
  "confidence": 0.0,
  "insufficient_evidence": false
}
```

The model must not invent a mechanism when the evidence only supports cause and effect.

Allowed strength values:

- `explicit`;
- `strongly_supported`;
- `inferred`;
- `insufficient`.

If the causal mechanism is absent, use `null` or an explicit missing value rather than `—` if the downstream parser expects JSON nullability.

---

# 18. Task 2 — Causal Rationale Synthesis

Task 2 is broader than Task 1.

It may require synthesizing the interaction into a coherent causal chain.

However, synthesis must remain **conversation-local**.

Do not merge multiple unrelated customers merely because they discuss similar products.

Expected conceptual structure:

```text
Trigger / business event
        ↓
Customer motivation/problem
        ↓
Agent response/action
        ↓
Intermediate effect
        ↓
Outcome / decision
        ↓
Uncertainty
```

Output schema:

```json
{
  "task": "task2",
  "summary": "...",
  "causal_chain": [
    {
      "step": 1,
      "event": "...",
      "role": "customer|agent|business",
      "evidence_turns": [1, 2]
    }
  ],
  "key_causes": [],
  "effects": [],
  "uncertainties": [],
  "confidence": 0.0,
  "insufficient_evidence": false
}
```

If multiple conversations are legitimately relevant because the user explicitly asks for a comparison, preserve separate conversation identities in the output.

Never collapse them into one fictional interaction.

---

# 19. Prompt Design

The system prompt must clearly define:

1. evidence is authoritative;
2. unsupported claims are forbidden;
3. source conversation identity must be respected;
4. causal inference must be conservative;
5. output must be valid JSON;
6. uncertainty must be reported.

Example core instruction:

```text
Use only the supplied evidence.
Do not combine facts from different conversations into one interaction.
Every causal claim must be directly supported or clearly marked as inferred.
If evidence is insufficient, return insufficient_evidence=true.
Do not invent missing outcomes.
Return valid JSON only.
```

---

# 20. Grounding Validation

After LLM generation, validate every claimed evidence turn.

For each cited turn:

- transcript exists;
- turn exists;
- turn belongs to retrieved conversation;
- claim has lexical/semantic support from the turn or its evidence window;
- no fabricated turn numbers;
- no conversation mixing.

System 3 should perform deterministic validation wherever possible.

A lightweight validator can check whether important answer entities and phrases are represented in the evidence.

Do not treat string overlap alone as proof of semantic correctness.

---

# 21. Cross-Conversation Contamination Guard

This is mandatory because System 1 testing exposed the failure mode.

Every generation request must maintain:

```text
allowed_transcript_ids
```

The answer gate must reject or flag output that claims one continuous interaction using facts from multiple unrelated IDs.

Example invalid synthesis:

```text
Amelia complained about pricing, Jack selected silver,
and Joseph visited the store...
```

if those statements came from unrelated conversations.

The correct behavior is either:

- answer using the relevant conversation only;
- explicitly present separate interactions if the query requests comparison;
- return insufficient evidence.

---

# 22. Post-Call / Internal Note Protection

The dataset may contain post-call or internal-note style content after the customer interaction.

Examples may include QA notes, internal observations, churn notes, or other text after a goodbye.

Such text must not automatically be treated as customer-facing causal evidence.

Implement a classifier/heuristic boundary where possible:

```text
conversation turns
      ↓
normal interaction
      ↓
post-call artifact detection
      ↓
mark separately
```

Store:

```text
is_post_call_artifact
```

Do not delete the original content. Exclude it from evidence by default unless the task explicitly asks about internal notes or post-call analysis.

---

# 23. Noisy Conversation Handling

Do not aggressively clean conversational language.

Preserve:

- fillers;
- pauses;
- false starts;
- incomplete sentences;
- punctuation errors;
- repetitions.

Normalize only for indexing and ranking.

The original form must remain available to the LLM and evidence viewer.

---

# 24. Laya / Jev Integration

Where applicable, use Laya as the local typed decision/gating layer rather than putting routing logic into arbitrary application code.

Laya should own deterministic decisions such as:

```text
query classification
metadata filter selection
evidence sufficiency
candidate acceptance/rejection
answer acceptance/rejection
benchmark decision traces
```

Jev compatibility should be treated as a protocol/schema boundary where needed, not as a requirement to send the hackathon dataset to an external hosted service.

The system must remain fully runnable locally.

Example conceptual decision:

```json
{
  "decision": "EXPAND_EVIDENCE",
  "reason": "Lexical hit is relevant but causal chain incomplete",
  "transcript_id": "...",
  "score": 0.37
}
```

All decision logic must be observable.

---

# 25. LLM Adapter Boundary

The retrieval system must not depend directly on a specific LLM.

Create an interface such as:

```python
class LLMClient(Protocol):
    def generate(self, prompt: str, **kwargs) -> str:
        ...
```

Supported implementations should include:

```text
BonsaiClient
OllamaClient
GroqClient (optional external benchmark)
```

The benchmark must be able to change the LLM without changing retrieval code.

For local Apple Silicon testing, `qwen3.5:9b-mlx` may be used through Ollama if installed.

The existing Bonsai endpoint remains a supported adapter, not a retrieval dependency.

---

# 26. Generation Configuration

Default benchmark generation settings should be deterministic where the provider permits:

```text
temperature = 0
structured JSON output = enabled where supported
```

Keep context bounded.

Initial targets:

```text
Task 1: 800–1500 evidence tokens
Task 2: 1500–2500 evidence tokens
```

These are starting targets and must be benchmarked.

Do not automatically send the model 4K+ tokens simply because the context window allows it.

System 1 testing showed that oversized context can become very expensive on slow local generation backends.

---

# 27. Latency Instrumentation

Every request must return a trace similar to:

```json
{
  "query_normalization_ms": 0.2,
  "metadata_filter_ms": 0.1,
  "fts_retrieval_ms": 2.1,
  "conversation_ranking_ms": 0.4,
  "turn_ranking_ms": 0.6,
  "neighbor_expansion_ms": 0.1,
  "evidence_pack_ms": 0.1,
  "evidence_gate_ms": 0.1,
  "generation_ttft_ms": 0,
  "generation_ms": 850,
  "parse_ms": 0.2,
  "grounding_ms": 0.3,
  "answer_gate_ms": 0.1,
  "total_ms": 955
}
```

If TTFT is unavailable, record null rather than inventing it.

Also record:

```text
input_tokens
output_tokens
context_tokens
```

and, when available:

```text
input_tokens_per_second
output_tokens_per_second
```

---

# 28. Benchmark Protocol

Use the same benchmark questions for all four RAG systems.

Hold constant:

- benchmark conversations;
- questions;
- LLM;
- temperature;
- output schema;
- evaluation rules;
- hardware;
- prompt where architecture does not require a structural change.

System 3's retrieval mechanism is the independent variable.

## 28.1 Required metrics

### Retrieval

- Recall@k where labels exist;
- relevant conversation retrieval rate;
- evidence precision;
- irrelevant conversation rate;
- cross-conversation contamination rate.

### Generation

- causal correctness;
- evidence grounding;
- unsupported claim rate;
- JSON validity;
- incomplete/insufficient-evidence accuracy.

### Performance

- p50 latency;
- p95 latency;
- mean latency;
- retrieval latency;
- generation latency;
- tokens/sec;
- peak memory where measurable.

### Operational

- index build time;
- index size;
- cold-start time;
- warm-request latency.

---

# 29. Ablation Studies

System 3 should support these ablations:

### A. Pure BM25

```text
query → BM25 → top conversations → LLM
```

### B. BM25 + metadata

```text
query → metadata + BM25 → candidates → LLM
```

### C. BM25 + causal cues

```text
query → BM25 + causal scoring → candidates → LLM
```

### D. BM25 + neighbor expansion

```text
query → BM25 → local context expansion → LLM
```

### E. Full System 3

```text
metadata
+ BM25
+ causal scoring
+ neighbor expansion
+ evidence gate
→ LLM
```

These experiments explain which parts actually improve quality.

---

# 30. Example Failure Modes

## Failure: relevant phrase but wrong conversation

Cause:

```text
lexical similarity without conversation-level ranking
```

Fix:

```text
rank conversations first
```

## Failure: correct cause but missing outcome

Cause:

```text
exact hit was too narrow
```

Fix:

```text
neighbor expansion
```

## Failure: many unrelated insurance conversations

Cause:

```text
generic terms dominate BM25
```

Fix:

```text
metadata boosts
specific phrase boosts
causal cue scoring
```

## Failure: LLM combines customers

Cause:

```text
multiple transcript IDs passed in one evidence pack
```

Fix:

```text
conversation-local evidence pack
cross-conversation contamination guard
```

## Failure: 100-second+ generation

Cause:

```text
slow LLM backend and/or oversized context
```

Fix:

```text
measure generation independently
reduce evidence tokens
use faster LLM backend
```

Do not spend engineering effort shaving 2 ms from FTS while generation takes 100 seconds.

---

# 31. Caching

Safe caches include:

- parsed query cache;
- normalized query cache;
- FTS result cache;
- conversation candidate cache;
- evidence-window cache.

Cache keys must include relevant configuration versions.

Do not cache generated answers during quality evaluation unless explicitly testing cache behavior.

Otherwise cached answers can contaminate benchmark results.

---

# 32. Reproducibility

Record:

```text
source dataset hash
index hash
code version / git commit
configuration hash
random seed
LLM model/version
prompt version
benchmark question version
```

Every benchmark output must be traceable to these values.

---

# 33. Configuration

Use a configuration file rather than hard-coded constants.

Example:

```yaml
system: system2
retrieval:
  backend: sqlite_fts5
  conversation_top_k: 3
  turns_per_conversation: 5
  neighbor_window: 2
  query_expansion: true
  metadata_boost: true
  causal_scoring: true

evidence:
  task1_max_tokens: 1500
  task2_max_tokens: 2500
  min_score: 0.30

generation:
  provider: ollama
  model: qwen3.5:9b-mlx
  temperature: 0
  num_ctx: 4096
```

Thresholds are experimental configuration, not code constants.

---

# 34. Security and Data Handling

The source dataset is internal hackathon material.

Default behavior:

- process locally;
- do not upload transcripts to third-party services;
- do not log complete conversations in production logs;
- redact or hash transcript IDs in telemetry when appropriate;
- keep API keys in environment variables;
- never commit secrets;
- never expose raw conversation data through debug endpoints.

If an external cloud LLM is used for a benchmark, it must be an explicit experiment and must comply with the hackathon's data-sharing rules.

---

# 35. Logging

Structured logs should contain:

```json
{
  "request_id": "...",
  "task": "task1",
  "query_hash": "...",
  "candidate_count": 3,
  "selected_transcript_ids": ["..."],
  "evidence_turn_count": 7,
  "retrieval_ms": 4.1,
  "generation_ms": 820.2,
  "grounding_pass": true
}
```

Do not log the complete transcript by default.

A developer debug mode may expose evidence locally, but must be disabled in normal operation.

---

# 36. Streamlit Interface

The UI should expose:

1. question input;
2. task selector;
3. final answer;
4. causal chain;
5. evidence turns;
6. transcript ID;
7. retrieval score;
8. evidence decision;
9. latency breakdown;
10. model/backend.

Do not overload the primary UI with internal debug information. Put detailed traces in an expandable section.

Example:

```text
Answer
────────────────────────
...

Evidence
────────────────────────
Conversation: abc123
Turns 18–23

Why this evidence?
────────────────────────
BM25: 0.71
Causal bonus: 0.12
Metadata: 0.10

Latency
────────────────────────
Retrieval: 4 ms
Generation: 840 ms
Total: 860 ms
```

---

# 37. Testing Requirements

## Unit tests

Test:

- transcript parsing;
- speaker normalization;
- text normalization;
- FTS indexing;
- BM25 ranking;
- query expansion;
- metadata filtering;
- causal scoring;
- neighbor expansion;
- evidence merging;
- token-budget packing;
- JSON parsing;
- grounding validation;
- cross-conversation guard.

## Integration tests

Test:

```text
query → FTS → candidate conversations → evidence → LLM → validator
```

with a small deterministic fixture.

## Regression tests

Maintain known examples for:

- refund;
- cancellation;
- booking errors;
- pricing dissatisfaction;
- plan comparison;
- upsell;
- product defects;
- service complaints.

---

# 38. Retrieval Evaluation Fixture

Before benchmarking the full 8K sample, create a manually inspectable fixture of at least 50 questions.

For every question record:

```json
{
  "question_id": "q001",
  "transcript_id": "...",
  "task": "task1",
  "expected_evidence_turns": [18, 19, 20, 21],
  "expected_cause": "...",
  "expected_effect": "..."
}
```

The fixture is not a replacement for the full benchmark. It is a debugging instrument.

---

# 39. System 1 Lessons That Must Carry Forward

The first System 1 test produced several important lessons.

## Lesson 1 — Retrieval was not the bottleneck

Observed behavior:

```text
retrieval ≈ few milliseconds
embedding ≈ sub-millisecond to low milliseconds
generation ≫ retrieval
```

Therefore System 3 must not be optimized around retrieval microseconds while ignoring LLM generation.

## Lesson 2 — Global chunk retrieval can contaminate evidence

A result with several chunks around scores such as:

```text
0.4185
0.4183
0.4171
0.4120
...
```

can represent multiple semantically similar but unrelated conversations.

System 3 explicitly prevents this with conversation-level candidate selection.

## Lesson 3 — Causal questions need contiguous context

A single lexical hit may identify the problem but miss:

- the trigger;
- the agent response;
- the outcome.

Neighbor expansion is therefore a core component, not a cosmetic feature.

## Lesson 4 — Context size affects generation latency

System 1 generated over roughly 4,000 context tokens and experienced very high generation latency on the slow local backend.

System 3 should keep evidence compact and task-specific.

## Lesson 5 — Quality must be separated from latency

A fast answer is not useful if evidence is wrong.

Measure retrieval quality and generation quality separately from latency.

## Lesson 6 — The LLM can correctly refuse when evidence is weak

An empty causal rationale can be the correct output when retrieval fails.

Do not modify prompts to force an answer from inadequate evidence.

---

# 40. Recommended Initial Implementation Order

Do not implement every advanced feature simultaneously.

## Phase 1 — Data contract

1. Load raw JSON.
2. Validate schema.
3. Normalize turns.
4. Preserve original fields.
5. Produce statistics.

## Phase 2 — Lexical index

1. Create SQLite database.
2. Insert conversations.
3. Insert turns.
4. Build FTS5 indexes.
5. Add indexes for metadata.
6. Run retrieval smoke tests.

## Phase 3 — Pure lexical baseline

Implement:

```text
query → BM25 → top conversations → top turns
```

No causal scoring yet.

Benchmark it.

## Phase 4 — Conversation-aware retrieval

Add:

```text
conversation candidate ranking
→ turn retrieval inside candidate
```

Benchmark the change.

## Phase 5 — Neighbor expansion

Add ±2 turn expansion.

Benchmark:

- evidence recall;
- context size;
- latency.

## Phase 6 — Causal scoring

Add deterministic causal/action/outcome cues.

Benchmark whether the cues improve causal evidence retrieval.

## Phase 7 — Evidence gate

Add:

```text
SUFFICIENT / EXPAND / INSUFFICIENT
```

## Phase 8 — LLM integration

Use the same LLM interface used by System 1.

## Phase 9 — Grounding validation

Reject unsupported claims and invalid turn references.

## Phase 10 — Streamlit

Only after retrieval and generation are stable.

## Phase 11 — Benchmark

Run the common benchmark against Systems 1–4.

---

# 41. Definition of Done

System 3 is complete only when all of the following are true:

- [ ] No dense embeddings are used.
- [ ] No vector database is used.
- [ ] Raw dataset remains untouched.
- [ ] Sampling is reproducible.
- [ ] SQLite FTS/BM25 index builds deterministically.
- [ ] Metadata filtering works.
- [ ] Conversation-level candidate selection works.
- [ ] Turn-level ranking works.
- [ ] Neighbor expansion never crosses conversation boundaries.
- [ ] Causal lexical scoring is observable.
- [ ] Evidence gate works.
- [ ] Task 1 output is valid structured JSON.
- [ ] Task 2 output is valid structured JSON.
- [ ] Grounding validation works.
- [ ] Cross-conversation contamination is detected/prevented.
- [ ] Post-call artifacts are handled safely.
- [ ] LLM provider is configurable.
- [ ] Latency is measured per stage.
- [ ] p50/p95 latency can be reported.
- [ ] Retrieval quality can be reported.
- [ ] Generation quality can be reported.
- [ ] Benchmark results are reproducible.
- [ ] Streamlit displays evidence and latency.
- [ ] Unit tests pass.
- [ ] Integration tests pass.

---

# 42. Final System 3 Contract

The final system must satisfy this conceptual contract:

```text
                 QUERY
                   │
                   ▼
          Lexical Query Analysis
                   │
                   ▼
        Metadata + FTS/BM25 Search
                   │
                   ▼
         TOP CONVERSATION CANDIDATES
                   │
                   ▼
       TURN-LEVEL LEXICAL SCORING
                   │
                   ▼
        CAUSAL SIGNAL PRIORITIZATION
                   │
                   ▼
        LOCAL NEIGHBOR EXPANSION
                   │
                   ▼
             EVIDENCE GATE
                   │
                   ▼
            COMPACT EVIDENCE
                   │
                   ▼
                  LLM
                   │
                   ▼
           STRUCTURED OUTPUT
                   │
                   ▼
          GROUNDING VALIDATOR
                   │
                   ▼
             ANSWER GATE
                   │
                   ▼
               RESPONSE
```

The defining principle is:

> **Use lexical and structured signals to find the right conversation and the right causal evidence; use the LLM only to interpret and synthesize that evidence.**

System 3 must remain a genuine vectorless retrieval architecture so that its benchmark result is scientifically interpretable against System 1, System 3, and System 4.


---

# APPENDIX Z — SYSTEM 3 LAYA IMPLEMENTATION BLUEPRINT

## Z.1 Purpose

Use Laya as the fast typed-decision/control plane around the vectorless retrieval pipeline. The large generative model remains responsible for causal rationale extraction and synthesis.

## Z.2 Recommended decisions

### Query classification

Question: `task_type`

Options:
- `task1_extraction` — identify explicit causal rationales supported by the evidence.
- `task2_synthesis` — synthesize a broader causal account from one or more relevant conversations.
- `unsupported` — request cannot be answered from this system's evidence contract.

### Evidence sufficiency

Question: `evidence_state`

Options:
- `strong` — enough direct evidence to answer.
- `expand` — promising evidence but causal chain may cross turns/chunks.
- `weak` — lexical match exists but evidence is not causally useful.
- `reject` — no sufficiently relevant evidence.

### Evidence inclusion

Question: `include_turn`

Options:
- `include` — directly supports cause/effect/mechanism/outcome.
- `supporting` — useful context but not itself causal.
- `exclude` — irrelevant, contradictory without resolution, or post-call contamination.

### Final answer gate

Question: `answer_status`

Options:
- `grounded`
- `partially_grounded`
- `insufficient_evidence`
- `cross_conversation_contamination`

## Z.3 Deterministic fallback

If Laya is disabled, use lexical rules:

- strong direct cause/effect cue + same transcript_id -> include
- matching named entity + relevant business-event term -> boost
- question/answer pair around the query -> boost
- goodbye/post-call marker -> downweight or exclude
- conflicting transcript_id -> never merge into one causal chain
- low BM25 score with no exact phrase/entity overlap -> reject or expand

## Z.4 Laya does not replace retrieval

Do not use Laya to rank 19,621 conversations as a substitute for an inverted index. First retrieve a bounded lexical candidate set; then apply Laya to the small candidate set. This keeps latency predictable and preserves the vectorless experimental claim.

## Z.5 Benchmark isolation

Report three Laya configurations when possible:

1. `rules_only`
2. `rules_plus_laya`
3. `laya_control_only` where appropriate

This reveals whether Laya improves System 3 because of its decision quality or simply because of additional compute.

## Z.6 Reference

Laya currently documents a Jev-compatible `/v1/systemone` interface and typed decisions. Verify the installed version and checkpoint before implementation because model and serving details can change.

