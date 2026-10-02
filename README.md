Causal RAG for Business Conversations
> **Anonymous technical project**
> 
> A production-oriented retrieval-augmented generation framework for
> extracting and synthesizing causal rationales from noisy business
> conversations.
￼
Overview
This project investigates how different Retrieval-Augmented Generation
(RAG) architectures perform on causal reasoning over multi-turn
business conversations.
The central idea is that causal RAG is not simply a matter of retrieving
semantically similar text. A useful answer may require:
	●	identifying the correct conversation,
	●	locating the relevant causal turns,
	●	preserving turn order and speaker identity,
	●	connecting cause → mechanism → effect,
	●	avoiding evidence from unrelated conversations,
	●	validating the generated answer against source evidence.
Four independent RAG systems were implemented and evaluated under a
common benchmark boundary:
￼
System                  Architecture            Main Purpose
￼
System 1            Dense Vector RAG        Semantic retrieval
baseline
System 2            Hybrid RAG              Dense + lexical
retrieval
System 3            Vectorless RAG          BM25/FTS-style
retrieval without
embeddings
System 4            Agentic / Quality RAG   Multi-stage retrieval, reranking, Laya gates and grounding
The systems do not feed their outputs into one another. They receive
the same benchmark inputs and are compared by the benchmark suite.
￼
Problem
The task is based on causal rationale extraction and synthesis from
conversational business events.
A conversation can contain a causal chain such as:
Customer problem
      ↓
Reason / motivation
      ↓
Trigger or business event
      ↓
Agent response / intervention
      ↓
Customer reaction
      ↓
Outcome / resolution
The relevant evidence may be distributed across many turns.
A conventional top-k chunk retriever can therefore produce a
superficially relevant but causally incorrect context.
Core challenge
> **Retrieve enough of the right conversation to reconstruct the causal
> chain without contaminating it with unrelated conversations.**
￼
Dataset
The project uses a proprietary internal conversational dataset.
Dataset statistics
Property                                      Value
￼
Conversations                            19,621
Total turns                             686,764
Mean turns / conversation                      ~35
Median turns / conversation                  35
Turn range                                1–79
Unique intents                               35
Interaction dates                             9
Domain                                    Insurance
Raw speaker labels                          661
Canonical Agent/Customer turn share      94.19%
The conversations contain realistic conversational noise including:
	●	fillers and hesitations,
	●	incomplete punctuation,
	●	non-grammatical segments,
	●	inconsistent speaker labels,
	●	semantically similar business interactions,
	●	multi-turn causal relationships.
The full proprietary dataset is not redistributed with this
repository.
￼
Exploratory Data Analysis
EDA was performed before building the retrieval systems.
Important observations:
1. Conversations are long enough for chunk boundaries to matter
With a median of approximately 35 turns, a causal relationship can
easily span multiple chunks.
This motivated:
Conversation retrieval
        ↓
Relevant turn retrieval
        ↓
Neighboring-turn expansion
        ↓
Causal evidence selection
rather than relying exclusively on globally ranked chunks.
2. Speaker labels are not perfectly standardized
Most turns use canonical Agent / Customer labels, but the dataset
contains many raw speaker-label variants.
Therefore the pipeline treats speaker normalization as a data-processing
concern rather than assuming a fixed schema.
3. Similar business vocabulary creates retrieval ambiguity
Terms such as:
	●	refund,
	●	booking,
	●	premium,
	●	cancellation,
	●	plan,
	●	customer,
	●	coverage,
can occur in many unrelated conversations.
This is particularly problematic for lexical/vectorless retrieval.
4. Metadata can help constrain retrieval
Available metadata such as intent, domain, reason-for-call and
transcript identity can be used as retrieval/reranking signals when
appropriate.
￼
System Architecture
Shared evaluation boundary
All systems receive the same benchmark inputs:
                    Benchmark Question
                           │
                    Query Normalization
                           │
             ┌─────────────┴─────────────┐
             │                           │
          System 1                   System 2
       Dense Vector                  Hybrid
             │                           │
          System 3                   System 4
        Vectorless                 Agentic / Quality
             │                           │
             └─────────────┬─────────────┘
                           │
                    Benchmark Suite
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
     Quality            Latency            Tokens
     Retrieval          Grounding          Diagnostics
Important: System 4 is an independent architecture. It does not
consume System 1, 2, or 3 outputs.
￼
System 1 — Dense Vector RAG
Goal
Establish a semantic-retrieval baseline.
Pipeline
Query
  ↓
Normalization
  ↓
Embedding
  ↓
Dense Vector Search
  ↓
Conversation / Chunk Selection
  ↓
Neighbor Expansion
  ↓
Evidence Gate
  ↓
LLM
  ↓
Structured JSON
  ↓
Grounding Validation
Main lesson
Early experiments showed that globally retrieving similar chunks can mix
multiple conversations.
For causal tasks, the better design is:
Retrieve conversation
        ↓
Retrieve turns within conversation
        ↓
Expand local context
        ↓
Build evidence pack
Main difficulties
	●	cross-conversation contamination,
	●	weak causal evidence near the top rank,
	●	local LLM generation latency,
	●	environment/dependency issues during benchmarking.
￼
System 2 — Hybrid RAG
System 2 combines semantic and lexical retrieval.
                 Query
                   │
          ┌────────┴────────┐
          ↓                 ↓
     Dense Search       Lexical Search
          │                 │
          └────────┬────────┘
                   ↓
              Candidate Merge
                   ↓
           Conversation Ranking
                   ↓
            Turn Expansion
                   ↓
             Evidence Pack
                   ↓
                  LLM
Why hybrid?
Dense retrieval provides semantic similarity.
Lexical retrieval provides:
	●	exact business terms,
	●	names,
	●	phrases,
	●	domain terminology,
	●	identifiers.
Combining both can recover evidence missed by either method
independently.
￼
System 3 — Vectorless RAG
System 3 deliberately removes embeddings and vector databases.
Retrieval
Query
  ↓
Normalization
  ↓
BM25 / FTS / Lexical Search
  ↓
Metadata + Phrase Features
  ↓
Conversation Aggregation
  ↓
Turn Selection
  ↓
Evidence Pack
  ↓
LLM
Why build a vectorless system?
To test whether semantic vector retrieval is actually necessary for this
dataset.
It also provides:
	●	simpler deployment,
	●	lower retrieval overhead,
	●	deterministic lexical behavior,
	●	easier debugging,
	●	no embedding model dependency.
Main limitation
Lexical overlap can be misleading.
Two unrelated conversations can contain almost identical vocabulary
while describing different customers, events, or outcomes.
￼
System 4 — Agentic / Quality RAG
System 4 is the most structured architecture.
Query
  ↓
Parallel Candidate Retrieval
  ├── Dense Retrieval
  └── Lexical Retrieval
           ↓
     Candidate Merge
           ↓
        Reranking
           ↓
   Causal Turn Selection
           ↓
       Laya Decision
           ↓
   Evidence Completeness
           ↓
       Compression
           ↓
          LLM
           ↓
    Grounding Validation
           ↓
       Answer Gate
Laya
Laya is used as a lightweight decision/control layer.
Typical decisions include:
	●	whether evidence is sufficient,
	●	whether an answer should be produced,
	●	task routing,
	●	answer gating,
	●	deterministic policy checks.
Laya is not used as a replacement for the retrieval architecture
being evaluated.
￼
Task Definitions
Task 1 — Causal Rationale Extraction
The system extracts structured causal relationships.
Typical output:
{
  "cause": "...",
  "effect": "...",
  "mechanism": "...",
  "evidence_turns": [12, 15, 18],
  "strength": "explicit"
}
The answer must be grounded in the retrieved conversation.
￼
Task 2 — Causal Rationale Synthesis
Task 2 is intended to reconstruct a broader causal chain.
A useful synthesis may contain:
Business event
      ↓
Customer motivation
      ↓
Cause / trigger
      ↓
Agent intervention
      ↓
Outcome
      ↓
Uncertainty
Task 2 requires broader evidence coverage than a narrow extraction task.
￼
Benchmark Suite
The benchmark suite evaluates the systems using a common set of
questions and a common output contract.
Retrieval metrics
Conversation Recall
Whether the relevant transcript is retrieved.
MRR
Mean Reciprocal Rank of the relevant conversation.
Evidence Precision
How much of the selected evidence is actually relevant.
Evidence Recall
How much of the required/reference evidence was recovered.
Cross-conversation contamination
Measures whether unrelated transcripts enter the evidence set.
￼
Latency metrics
The benchmark records:
	●	total latency,
	●	retrieval latency,
	●	embedding latency,
	●	reranking latency,
	●	evidence construction latency,
	●	prompt construction latency,
	●	LLM generation latency,
	●	output parsing latency,
	●	grounding validation latency,
	●	p50 latency,
	●	p95 latency.
This stage-level instrumentation was important because retrieval was
often much faster than generation.
￼
Latest Benchmark Snapshot
The latest complete benchmark archive contains 13 Task 1 queries.
￼
System           Conversation     Evidence     Evidence   Contamination Mean Latency
Recall    Precision       Recall
￼
System 1                  —          —          —             —          —
System 2 —     84.62%   84.62%   62.82%      15.38%  15.17 s
Hybrid
System 3 —     53.85%   44.16%   38.46%      61.54%   7.90 s
Vectorless
System 4 —     84.62%   84.62%    7.05%      15.38% 250 ms* Agentic
* System 4’s 250 ms result is not directly comparable to the full
LLM-powered latency because the measured run used a deterministic
fallback and therefore bypassed the normal LLM generation stage.
System 1 had a benchmark-environment dependency failure in the latest
run, so its quantitative result is intentionally not fabricated.
￼
Interpreting the Results
System 2
System 2 provides a strong evidence-quality baseline.
Its main limitation is latency.
The traces showed that local LLM generation dominates the request rather
than retrieval.
Main optimization target
Generation speed
        ↓
Smaller/faster local model
        +
Context reduction
        +
Caching
        +
Efficient structured generation
￼
System 3
System 3 is substantially faster than System 2.
However:
Lexical similarity
        ≠
Causal relevance
Its high contamination indicates that lexical matching needs stronger
discrimination.
Potential improvements:
	●	field-weighted BM25,
	●	exact entity boosts,
	●	phrase matching,
	●	metadata constraints,
	●	speaker-aware scoring,
	●	conversation-level aggregation,
	●	negative-term penalties.
￼
System 4
System 4 substantially improves candidate selection and reduces
contamination.
However, its very low evidence recall reveals a different problem:
> **The system is selecting evidence that is precise but incomplete.**
In other words:
High precision
      +
Low recall
      ↓
Too much evidence compression
The next step is to ensure causal completeness before compression.
￼
Major Failure Modes
￼
Problem                 Likely Cause            Proposed Fix
￼
Cross-conversation      Global chunk ranking    Conversation-first
mixing                                          retrieval
Weak causal evidence    Isolated/filler turns   Neighbor expansion
rank highly
Vectorless ambiguity    Shared vocabulary       BM25 field weighting +
entity boosts
High LLM latency        Generation dominates    Smaller/faster model
Low System 4 evidence   Aggressive compression  Causal completeness
recall                                          gate
Grounded but causally   Grounding ≠ causal      Causal consistency
wrong answer            validity                validator
Task routing            Weak task contract      Stricter schema and
instability                                     task-specific
evaluation
System 4 timing anomaly Deterministic fallback  Benchmark full LLM generation
￼
Grounding vs. Causal Correctness
One of the most important lessons from the project is:
> **A grounded answer is not automatically a causally correct answer.**
A response may quote real turns from the correct transcript but
incorrectly label one statement as the cause and another as the effect.
Therefore the validator should check:
Source grounding
Does the claim appear in the selected evidence?
Chronology
Does the causal ordering make sense?
Speaker attribution
Is the statement attributed to the correct participant?
Causal consistency
Does the conversation actually support the claimed relationship?
Evidence completeness
Are the required causal links present?
￼
Local LLM Strategy
The project is designed around local inference where possible.
The LLM layer is abstracted so that retrieval experiments are
independent from the model provider.
Possible local deployment:
RAG Application
      │
      ↓
LLM Adapter
      │
      ├── Ollama
      │      └── Qwen MLX
      │
      └── Local model endpoint
This makes it possible to compare model size and generation speed
without rewriting the RAG pipeline.
￼
Production-Oriented Architecture
Recommended modular structure:
project/
│
├── data/
│   ├── normalization/
│   ├── sampling/
│   └── schemas/
│
├── retrieval/
│   ├── dense/
│   ├── lexical/
│   ├── hybrid/
│   └── reranking/
│
├── evidence/
│   ├── expansion/
│   ├── causal_selection/
│   └── compression/
│
├── decision/
│   └── laya/
│
├── generation/
│   ├── ollama/
│   └── local_adapters/
│
├── validation/
│   ├── grounding/
│   ├── chronology/
│   └── causal_consistency/
│
├── benchmark/
│   ├── questions/
│   ├── metrics/
│   ├── traces/
│   └── reports/
│
├── app/
│   └── streamlit/
│
└── README.md
￼
Reproducibility
All four systems should be evaluated with:
	1.	the same sampled benchmark questions,
	2.	the same task definitions,
	3.	the same output schema,
	4.	the same evaluation metrics,
	5.	the same grounding criteria,
	6.	the same benchmark environment wherever possible.
Do not compare:
System A + full LLM generation
against:
System B + deterministic fallback
as if their latency measurements were equivalent.
￼
Future Work
The following experiments were identified but were not fully completed
because of time constraints.
1. Complete System 1 benchmark
Resolve the benchmark environment dependency issue and rerun the full
suite.
2. Enable real LLM generation in System 4
Measure:
	●	TTFT,
	●	output tokens,
	●	tokens/sec,
	●	total latency,
	●	generation quality.
3. Improve System 4 evidence recall
Introduce a causal completeness checklist:
Cause present?
      ↓
Agent intervention present?
      ↓
Effect/outcome present?
      ↓
Chronology valid?
      ↓
Compress
If a required component is missing, expand the local conversation
context.
4. Improve Vectorless RAG
Test:
	●	BM25 parameter tuning,
	●	field weights,
	●	exact phrase boosts,
	●	entity matching,
	●	metadata filtering,
	●	speaker-aware features,
	●	conversation-level score aggregation.
5. Model-size ablation
Compare local models under identical prompts and contexts, particularly
smaller and larger Qwen MLX variants.
6. Larger benchmark
The current 13-query run is primarily diagnostic.
A final evaluation should contain substantially more questions with
balanced Task 1 and Task 2 coverage.
7. Human / calibrated quality evaluation
Add explicit labels for:
	●	causal correctness,
	●	evidence completeness,
	●	grounding,
	●	hallucination,
	●	attribution,
	●	uncertainty handling.
￼
Key Findings
The project leads to several practical conclusions:
Finding 1
Causal RAG is primarily an evidence-management problem.
Finding 2
Conversation identity matters more than simply increasing top-k.
Finding 3
Vectorless retrieval is viable but requires stronger lexical
discrimination.
Finding 4
Hybrid retrieval provides a strong evidence-quality baseline, but
generation becomes the latency bottleneck.
Finding 5
System 4 demonstrates the value of multi-stage retrieval and decision
gating, but aggressive evidence compression currently hurts evidence
recall.
Finding 6
Grounding validation must be separated from causal validation.
Finding 7
The four systems should remain independent experimental alternatives
and only share the benchmark/evaluation boundary.
￼
Recommended Next Iteration
The recommended engineering sequence is:
1. Fix System 1 benchmark
          ↓
2. Complete Task 2 benchmark
          ↓
3. Improve System 4 evidence recall
          ↓
4. Enable real LLM generation in System 4
          ↓
5. Optimize System 2 generation latency
          ↓
6. Improve System 3 lexical discrimination
          ↓
7. Run a larger balanced benchmark
          ↓
8. Add human / LLM-judge quality evaluation
          ↓
9. Select architecture based on measured
   quality + latency + token efficiency
￼
Project Status
￼
Component                           Status
￼
Dataset EDA                         Complete
8K stratified sampling              Complete
System 1 architecture               Implemented / benchmark environment
needs rerun
System 2 Hybrid RAG                 Implemented
System 3 Vectorless RAG             Implemented
System 4 Agentic RAG                Implemented
Laya decision layer                 Integrated
Benchmark suite                     Implemented
Retrieval metrics                   Implemented
Latency tracing                     Implemented
Contamination analysis              Implemented
Task 1 benchmark                    Available
Task 2 benchmark                    Needs dedicated quantitative run
Full System 4 LLM benchmark         Pending
Larger-scale final evaluation       Pending
￼
Privacy / Data Notice
The underlying conversational dataset is internal/proprietary.
This repository should not contain:
	●	raw confidential transcripts,
	●	personally identifying information,
	●	credentials,
	●	API keys,
	●	private model endpoints,
	●	generated caches containing sensitive source data.
Use sanitized benchmark artifacts and aggregate metrics when sharing the
project externally.
￼
Presentation and Report
The project includes:
	●	IEEE-style anonymous technical report
	●	architecture diagrams
	●	EDA visualizations
	●	benchmark plots
	●	latency analysis
	●	contamination analysis
	●	failure-mode analysis
	●	project presentation
The presentation summarizes the four-system architecture, benchmark
methodology, latest results, failure modes, and future work.
￼
Final Summary
This project explores four independent ways of solving causal
conversational RAG:
System 1
Dense Vector RAG
        │
        │ semantic retrieval
        ▼

System 2
Hybrid RAG
        │
        │ dense + lexical
        ▼

System 3
Vectorless RAG
        │
        │ lexical / BM25
        ▼

System 4
Agentic / Quality RAG
        │
        │ retrieval + reranking
        │ + causal selection
        │ + Laya + grounding
        ▼

COMMON BENCHMARK
quality • retrieval • grounding
latency • tokens • diagnostics
The core engineering insight is that improving causal RAG is not simply
about finding a better embedding model or a larger LLM. The most
important component is constructing complete, transcript-faithful,