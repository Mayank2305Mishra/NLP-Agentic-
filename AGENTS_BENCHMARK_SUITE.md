# AGENTS.md — Production Benchmark Suite for Four RAG Systems

## 0. Mission

Build a reproducible, production-grade benchmark suite for the four RAG systems:

1. System 1 — Dense Vector RAG
2. System 2 — Hybrid RAG
3. System 3 — Vectorless RAG
4. System 4 — Agentic / Quality RAG

The benchmark must determine not only which system produces the best answers, but **why**: retrieval quality, evidence quality, grounding, latency, token usage, generation efficiency, context efficiency, robustness, and cost where applicable.

The benchmark is an evaluator, not another RAG implementation. It must not silently alter a system's retrieval or generation behavior.

---

## 1. Core Experimental Principle

The comparison must isolate architecture differences.

Keep fixed whenever the experiment intends to compare retrieval architectures:

- same benchmark questions
- same gold/reference evidence where available
- same task labels
- same source corpus snapshot
- same LLM
- same system prompt
- same output schema
- same temperature and decoding settings
- same maximum output tokens
- same evaluation code
- same hardware/API environment

Only the intended system component may differ.

Never compare System 1 with one LLM and System 3 with another and call the result a retrieval benchmark.

If model comparison is desired, create a separate **LLM Matrix Experiment** with the RAG architecture held constant.

---

## 2. Benchmark Questions

The benchmark must contain two primary task families.

### Task 1 — Causal Rationale Extraction

Measure whether the system can identify:

- trigger/business event
- customer motivation/problem
- cause
- effect
- mechanism
- relevant agent action
- outcome/resolution
- uncertainty
- supporting turns

Task 1 should favor precise, minimal evidence rather than broad summarization.

### Task 2 — Causal Rationale Synthesis

Measure whether the system can synthesize a coherent causal explanation from the relevant conversation evidence.

Measure:

- causal completeness
- factual consistency
- evidence grounding
- chronology
- attribution
- uncertainty handling
- irrelevant-content resistance

Do not require the answer to reproduce the wording of a reference answer.

---

## 3. Benchmark Dataset Structure

Create a versioned benchmark dataset.

Recommended structure:

```text
benchmarks/
├── datasets/
│   ├── benchmark_questions.jsonl
│   ├── benchmark_questions.parquet
│   ├── gold_evidence.jsonl
│   ├── gold_answers.jsonl
│   └── dataset_manifest.json
│
├── splits/
│   ├── dev.jsonl
│   ├── test.jsonl
│   └── stress.jsonl
│
├── configs/
│   ├── system1.yaml
│   ├── system2.yaml
│   ├── system3.yaml
│   └── system4.yaml
│
├── runs/
├── reports/
└── artifacts/
```

Each benchmark item must have a stable `question_id`.

Recommended schema:

```json
{
  "question_id": "q_000001",
  "task": "task1",
  "query": "...",
  "transcript_ids": ["..."],
  "gold_turns": [12, 13, 14],
  "gold_evidence": ["..."],
  "reference_answer": "...",
  "gold_causes": ["..."],
  "gold_effects": ["..."],
  "gold_outcomes": ["..."],
  "difficulty": "medium",
  "metadata": {}
}
```

Do not put test-set answers into prompts.

---

## 4. Benchmark Splits

Use at least three splits.

### Dev

Used for:

- threshold calibration
- retrieval tuning
- prompt development
- evidence-gate calibration
- Laya decision thresholds

### Test

Locked evaluation set.

No tuning based on test performance.

### Stress

Designed to expose failure modes:

- similar conversations
- repeated customer names
- repeated business terminology
- short conversations
- long conversations
- noisy punctuation
- filler speech
- weak lexical overlap
- high lexical overlap but wrong conversation
- causal chain crossing chunk boundaries
- multiple candidate causes
- incomplete outcomes
- contradictory statements
- post-call artifacts
- metadata missing
- irrelevant but semantically similar conversations

---

## 5. Dataset Leakage Prevention

The benchmark runner must verify:

- no question appears in both dev and test
- no transcript appears across incompatible splits unless explicitly allowed
- no reference answer is sent to the LLM
- no gold evidence is inserted into system prompts
- no benchmark metadata leaks the expected answer

Generate a dataset manifest containing:

- corpus hash
- benchmark hash
- split hashes
- creation timestamp
- question counts
- transcript counts
- code version

---

## 6. Four-System Contract

Every RAG system must expose the same interface.

```python
result = system.answer(
    query=query,
    task=task,
    config=config,
)
```

The returned object must contain:

```json
{
  "answer": {},
  "evidence": [],
  "trace": {},
  "usage": {},
  "system_metadata": {}
}
```

The benchmark must never inspect private implementation details to obtain the answer.

---

## 7. Required Trace Schema

Every system must expose a normalized trace.

```json
{
  "timing_ms": {
    "request": 0,
    "query_normalization": 0,
    "retrieval": 0,
    "reranking": 0,
    "evidence_selection": 0,
    "compression": 0,
    "llm_generation": 0,
    "validation": 0
  },
  "retrieval": {
    "candidate_count": 0,
    "conversation_count": 0,
    "chunk_count": 0,
    "selected_count": 0
  },
  "tokens": {
    "input": 0,
    "output": 0,
    "context": 0,
    "total": 0
  },
  "retrieval_scores": [],
  "evidence_ids": [],
  "transcript_ids": [],
  "decisions": []
}
```

Missing fields must be represented as null, not silently omitted.

---

## 8. Latency Benchmarking

Latency is a first-class metric.

Measure:

- cold-start latency
- warm latency
- p50
- p90
- p95
- p99
- mean
- standard deviation
- minimum
- maximum

Measure separately:

```text
query normalization
retrieval
reranking
Laya decision time
evidence packing
compression
prompt construction
LLM time-to-first-token
LLM generation
JSON parsing
validation
end-to-end
```

Never report only total latency.

System 1 previously demonstrated why this matters: retrieval can take only a few milliseconds while generation consumes tens or hundreds of seconds.

### Cold vs warm

Run dedicated cold and warm experiments.

Cold runs must not be mixed into normal latency statistics.

For local models record:

- model load time
- first-token latency
- generation duration
- prompt processing duration if exposed

For cloud APIs record:

- network/request overhead when measurable
- provider-reported usage
- first-token latency when streaming is available
- total response latency

---

## 9. Token Efficiency

Measure:

- input tokens
- evidence/context tokens
- prompt tokens
- output tokens
- total tokens
- useful evidence tokens
- irrelevant evidence tokens
- answer tokens

Derived metrics:

```text
context_compression_ratio
= raw_retrieved_tokens / final_context_tokens

useful_evidence_ratio
= gold-overlapping-evidence-tokens / final_context_tokens

output_efficiency
= useful_answer_information / output_tokens

retrieval_token_overhead
= retrieved_context_tokens / answer_tokens
```

Do not use vague token-efficiency claims. Every metric must have a precise definition.

---

## 10. Generation Efficiency

Measure:

- tokens/sec
- input tokens/sec where available
- output tokens/sec
- time to first token
- total generation time
- output length

For streaming APIs, calculate:

```text
TTFT = first token timestamp - request start

TPOT = generation duration / generated tokens

tokens_per_second = generated tokens / generation duration
```

If provider timing is unavailable, mark the metric unavailable rather than estimating it from unrelated timestamps.

---

## 11. Retrieval Quality Metrics

Every system must be evaluated independently of the LLM.

Required metrics where gold evidence exists:

### Recall@K

Fraction of questions for which at least one gold evidence unit appears in top K.

Report:

- Recall@1
- Recall@3
- Recall@5
- Recall@10

### Precision@K

Fraction of retrieved evidence units that are relevant.

### MRR

Mean reciprocal rank of the first relevant evidence unit.

### MAP

Mean average precision across queries where applicable.

### nDCG@K

Use graded relevance where gold annotations support it.

### Conversation Recall

Whether the correct `transcript_id` is retrieved before turn-level retrieval.

This is mandatory because the dataset contains many semantically similar conversations.

---

## 12. Cross-Conversation Contamination

This is a mandatory benchmark metric because it was observed in earlier experiments.

Define:

```text
contamination_rate
= queries where final evidence contains an irrelevant transcript_id
  / total evaluated queries
```

Also report:

- number of distinct transcript IDs in final context
- gold transcript coverage
- irrelevant transcript count
- mixed-conversation rate

A system must not receive a quality score advantage for producing a correct-looking answer from unrelated conversations.

---

## 13. Evidence Quality Metrics

Measure:

### Evidence precision

Does each cited evidence unit support the claim it is attached to?

### Evidence recall

Does the final evidence include the information required to reconstruct the causal chain?

### Evidence sufficiency

Can an evaluator reconstruct the answer from the evidence without relying on hidden information?

### Evidence redundancy

How much of the evidence repeats information already present?

### Evidence compactness

How much irrelevant material remains after evidence selection/compression?

---

## 14. Causal Quality Metrics

Do not evaluate Task 1 as ordinary semantic similarity alone.

Score the following dimensions independently:

1. Trigger identification
2. Cause identification
3. Effect identification
4. Mechanism identification
5. Agent action identification
6. Outcome identification
7. Temporal ordering
8. Attribution correctness
9. Uncertainty correctness
10. Evidence grounding

Each dimension should be scored independently before producing any aggregate score.

Do not create a single score unless the weighting is explicitly defined in the benchmark configuration.

---

## 15. Task 1 Exact Evaluation

Parse model output into the canonical schema.

```json
{
  "causal_rationales": [
    {
      "cause": "...",
      "effect": "...",
      "mechanism": "...",
      "evidence_turns": [1, 2, 3],
      "confidence": 0.0
    }
  ],
  "insufficient_evidence": false
}
```

Evaluate:

- schema validity
- cause semantic match
- effect semantic match
- mechanism match
- evidence-turn overlap
- grounding
- unsupported-claim rate
- abstention correctness

Use both automatic and judge-based evaluation where appropriate.

---

## 16. Task 2 Evaluation

Evaluate synthesis on:

- causal completeness
- factual consistency
- coverage of important events
- chronological coherence
- entity consistency
- business-event consistency
- evidence grounding
- uncertainty handling
- irrelevant detail rate

Task 2 must not be rewarded for mentioning many unrelated conversations.

---

## 17. Automatic Semantic Evaluation

Use embedding-based or model-based semantic evaluation only as a supplementary metric.

It must not replace evidence-grounding evaluation.

Possible metrics:

- BERTScore
- semantic similarity
- LLM-as-judge
- claim-level entailment

Every judge prompt must explicitly distinguish:

- supported fact
- plausible inference
- unsupported claim
- contradiction

Store raw judge outputs for auditability.

---

## 18. LLM-as-Judge Protocol

Use a separate judge model from the production model where possible.

Judge input:

```text
Question
Reference answer / gold rationale if available
System answer
Retrieved evidence
```

The judge must not receive the system name when testing quality to reduce preference bias.

Require structured output:

```json
{
  "grounding": 0,
  "causal_correctness": 0,
  "completeness": 0,
  "relevance": 0,
  "uncertainty_handling": 0,
  "unsupported_claims": [],
  "reason": "..."
}
```

Run inter-rater checks on a manually reviewed subset.

---

## 19. Hallucination / Unsupported Claim Metrics

Extract atomic claims from the final answer.

For each claim classify:

- supported
- partially supported
- unsupported
- contradicted

Report:

```text
unsupported_claim_rate
contradiction_rate
supported_claim_rate
```

An answer with high semantic similarity but unsupported claims must not receive a high grounding score.

---

## 20. Abstention Quality

Systems are allowed to say evidence is insufficient.

Measure:

- correct abstention
- incorrect abstention
- false confidence
- evidence sufficiency calibration

Report:

```text
abstention_precision
abstention_recall
coverage
selective_accuracy
```

Plot quality against answer coverage for different evidence thresholds.

---

## 21. Calibration

If systems emit confidence, evaluate it.

Metrics:

- Expected Calibration Error
- Brier score
- reliability diagram
- confidence vs correctness

Never interpret raw model confidence as calibrated probability without validation.

---

## 22. Robustness Suite

Create perturbation tests:

- punctuation removed
- filler words inserted
- capitalization changed
- spelling noise
- paraphrased query
- shortened query
- entity ambiguity
- irrelevant context injection
- reordered irrelevant evidence
- long-context distraction

Measure performance degradation relative to clean inputs.

---

## 23. Retrieval Adversarial Tests

Create cases where:

- wrong conversations share the same keywords
- correct conversation uses synonyms
- customer names repeat
- business names repeat
- multiple conversations discuss the same policy
- a correct cause appears several turns before the effect
- outcome occurs after a long agent explanation

The benchmark must specifically test whether each architecture handles these cases.

---

## 24. Long-Context Efficiency

Bucket queries by context size:

```text
<500 tokens
500–1000
1000–2000
2000–4000
4000–8000
8000+
```

Measure quality and latency per bucket.

This reveals whether an architecture gains quality by simply sending more tokens to the LLM.

---

## 25. Scaling Benchmark

Run benchmark subsets of:

- 50 queries
- 100 queries
- 500 queries
- 1,000 queries
- full benchmark where feasible

Measure:

- throughput
- total runtime
- memory
- CPU/GPU utilization when available
- API rate-limit behavior
- failure/retry rate

Do not extrapolate full-benchmark cost from a tiny sample without reporting the assumptions.

---

## 26. Concurrency Benchmark

Test concurrency levels appropriate to the deployment.

Example:

```text
1
2
4
8
16
```

Report:

- throughput
- p50/p95 latency
- error rate
- queue time
- model saturation
- memory pressure

Do not run concurrency tests against providers whose free tier would be violated.

---

## 27. Cost Benchmark

Where a paid or metered API is used, calculate:

```text
input_cost
output_cost
retrieval_infrastructure_cost
reranker_cost
embedding_cost
estimated_cost_per_1k_queries
estimated_cost_per_10k_queries
```

For free/local systems report:

- API cost = 0
- infrastructure/runtime assumptions separately

Never claim “free” means zero operational cost.

---

## 28. Failure Metrics

Track:

- timeout rate
- API errors
- malformed JSON
- parser failures
- empty answers
- retrieval failures
- evidence-gate failures
- model failures
- retries
- rate-limit errors
- OOM events

Every failed query must remain in the run artifact.

Never silently drop failures from aggregate metrics.

---

## 29. Reproducibility

Every run must save:

```text
run_id
Git commit / code version
benchmark version
corpus hash
system version
LLM model
LLM model version if available
prompt version
configuration hash
random seed
hardware
OS/runtime
Python version
package lock/hash
start time
end time
```

A benchmark result without configuration provenance is incomplete.

---

## 30. Run Directory

Create:

```text
runs/<run_id>/
├── manifest.json
├── config.yaml
├── environment.json
├── questions.jsonl
├── results.jsonl
├── failures.jsonl
├── traces.jsonl
├── judge_results.jsonl
├── retrieval_results.jsonl
├── aggregate_metrics.json
├── latency_metrics.json
├── token_metrics.json
├── quality_metrics.json
├── contamination_metrics.json
├── plots/
│   ├── latency_cdf.png
│   ├── latency_boxplot.png
│   ├── tokens_vs_quality.png
│   ├── context_vs_quality.png
│   ├── recall_at_k.png
│   ├── precision_at_k.png
│   ├── quality_vs_latency.png
│   └── system_comparison.png
└── report.md
```

---

## 31. Result Record

Each question/system pair must generate one durable record.

```json
{
  "run_id": "...",
  "question_id": "...",
  "system": "system3",
  "task": "task1",
  "answer": {},
  "evidence": [],
  "metrics": {},
  "timing": {},
  "usage": {},
  "errors": [],
  "judge": {},
  "environment": {}
}
```

Do not store only aggregates.

Per-query records are mandatory for debugging.

---

## 32. Benchmark Runner

Implement:

```text
benchmark/
├── runner.py
├── registry.py
├── schemas.py
├── datasets.py
├── metrics/
│   ├── retrieval.py
│   ├── evidence.py
│   ├── causal.py
│   ├── grounding.py
│   ├── latency.py
│   ├── tokens.py
│   ├── robustness.py
│   └── calibration.py
├── judges/
├── reports/
├── plots/
├── exporters/
└── configs/
```

The runner must support:

```bash
python -m benchmark.runner \
  --systems system1 system2 system3 system4 \
  --split test \
  --limit 100 \
  --seed 42
```

Also support single-system debugging:

```bash
python -m benchmark.runner --systems system3 --question-id q_000123
```

---

## 33. Paired Evaluation

For every question, run all selected systems against the same query.

Do not compare independent random samples.

Store paired results so that statistical tests can exploit within-question pairing.

---

## 34. Statistical Analysis

Report uncertainty.

Use paired bootstrap confidence intervals for system differences where appropriate.

For paired binary outcomes use suitable paired tests.

For continuous latency metrics report confidence intervals and percentiles.

Avoid declaring meaningful differences from tiny raw gaps without uncertainty analysis.

Do not convert statistical significance into a subjective “best” label automatically.

---

## 35. Recommended Aggregate Dashboard

The final report should contain at least:

### Quality

- causal correctness
- grounding
- evidence precision
- evidence recall
- answer relevance
- abstention quality

### Retrieval

- Recall@K
- MRR
- nDCG
- conversation recall
- contamination rate

### Efficiency

- p50/p95 latency
- TTFT
- generation tok/s
- input tokens/query
- context tokens/query
- output tokens/query
- compression ratio

### Reliability

- failure rate
- malformed-output rate
- timeout rate
- retry rate

### Cost

- tokens/query
- cost/query where applicable
- cost/1k queries

---

## 36. Pareto Analysis

Do not collapse every metric into a single score by default.

Generate Pareto-frontier plots for:

```text
quality vs latency
quality vs context tokens
quality vs output tokens
quality vs cost
retrieval recall vs retrieval latency
```

A system is Pareto-efficient when no measured alternative is better on all selected dimensions.

This is descriptive analysis, not an automatic overall ranking.

---

## 37. System-Specific Diagnostics

### System 1 — Dense Vector RAG

Measure:

- embedding latency
- vector-search latency
- embedding token/cost if applicable
- Recall@K
- semantic retrieval failures

### System 2 — Hybrid RAG

Measure:

- BM25 latency
- dense latency
- fusion latency
- contribution of lexical-only candidates
- contribution of dense-only candidates
- fusion overlap

### System 3 — Vectorless RAG

Measure:

- indexing time
- index size
- BM25 latency
- exact-match contribution
- entity-match contribution
- metadata-filter contribution
- lexical failure rate

### System 4 — Agentic / Quality RAG

Measure separately:

- initial retrieval
- reranking
- Laya decisions
- evidence selection
- compression
- validation
- additional LLM calls

System 4 must not hide its additional computation inside “retrieval latency.”

---

## 38. Laya Benchmark Integration

Laya is a decision/control component, not a quality oracle.

Where integrated, record:

```json
{
  "decision": "EXPAND",
  "confidence": 0.0,
  "decision_latency_ms": 0,
  "policy": "evidence_gate_v1",
  "reason_code": "LOW_COVERAGE"
}
```

Measure:

- Laya decision latency
- decisions by category
- expansion frequency
- rejection frequency
- false rejection
- false acceptance
- impact on downstream quality
- impact on latency

Run ablations:

```text
System 4 without Laya
System 4 with Laya
```

This is necessary to prove whether Laya improves the system rather than merely adding complexity.

---

## 39. Laya / Jev Isolation

If a Jev-compatible System-One interface is used, record protocol overhead separately.

Do not allow framework transport, serialization, or control-plane latency to contaminate retrieval latency.

Benchmark:

```text
core decision latency
framework overhead
end-to-end latency
```

---

## 40. Ablation Suite

The benchmark suite must support component ablations.

Required examples:

### Retrieval

- no metadata
- no query expansion
- no neighbor expansion
- no causal scoring
- top-k changes

### System 3

- BM25 only
- BM25 + metadata
- BM25 + entity matching
- BM25 + causal scoring
- BM25 + neighbor expansion

### System 4

- no reranker
- no compression
- no Laya
- no validation
- no adaptive retrieval

### Generation

- short context
- medium context
- long context

Every ablation must have a unique configuration ID.

---

## 41. Context-Length Ablation

For each system evaluate several evidence budgets.

Example:

```text
500 tokens
1000
1500
2500
4000
6000
```

Measure whether quality improvements come from better retrieval or merely more context.

Plot:

```text
context_tokens vs causal_quality
context_tokens vs latency
context_tokens vs grounding
```

---

## 42. Top-K Ablation

Evaluate:

```text
K = 1, 3, 5, 8, 10, 20
```

For each K record:

- retrieval recall
- evidence precision
- contamination
- context size
- generation latency
- final quality

Do not assume larger K is better.

---

## 43. Threshold Calibration

All retrieval/evidence thresholds must be calibrated on dev.

Examples:

- minimum BM25 score
- vector similarity threshold
- reranker threshold
- evidence sufficiency threshold
- Laya confidence threshold
- abstention threshold

Never tune these thresholds directly on the locked test set.

---

## 44. Human Evaluation

Maintain a small manually reviewed subset.

Reviewers should score:

- factual correctness
- causal correctness
- evidence support
- relevance
- completeness
- uncertainty handling

Do not ask reviewers to infer hidden model internals.

Measure reviewer agreement where multiple reviewers are available.

---

## 45. Error Taxonomy

Every failed or poor result should be assigned one or more categories:

```text
RETRIEVAL_MISS
WRONG_CONVERSATION
WRONG_TURN
CAUSAL_CHAIN_INCOMPLETE
CONTEXT_TOO_LARGE
CONTEXT_TOO_SMALL
EVIDENCE_REDUNDANT
HALLUCINATION
CONTRADICTION
WRONG_ENTITY
WRONG_TEMPORAL_ORDER
POST_CALL_CONTAMINATION
PARSER_FAILURE
LLM_FAILURE
TIMEOUT
RATE_LIMIT
INSUFFICIENT_EVIDENCE_NOT_DETECTED
FALSE_ABSTENTION
```

Generate an error histogram by system.

---

## 46. Failure Case Archive

Every benchmark run should automatically save representative failures.

```text
artifacts/failures/
├── retrieval_miss/
├── wrong_conversation/
├── hallucination/
├── contamination/
├── causal_incomplete/
└── latency_outlier/
```

Each case should contain:

- question
- expected evidence
- retrieved evidence
- answer
- trace
- metrics
- failure label

This turns benchmarking into an engineering feedback loop.

---

## 47. Regression Tests

Create a fixed regression set containing known failures discovered during development.

At minimum include:

1. The earlier System 1 case where the top chunk was only the closing turns.
2. A case where unrelated insurance conversations dominate retrieval.
3. A case where the causal chain crosses chunk boundaries.
4. A case with repeated customer/business terms.
5. A case with post-call artifacts.
6. A case with insufficient evidence.
7. A case where the final outcome is not explicitly stated.

Every new implementation must run this suite before benchmark publication.

---

## 48. Benchmark Reproducibility Commands

Provide commands such as:

```bash
# Build benchmark
python -m benchmark.datasets.build

# Validate dataset
python -m benchmark.datasets.validate

# Run retrieval-only benchmark
python -m benchmark.runner --stage retrieval

# Run full benchmark
python -m benchmark.runner --systems system1 system2 system3 system4

# Run quality evaluation
python -m benchmark.evaluate --run-id <RUN_ID>

# Generate report
python -m benchmark.report --run-id <RUN_ID>

# Generate plots
python -m benchmark.plots --run-id <RUN_ID>
```

---

## 49. Benchmark Report

Generate both:

```text
reports/<run_id>.md
reports/<run_id>.html
```

The report must contain:

1. Executive summary
2. Experimental setup
3. Dataset statistics
4. Retrieval results
5. Evidence results
6. Task 1 results
7. Task 2 results
8. Latency analysis
9. Token analysis
10. Reliability analysis
11. Cost analysis
12. Robustness analysis
13. Ablations
14. Error analysis
15. Pareto analysis
16. Representative successes
17. Representative failures
18. Limitations
19. Reproducibility information

Do not write unsupported conclusions.

---

## 50. Required Visualizations

Generate at least:

1. Recall@K by system
2. MRR by system
3. Conversation recall by system
4. Evidence precision/recall
5. Cross-conversation contamination rate
6. p50/p95 latency
7. Latency distribution
8. Retrieval vs generation latency breakdown
9. Tokens/query
10. Context tokens vs quality
11. Output tokens vs quality
12. Quality vs latency scatter
13. Quality vs cost scatter when applicable
14. Robustness degradation
15. Error taxonomy
16. Ablation results
17. Laya impact
18. Pareto frontier

Use confidence intervals where possible.

---

## 51. Benchmark Dashboard

The Streamlit benchmark dashboard should provide:

- run selector
- system selector
- task selector
- metric selector
- question-level drilldown
- latency trace
- retrieved evidence viewer
- transcript IDs
- evidence scores
- generated answer
- judge output
- failure classification
- side-by-side system comparison

A reviewer must be able to inspect a single bad query from aggregate chart → query → evidence → answer → trace.

---

## 52. Side-by-Side Comparison

Provide a comparison view like:

```text
Question

System 1
  evidence
  answer
  latency
  tokens
  grounding

System 2
  evidence
  answer
  latency
  tokens
  grounding

System 3
  evidence
  answer
  latency
  tokens
  grounding

System 4
  evidence
  answer
  latency
  tokens
  grounding
```

The dashboard must not hide failures behind aggregate scores.

---

## 53. Quality-Speed Frontier

The benchmark must explicitly evaluate the trade-off between quality and speed.

Useful derived metrics:

```text
quality_per_second
quality_per_1k_input_tokens
quality_per_1k_total_tokens
recall_per_ms
useful_evidence_per_context_token
```

These are secondary metrics and must not replace the underlying measurements.

---

## 54. Model Matrix

Support multiple LLMs without changing benchmark logic.

Example:

```text
models/
├── bonsai
├── qwen3.5-9b-mlx
├── groq-gpt-oss
└── gemini-flash
```

Run:

```text
System 1 × Model A
System 1 × Model B
System 3 × Model A
System 3 × Model B
```

For a model benchmark, freeze retrieval architecture.

For a RAG architecture benchmark, freeze the model.

Never mix these two experiments in one table without labeling the factor being changed.

---

## 55. Local Model Benchmarking

For Ollama/MLX/local models record:

- model identifier
- quantization
- context size
- temperature
- max output tokens
- GPU/CPU/backend
- memory usage where available
- model load time
- warm generation rate

For Apple Silicon specifically record the backend and unified-memory environment when available.

---

## 56. API Model Benchmarking

For cloud models record:

- provider
- model identifier
- API version if available
- region if relevant
- request timestamp
- usage tokens
- provider latency metadata
- rate-limit headers when permitted
- retries
- error codes

Do not expose API keys in benchmark artifacts.

---

## 57. Caching Rules

Benchmark cache behavior separately.

Two modes:

```text
cache_disabled
cache_enabled
```

Primary architecture comparisons should generally use cache-disabled or cache-controlled measurements so one system does not gain an unfair advantage from stale retrieval/generation cache.

Measure cache hit rate separately.

---

## 58. Warm-Up Rules

Before warm latency measurement:

- initialize model
- initialize index
- initialize tokenizer
- execute defined warm-up requests

Discard warm-up measurements from reported latency unless explicitly studying cold-start behavior.

---

## 59. Hardware Monitoring

For local runs optionally capture:

- RAM usage
- unified memory pressure
- CPU utilization
- GPU utilization where available
- process memory
- disk usage for indexes/models

Do not require platform-specific counters for correctness.

---

## 60. Index Build Benchmark

Measure retrieval-system setup separately:

```text
corpus parsing time
normalization time
index construction time
index serialization time
index load time
index disk size
```

For System 3 this is especially important because vectorless indexing should not be treated as free.

For System 1 separately report:

```text
embedding generation time
vector index build time
vector index size
```

---

## 61. Query Throughput

Report:

```text
queries/sec
queries/minute
successful queries/sec
```

Separate:

- retrieval throughput
- end-to-end throughput

For APIs, respect provider rate limits.

---

## 62. Benchmark Quality Gates

A benchmark run is valid only if:

- dataset validation passes
- schema validation passes
- all systems receive the same question set
- configuration is recorded
- failures are retained
- no test leakage is detected
- at least the required metrics are present
- environment information is captured

If a gate fails, mark the run invalid rather than silently producing a partial leaderboard.

---

## 63. Missing Metrics

Never substitute zero for unavailable metrics.

Use:

```json
{"value": null, "status": "unavailable", "reason": "provider_did_not_expose_ttft"}
```

This prevents unavailable measurements from looking like excellent measurements.

---

## 64. Benchmark Configuration Example

```yaml
benchmark:
  name: causal-rag-v1
  split: test
  seed: 42
  max_questions: 500

retrieval:
  recall_k: [1, 3, 5, 10]

context:
  budgets: [500, 1000, 1500, 2500, 4000]

llm:
  temperature: 0
  max_output_tokens: 800

latency:
  warmup_requests: 3
  repetitions: 1

quality:
  llm_judge: true
  human_subset: 100

systems:
  - system1
  - system2
  - system3
  - system4
```

---

## 65. Recommended First Benchmark

Before running the full 8k dataset, run:

```text
50 dev questions
↓
4 systems
↓
1 fixed LLM
↓
full trace
↓
automatic metrics
↓
manual inspection
```

Then run 100–500 test questions.

Only after the benchmark is stable should the full evaluation be attempted.

---

## 66. Benchmark Phases

### Phase A — Retrieval-only

No LLM.

Measure:

- Recall@K
- MRR
- nDCG
- contamination
- retrieval latency

### Phase B — Evidence-to-LLM

Measure:

- answer quality
- grounding
- generation latency
- token usage

### Phase C — Robustness

Run perturbation and adversarial sets.

### Phase D — Ablation

Remove components individually.

### Phase E — Scale

Run increasing dataset sizes.

### Phase F — Final report

Generate all artifacts and freeze the run manifest.

---

## 67. Benchmarking the Earlier Observed Failure

The benchmark must include a query whose naive top-k retrieval returns closing turns such as:

```text
Customer: definitely thanks again
Agent: you're welcome
Customer: goodbye
```

The benchmark should verify whether the architecture expands to earlier turns containing the actual causal event.

This is a regression requirement, not an optional example.

---

## 68. Benchmarking Cross-Conversation Failure

Include questions where multiple conversations contain terms such as:

- insurance
- gold plan
- silver plan
- refund
- promotion
- deductible
- coverage

The benchmark must determine whether the system retrieves the correct conversation rather than assembling an answer from several unrelated conversations.

---

## 69. Quality Definition

Do not define quality as one generic LLM score.

The benchmark should present a metric vector:

```text
retrieval recall
conversation recall
evidence precision
evidence sufficiency
causal correctness
grounding
relevance
abstention quality
latency
tokens
failure rate
```

An aggregate score may be created only in a separately configured experiment with transparent weights.

---

## 70. Final Deliverables

A completed benchmark implementation must produce:

```text
benchmark_suite/
├── AGENTS.md
├── benchmark/
├── configs/
├── datasets/
├── runs/
├── reports/
├── plots/
├── scripts/
├── tests/
├── requirements.txt
├── README.md
└── Makefile
```

And each completed run must produce:

```text
manifest
raw results
traces
retrieval metrics
quality metrics
latency metrics
token metrics
failure archive
plots
markdown report
HTML report
```

---

## 71. Definition of Done

The benchmark suite is complete when:

- all four systems implement the same runner interface
- the same questions are paired across systems
- retrieval is evaluated independently
- Task 1 and Task 2 are evaluated separately
- evidence grounding is measured
- cross-conversation contamination is measured
- p50/p95/p99 latency is measured
- cold and warm latency are separated
- TTFT and generation throughput are measured where available
- input/context/output tokens are recorded
- context efficiency is measured
- failures are retained
- robustness tests exist
- ablations exist
- Laya impact can be measured separately
- local and cloud models can be benchmarked without changing the evaluator
- benchmark runs are reproducible
- reports and plots are generated automatically
- no metric silently substitutes unavailable values with zero
- regression cases from previous development are included

---

## 72. Engineering Rule

The benchmark suite must answer four separate questions:

1. **Did retrieval find the right evidence?**
2. **Did the LLM reason correctly from that evidence?**
3. **Did the system remain grounded and avoid contamination/hallucination?**
4. **How much time and compute/token budget did it require?**

Never allow one aggregate number to hide failures in any of these dimensions.

---

## 73. Final Benchmark Philosophy

The purpose of this suite is not to manufacture a winner.

The purpose is to produce a defensible engineering comparison of four RAG architectures under controlled conditions.

Every reported improvement must be attributable to a documented architectural or configuration change.

Every latency claim must identify its measurement boundary.

Every quality claim must identify its evaluation method.

Every benchmark result must remain traceable from aggregate metric → individual query → retrieved evidence → model output → timing trace.
