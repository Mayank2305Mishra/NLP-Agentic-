# System 4 — Agentic / Quality RAG

Production-grade System 4 of the conversational causal-rationale benchmark: a
multi-stage retrieval-and-evidence pipeline that maximizes evidence quality and
causal correctness — broad candidate retrieval → conversation consistency gate →
swappable reranker → causal turn selection → local expansion → typed Laya
decisions → compression → provenance-preserving evidence pack → structured LLM
generation → grounding validation → final answer gate (answer **or abstain**).

The system is **local-first** (stdlib-only core), **provenance-preserving**
(every evidence item keeps transcript id, turn, stage, score, and reason), and
**observable** (per-stage latency, typed decisions, structured traces).

Full specification: [`AGENTS_SYSTEM_4.md`](AGENTS_SYSTEM_4.md).

## Quickstart

```bash
# 1. Build the deterministic index (validated by content hash + version)
python scripts/build_index.py --source final_transcripts.json --output artifacts/index.pkl

# 2. Generate the fixed benchmark question set (seeded, reproducible)
python scripts/build_questions.py --source final_transcripts.json --output benchmarks/questions.json

# 3. Run the benchmark (add --matrix to compare reranker/compression variants)
python scripts/run_benchmark.py --index artifacts/index.pkl \
    --questions benchmarks/questions.json --output artifacts/benchmark --repeats 1

# 4. Interactive UI
streamlit run system4/app/streamlit_app.py

# 5. Inspect structured request traces (per-stage latency, Laya, errors)
python scripts/inspect_trace.py --last 20 --aggregate
```

Optional generation with a local Ollama model (otherwise a deterministic,
evidence-cited fallback answer is produced):

```bash
cp .env.example .env      # set OLLAMA_MODEL=qwen3.5-9b-instruct-mlx, etc.
```

## Architecture (canonical pipeline, §52)

```text
QUERY → Query Normalizer → Task Router → Dense + Lexical Retrieval
      → Candidate Pool (RRF) → Conversation Consistency Gate → Reranker
      → Anti-Contamination Gate → Relevant Turns → Laya (accept/expand/reject)
      → Causal Expansion → Compression → Evidence Pack → Prompt → LLM
      → Structured Parser → Grounding Validator → Causal/Consistency Checks
      → Final Answer Gate → ANSWER | ABSTAIN
```

Design priority: `evidence quality > recall after sufficient candidate
generation > context efficiency > latency`, hard-grounded in retrieved
evidence, preferring abstention over unsupported causal claims.

## Repository layout

```text
system4/
├── schema.py            # typed contracts: roles, decisions, provenance types
├── pipeline.py          # canonical stage wiring, timings, fallbacks, cache
├── cache.py             # versioned answer-cache keys (§39)
├── observability.py     # JSONL traces with redaction + rotation (§40, §41)
├── config/              # settings (all thresholds) + hardened prompts
├── ingestion/           # normalization, post-call detection, artifacts
├── retrieval/           # dense, lexical, RRF fusion, consistency gate
├── reranking/           # feature / cross-encoder / passthrough rerankers
├── evidence/            # causal scoring, selection, expansion, compression, packer, feature gate
├── decision/            # Laya facade, deterministic policy, Jev adapter, typed schemas
├── llm/                 # provider-independent protocol + Ollama/Bonsai/Cloud + JSON parser
├── validation/          # grounding, citations, consistency, causal shape, answer gate
├── evaluation/          # retrieval/quality/latency metrics, benchmark runner, reports
└── app/streamlit_app.py # §45 UI
scripts/                 # build_index, run_benchmark, inspect_trace, build_questions
tests/                   # 92 unit + regression + end-to-end tests
benchmarks/questions.json# fixed, seeded benchmark question set (§35)
```

## Key guarantees

| Guarantee | Where |
|---|---|
| No cross-conversation contamination by default | pipeline §23, `validation/consistency.py` |
| Similarity is never used as confidence | rank fusion §9, max-normalized features §12 |
| Mechanisms never invented; missing links → uncertainty | `evidence/causal_scoring.py`, `validation/causal.py` |
| Transcript-scoped citations (`tid:turn`) | `validation/citation.py` |
| Abstention is a first-class outcome | `validation/answer_gate.py` |
| Every stage timed independently (TTFT included) | pipeline §32 |
| Documented fallbacks for every failure | pipeline §43, `decision/policies.py` |
| Traces carry ids/scores/counts — never transcript text | `observability.py` §41 |
| Remote LLM requires explicit opt-in | `llm/__init__.py` (`SYSTEM4_ALLOW_REMOTE=1`) |

## Testing

```bash
python -m pytest tests/ -q     # 92 tests: unit, regression (§47), acceptance (§48)
```

## Benchmarking

```bash
# single configuration
python scripts/run_benchmark.py --index artifacts/index.pkl --questions benchmarks/questions.json

# component matrix: rerankers × compression levels (§36, §37)
python scripts/run_benchmark.py --index artifacts/index.pkl --questions benchmarks/questions.json --matrix
```

Reports are written as JSON + Markdown with retrieval (Recall@k, MRR, nDCG,
contamination), quality (grounding, abstention precision, completeness), and
per-stage latency (mean/p50/p95) — quality vs latency is always reported
together (§33).

## Configuration

Every threshold lives in `system4/config/settings.py` and is fingerprinted into
cache keys. `benchmarks/` values are operating defaults — benchmark-calibrate
them (§12, §19, §35), never tune only on the final evaluation set.
