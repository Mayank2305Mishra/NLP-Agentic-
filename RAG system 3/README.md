# System 3 — Vectorless Causal RAG

System 3 retrieves conversation candidates with SQLite FTS5/BM25, ranks turns with deterministic lexical and causal cues, expands neighboring turns, and sends one conversation-local evidence pack to a configurable LLM. It uses no embeddings or vector database.

## Setup

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[ui,dev]'
```

Set the raw dataset path in `configs/system3.yaml` if needed. The default is `final_transcripts.json` in this directory. Raw data is read only; generated data, index, logs, and benchmark results are ignored by git.

## Build the reproducible sample index

```sh
system3 build-data
system3 sample
system3 build-index --population sample --reset
```

Use `--population full` for all 19,621 conversations. The sample manifest records the source SHA-256, seed, stratification field, and selected transcript IDs.

## Health check

```sh
system3 health          # index + FTS integrity, LLM reachability, log writability
system3 health --fast   # counts only, skips the FTS integrity scan
```

Exits non-zero when the service is degraded, so it can gate a deploy or startup probe.

## Ask and inspect retrieval

```sh
system3 ask 'Why was the customer unhappy and what did the agent do?' --task task1
system3 ask 'Summarize the causal chain in this conversation' --task task2
system3 ask 'Why was the customer unhappy?' --ablation pure_bm25
system3 serve
```

Ablation mode is request-scoped: the shared pipeline is never mutated, so concurrent sessions cannot observe each other's mode.

The default provider is local Ollama. Set `generation.provider` and its URL/model in the YAML config to use Ollama or the local Bonsai endpoint. No transcript is sent to a cloud service by default.

## Evaluate

```sh
system3 benchmark --retrieval-only
system3 benchmark
system3 benchmark --ablation pure_bm25 --retrieval-only
system3 benchmark --workers 4          # parallel questions; identical results
```

The pipeline is thread-safe (thread-local SQLite connections, stateless retriever, locked caches/logs), so `--workers` overlaps generation waits and cuts wall-clock time without changing any retrieval decision.

The fixture has 64 reproducible lexical anchor cases with expected transcript and turn labels. Treat those labels as retrieval checks; they are not human judgments of causal correctness. Benchmark JSON records retrieval quality, stage latency, index metadata, and runtime fingerprint.

## Development

```sh
python -m pytest
```

The package is organized by pipeline stage under `src/system3`: dataset and sampling, SQLite store and indexing, query analysis, retrieval/scoring/evidence, LLM adapters, grounding, observability, health checks, and benchmarking. `app/streamlit_app.py` exposes the same pipeline with evidence and trace details.

LLM clients use a pooled keep-alive session with bounded connection retries (`generation.connect_retries`); retries never fire after tokens have streamed. Benchmark JSON records retrieval quality, stage latency, index metadata, wall-clock throughput, and runtime fingerprint.
