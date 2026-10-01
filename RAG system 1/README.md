# Production System 1 — Traditional Dense Vector RAG

Local-first baseline for **Causal Rationale Extraction and Synthesis from Conversational Data**.
See `AGENTS.md` (the engineering contract) for the full spec.

- **System:** Traditional Dense Vector RAG · **LLM:** Ollama `qwen3.5:4b-mlx` (local) · **System-1:** Laya (local) · **Index:** FAISS
- One canonical pipeline (`app/main.py:QueryService`) shared by Streamlit, CLI, benchmark, and tests.
- Source dataset (`final_transcript.json` / `final_transcripts.json`) is **read-only**; all artifacts go under `generated/` (gitignored).

## Quickstart

```bash
pip install -e ".[all]"        # or: pip install -e .
cp .env.example .env           # optional; defaults work

python -m scripts.validate_dataset
python -m scripts.build_index --progress
python -m scripts.query --health
python -m scripts.query --task task1 --query "Why did customers request urgent booking changes?"
make ui                        # Streamlit on 127.0.0.1
python -m pytest -q
```

## Layout

`app/` (service layer) · `prompts/` (versioned `task1_v1.txt`, `task2_v1.txt`) ·
`scripts/` (thin CLI over `QueryService`) · `tests/` · `generated/` (index, traces, benchmarks).

## Generation (local Ollama)

The generator is your local Ollama model `qwen3.5:4b-mlx` — the sole LLM in
the system. Only `app/generation/ollama_client.py` knows the endpoint contract
(native `POST /api/chat`, `format: json`, non-streaming, thinking off).
Nothing leaves the laptop. If Ollama is down or the model isn't pulled, the
UI/CLI surface a typed error — no fake answers.

```bash
ollama pull qwen3.5:4b-mlx   # if not already pulled
python -m scripts.query --health
python -m scripts.query --task task1 --query "Why did customers request urgent booking changes?"
```

> NOTE: prior backends (Bonsai stub, Grok, Groq, Gemini) were removed per
> explicit instruction; retrieval, System-1, validation, and evaluation
> layers are unchanged.

## Notes

- **Embeddings:** default `EMBEDDING_BACKEND=hashing` — deterministic, local, no download (sklearn HashingVectorizer, 768-d, L2-normalized). Set `EMBEDDING_BACKEND=sentence-transformers` + `EMBEDDING_MODEL=<name>` for a transformer backend (requires the `embeddings-st` extra). The active model/dim is recorded in the index manifest and every benchmark.
- **Vector store:** `VECTOR_STORE_TYPE=auto` prefers FAISS, falls back to a numpy brute-force store behind the same `VectorStore` protocol (explicitly recorded, never silent architecture switching in results).
- **Generator:** only `app/generation/ollama_client.py` knows its endpoint contract. On failure the UI/CLI surface a typed error — no fake answers.
- **Quality figures are engineering proxies, NOT official hackathon scores.**
