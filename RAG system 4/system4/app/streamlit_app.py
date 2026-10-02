"""System 4 Streamlit UI (AGENTS_SYSTEM_4.md §45).

Exposes query, task, final answer, causal rationale fields, evidence/source
IDs, confidence, insufficient-evidence status, latency breakdown, and
retrieval/reranker statistics. Advanced debug mode adds candidate
conversations, reranker scores, Laya decisions, evidence selection reasons,
and compression statistics. Raw transcript text is not shown by default (§41).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Streamlit runs this file as a script, so the repository root is not guaranteed
# to be on sys.path when launched with `streamlit run system4/app/streamlit_app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from system4.ingestion.artifacts import load_artifact
from system4.pipeline import QualityRAG
from system4.schema import SYSTEM_VERSION

st.set_page_config(page_title="System 4 — Quality RAG", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading index…")
def app(source_hint: str):
    artifact = load_artifact(source_hint, expected_version=SYSTEM_VERSION)
    return QualityRAG(
        artifact["index"],
        source_hash=artifact.get("source_hash", ""),
    ), artifact


INDEX_PATH = os.getenv("SYSTEM4_INDEX", "artifacts/index.pkl")
try:
    rag, artifact = app(INDEX_PATH)
except Exception as exc:  # missing/stale index — tell the operator exactly what to run
    st.error(
        f"Index unavailable: {exc}\n\n"
        f"Build it with:\n\n"
        f"`python scripts/build_index.py --source final_transcripts.json --output {INDEX_PATH}`"
    )
    st.stop()

st.title("System 4 — Agentic / Quality RAG")
st.caption(
    "Broad retrieval → consistency gate → reranker → causal evidence → Laya → "
    "compression → grounded generation → answer gate"
)

with st.sidebar:
    st.header("Settings")
    task = st.selectbox("Task", ["task1", "task2", "auto"], index=0,
                        help="task1: causal rationales · task2: interaction summary · auto: routed by Laya")
    detail = st.select_slider("Answer detail", options=["concise", "detailed", "thorough"], value="detailed")
    cross = st.checkbox("Cross-conversation analysis", value=False,
                        help="Off by default: single-interaction answers keep exactly one source (§23).")
    debug = st.checkbox("Advanced debug mode", value=False,
                        help="Candidates, reranker scores, Laya decisions, selection reasons (§45).")
    st.divider()
    st.caption(
        f"Index: `{INDEX_PATH}`  \n"
        f"Conversations: {artifact.get('conversation_count', '—')}  \n"
        f"Version: `{SYSTEM_VERSION}`  \n"
        f"Source hash: `{str(artifact.get('source_hash', ''))[:12]}…`  \n"
        f"LLM: `{type(rag.llm).__name__ if rag.llm else 'deterministic fallback (set OLLAMA_MODEL)'}`"
    )

query = st.text_area("Question about a conversation", height=120,
                     placeholder="Why was the refund approved after the cancellation?")
run = st.button("Analyze", type="primary")

if run and query.strip():
    with st.spinner("Running the evidence pipeline…"):
        result = rag.ask(query, task, cross_conversation=cross, detail=detail, debug=debug)
    answer, trace = result.answer, result.trace

    # --- headline status ----------------------------------------------------
    gate = trace.get("answer_gate", "—")
    grounding = trace.get("grounding_status", "—")
    color = {"ANSWER": "green", "ANSWER_WITH_UNCERTAINTY": "orange", "ABSTAIN": "red"}.get(gate, "gray")
    left, mid, right = st.columns(3)
    left.metric("Answer gate", gate)
    mid.metric("Grounding", f"{grounding} ({trace.get('grounding_ratio', 0):.2f})" if isinstance(trace.get("grounding_ratio"), (int, float)) else grounding)
    right.metric("Total latency", f"{trace.get('total_ms', 0):.0f} ms")
    st.markdown(f":{color}[{'Answer delivered' if gate != 'ABSTAIN' else 'Abstained — insufficient or unsupported evidence'}]"
                + (" · `insufficient_evidence`" if answer.get("insufficient_evidence") else ""))

    if answer.get("reason"):
        st.info(f"Reason: {answer['reason']}")

    # --- final answer -------------------------------------------------------
    st.subheader("Final answer")
    if answer.get("task") == "task1":
        rationales = answer.get("causal_rationales") or []
        if not rationales:
            st.write("_No causal rationale passed the answer gate._")
        for index, rationale in enumerate(rationales, 1):
            with st.expander(
                f"Rationale {index} — {rationale.get('strength', '?')} "
                f"(turns {rationale.get('evidence_turns', [])})",
                expanded=True,
            ):
                st.write(f"**Cause:** {rationale.get('cause', '')}")
                st.write(f"**Effect:** {rationale.get('effect', '')}")
                mechanism = rationale.get("mechanism")
                st.write(f"**Mechanism:** {mechanism if mechanism else '_not established by evidence (not inferred)_'}")
    else:
        st.write(answer.get("summary") or "_No summary passed the answer gate._")
        for label, key in [("Key causes", "key_causes"), ("Effects", "effects"), ("Mechanisms", "mechanisms")]:
            values = answer.get(key) or []
            if values:
                st.write(f"**{label}:**")
                for value in values:
                    st.write(f"- {value}")
        uncertainties = answer.get("uncertainties") or []
        if uncertainties:
            st.warning("**Uncertainties:**\n\n" + "\n".join(f"- {u}" for u in uncertainties))

    confidence = answer.get("confidence")
    sources = answer.get("source_transcripts") or trace.get("selected_transcript_ids") or []
    st.caption(
        f"Confidence: {confidence if confidence is not None else '—'} · "
        f"Sources: `{', '.join(sources) or '—'}` · "
        f"Generation: `{trace.get('generation_mode', '—')}` · "
        f"Cache hit: `{trace.get('cache_hit', False)}`"
    )

    # --- evidence -----------------------------------------------------------
    st.subheader("Evidence")
    if not result.evidence:
        st.write("_No evidence selected._")
    else:
        rows = [
            {
                "source": item.turn.transcript_id,
                "turn": item.turn.turn_id,
                "speaker": item.turn.speaker,
                "score": round(item.score, 3),
                "roles": ", ".join(item.roles),
                "reason": item.reason,
                "stage": item.retrieval_stage,
            }
            for item in result.evidence
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # --- latency breakdown --------------------------------------------------
    st.subheader("Latency breakdown (stage-by-stage)")
    stages = {
        key: value for key, value in trace.items()
        if key.endswith("_ms") and isinstance(value, (int, float))
    }
    latency_rows = [
        {"stage": stage, "ms": round(value, 2)}
        for stage, value in sorted(stages.items(), key=lambda kv: -kv[1])
    ]
    left, right = st.columns(2)
    left.dataframe(latency_rows, use_container_width=True, hide_index=True)
    right.write(
        f"- **Context tokens:** {trace.get('context_tokens', '—')}\n"
        f"- **Input tokens:** {int(trace.get('input_tokens', 0) or 0)}\n"
        f"- **Output tokens:** {int(trace.get('output_tokens', 0) or 0)}\n"
        f"- **TTFT:** {trace.get('llm_ttft_ms', 0)} ms\n"
        f"- **Generation tokens/s:** {trace.get('generation_tokens_per_second', 0)}\n"
        f"- **Request id:** `{trace.get('request_id', '')[:12]}…`"
    )

    # --- retrieval / reranker statistics ------------------------------------
    st.subheader("Retrieval & reranker statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Candidates", trace.get("candidate_conversations", 0))
    col2.metric("After gate", trace.get("gated_conversations", 0))
    col3.metric("Reranked", trace.get("reranked_conversations", 0))
    col4.metric("Selected turns", len(trace.get("selected_turns") or []))
    st.caption(
        f"Reranker: `{trace.get('reranker', {}).get('name', '—')}` "
        f"({trace.get('reranker', {}).get('version', '—')}) · "
        f"Laya: `{trace.get('decision_policy', '—')}` → "
        f"`{'/'.join(trace.get('laya_decisions') or ['-'])}` / "
        f"`{'/'.join(trace.get('laya_answer_gates') or ['-'])}` — {trace.get('laya_reason', '')} · "
        f"Compression: `{trace.get('compression_level', '—')}`"
    )

    if trace.get("errors"):
        st.warning("Stage degradations: " + "; ".join(
            f"{e.get('stage')}: {e.get('error')}" for e in trace["errors"]
        ))

    # --- advanced debug mode ------------------------------------------------
    if debug and "debug" in trace:
        st.subheader("Advanced debug")
        dbg = trace["debug"]
        st.markdown("**Candidate conversations & scores**")
        st.dataframe(dbg.get("candidates", []), use_container_width=True, hide_index=True)
        st.markdown("**Laya decisions**")
        st.json(dbg.get("laya", []))
        st.markdown("**Evidence selection reasons**")
        st.dataframe(dbg.get("selection", []), use_container_width=True, hide_index=True)
        left, right = st.columns(2)
        with left:
            st.markdown("**Evidence features (§19)**")
            st.json(dbg.get("features", {}))
        with right:
            st.markdown("**Compression statistics (§21)**")
            st.json(trace.get("compression", {}))
        st.markdown("**Consistency gate breakdown (§10)**")
        st.json(dbg.get("consistency_breakdown", {}))

elif run:
    st.warning("Enter a question first.")
