from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from system3.config import load_config  # noqa: E402
from system3.pipeline import Pipeline  # noqa: E402

st.set_page_config(page_title="System 3 — Lexical Causal RAG", layout="wide")


@st.cache_resource(show_spinner=False)
def get_pipeline() -> Pipeline:
    config = load_config()
    return Pipeline(config)


def render_answer(result) -> None:
    answer = result.answer or {}
    st.subheader("Answer")
    if answer.get("insufficient_evidence"):
        st.warning("Insufficient evidence — no causal answer was synthesized.")
        return
    if result.task == "task1":
        st.metric("Confidence", answer.get("confidence", "n/a"))
        for idx, item in enumerate(answer.get("causal_rationales") or [], 1):
            with st.container(border=True):
                st.markdown(f"**Rationale {idx}** · strength: `{item.get('strength')}`")
                st.markdown(f"- **Cause:** {item.get('cause')}")
                st.markdown(f"- **Effect:** {item.get('effect')}")
                mechanism = item.get("mechanism")
                st.markdown(f"- **Mechanism:** {mechanism if mechanism else '_not stated_'}")
                st.markdown(f"- **Evidence turns:** {item.get('evidence_turns')}")
    else:
        st.markdown(answer.get("summary") or "_no summary_")
        chain = answer.get("causal_chain") or []
        if chain:
            st.markdown("**Causal chain**")
            st.table(
                [
                    {
                        "step": s.get("step"),
                        "role": s.get("role"),
                        "event": s.get("event"),
                        "turns": s.get("evidence_turns"),
                    }
                    for s in chain
                ]
            )
        cols = st.columns(3)
        cols[0].markdown("**Key causes**")
        cols[0].write(answer.get("key_causes") or [])
        cols[1].markdown("**Effects**")
        cols[1].write(answer.get("effects") or [])
        cols[2].markdown("**Uncertainties**")
        cols[2].write(answer.get("uncertainties") or [])
        st.metric("Confidence", answer.get("confidence", "n/a"))


def render_evidence(result) -> None:
    st.subheader("Evidence")
    if not result.evidence:
        st.info("No evidence conversations were selected.")
        return
    for conv in result.evidence:
        turns = conv["turns"]
        span = f"Turns {turns[0]['turn_id']}–{turns[-1]['turn_id']}" if turns else "n/a"
        with st.container(border=True):
            st.markdown(
                f"**Conversation:** `{conv['transcript_id']}` · {span} · "
                f"conversation score **{conv['conversation_score']:.3f}**"
            )
            for turn in turns:
                stage = "seed" if turn["seed"] else "neighbor"
                st.markdown(
                    f"**T{turn['turn_id']:03d} | {turn['speaker']}** "
                    f"({stage}, score {turn['score']:.3f})"
                )
                st.write(turn["text"])


def render_latency(result) -> None:
    st.subheader("Latency")
    trace = result.trace
    retrieval_total = sum(
        trace.get(k) or 0.0
        for k in (
            "query_normalization_ms",
            "metadata_filter_ms",
            "fts_retrieval_ms",
            "conversation_ranking_ms",
            "turn_ranking_ms",
            "neighbor_expansion_ms",
            "evidence_pack_ms",
            "evidence_gate_ms",
        )
    )
    cols = st.columns(4)
    cols[0].metric("Retrieval", f"{retrieval_total:.1f} ms")
    cols[1].metric("Generation", f"{trace.get('generation_ms') or 0:.1f} ms")
    cols[2].metric("TTFT", f"{trace.get('generation_ttft_ms') or 0:.1f} ms")
    cols[3].metric("Total", f"{trace.get('total_ms') or 0:.1f} ms")
    with st.expander("Full stage trace"):
        st.json({k: v for k, v in trace.items()})
    if result.tokens:
        with st.expander("Tokens"):
            st.json(result.tokens)


def render_debug(result) -> None:
    with st.expander("Why this evidence? (score components)"):
        for cand in result.candidates[:6]:
            st.markdown(
                f"**`{cand['transcript_id']}`** score {cand['score']:.3f} "
                f"(bm25 raw {cand['bm25_raw']:.3f})"
            )
            st.json(cand["components"])
    with st.expander("Query plan"):
        st.json(result.query_plan)
    with st.expander("Decisions (typed gate trace)"):
        st.json(result.answer_gate.get("decisions", []))
    with st.expander("Grounding report"):
        st.json(result.grounding)
    with st.expander("Raw model output"):
        st.code(result.raw_output or "(not invoked)", language="json")
    if result.errors:
        st.error(result.errors)


def main() -> None:
    st.title("System 3 — Vectorless Lexical Causal RAG")
    st.caption(
        "SQLite FTS5/BM25 retrieval · conversation-aware evidence · "
        f"model: `{get_pipeline().config.generation.model}` "
        f"({get_pipeline().config.generation.provider})"
    )

    question = st.text_area(
        "Question", height=100, placeholder="Why was the customer unhappy and what did the agent do?"
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        task = st.selectbox("Task", ["auto", "task1", "task2"], index=0)
    with col2:
        ablation = st.selectbox(
            "Retrieval mode",
            ["full", "pure_bm25", "bm25_metadata", "bm25_causal", "bm25_expansion"],
            index=0,
        )

    if st.button("Run", type="primary") and question.strip():
        pipeline = get_pipeline()
        with st.spinner("Retrieving and synthesizing..."):
            # Ablation is request-scoped: the shared pipeline is never mutated.
            result = pipeline.ask(
                question,
                task=None if task == "auto" else task,
                ablation=ablation,
            )

        gate_col, answer_col, model_col = st.columns(3)
        gate_col.metric("Evidence gate", result.evidence_gate.get("decision"))
        answer_col.metric("Answer gate", result.answer_gate.get("decision"))
        model_col.metric("Model", f"{result.provider}/{result.model}")

        render_answer(result)
        render_evidence(result)
        render_latency(result)
        render_debug(result)

        st.session_state["last_result"] = result.to_dict()
    elif "last_result" in st.session_state:
        st.divider()
        st.caption("Previous result (JSON export)")
        st.json(
            {k: v for k, v in st.session_state["last_result"].items() if k != "raw_output"}
        )


if __name__ == "__main__":
    main()
