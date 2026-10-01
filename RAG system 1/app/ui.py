"""Thin Streamlit UI — no business logic here; all calls go through QueryService.

Pages: Query / Benchmark / System diagnostics.
Binds to localhost by default (see Makefile). Run: make ui
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root

import streamlit as st

from app.config import get_settings
from app.main import SYSTEM_LABEL, SYSTEM1_LABEL, QueryService, llm_label

st.set_page_config(page_title="RAG System 1 — Traditional Dense RAG", layout="wide")

settings = get_settings()
svc = QueryService(settings)

st.caption(f"System: {SYSTEM_LABEL} · LLM: {llm_label(settings.ollama_model)} · System-1: {SYSTEM1_LABEL} · Index: FAISS")

tabs = st.tabs(["Query", "Benchmark", "System diagnostics"])

with tabs[0]:
    st.header("Query")
    col1, col2 = st.columns([3, 1])
    with col1:
        task = st.selectbox("Task", ["task1", "task2"], index=0,
                            help="task1: causal rationale extraction · task2: synthesis")
        query = st.text_area("Question / request", height=120,
                             placeholder="e.g. Why did customers request urgent booking changes?")
    with col2:
        st.subheader("Advanced (benchmark-affecting)")
        top_k = st.slider("top_k", 1, 20, settings.retrieval_top_k)
        domain = st.text_input("domain filter (optional)", value="")
        intent = st.text_input("intent filter (optional)", value="")
    if st.button("Run", type="primary"):
        if not query.strip():
            st.error("Empty query — please provide a question.")
        else:
            with st.spinner("Retrieving evidence and generating…"):
                out = svc.run(task=task, query=query,
                              domain=domain or None, intent=intent or None, top_k=top_k)
            if not out.get("success"):
                err = out.get("error", {})
                st.error(f"{err.get('code')}: {err.get('message')}")
            else:
                st.subheader("Answer")
                st.json(out.get("result"), expanded=False)
                res = out.get("result", {})
                if task == "task1":
                    for r in res.get("causal_rationales", []):
                        st.markdown(f"**Cause:** {r.get('cause')}\n\n**Effect:** {r.get('effect')}"
                                    f"\n\n*Mechanism:* {r.get('mechanism') or '—'} "
                                    f"· *strength:* `{r.get('strength')}`")
                        for ev in r.get("evidence", []):
                            st.code(f"{ev.get('source_id')} turns {ev.get('turn_start')}-{ev.get('turn_end')}: {ev.get('quote')[:300]}")
                else:
                    st.markdown(res.get("synthesis", ""))
                    st.markdown(f"**Key causes:** {res.get('key_causes')}")
                    st.markdown(f"**Effects:** {res.get('business_effects')}")
                    st.markdown(f"**Uncertainties:** {res.get('uncertainties')}")
                st.subheader("Evidence / sources")
                for c in out.get("evidence", []):
                    with st.expander(f"{c['chunk_id']} (score {c['score']})"):
                        st.text(c["text"][:2000])
                        st.json({k: v for k, v in c.items() if k != "text"})
                st.subheader("Latency breakdown")
                st.json(out.get("metrics"))
                with st.expander("Full trace"):
                    st.json(out.get("trace"))

with tabs[1]:
    st.header("Benchmark")
    num_cases = st.number_input("cases", 1, 50, 6)
    if st.button("Run benchmark"):
        from app.evaluation.benchmark import run_benchmark
        from app.evaluation.reports import render_markdown
        with st.spinner("Running benchmark…"):
            report = run_benchmark(settings, task="task1", num_cases=int(num_cases))
        st.markdown(render_markdown(report))
        with st.expander("Machine-readable report"):
            st.json({k: v for k, v in report.items() if k != "cases"})

with tabs[2]:
    st.header("System diagnostics")
    if st.button("Refresh health"):
        from app.observability.health import check_health
        report = check_health(settings)
        for c in report["checks"]:
            (st.success if c["ok"] else st.error)(f"{c['name']}: {c['detail']}")
    st.subheader("Configuration")
    st.json({"source": str(settings.source_path), "generated": str(settings.generated_dir),
             "embedding": f"{settings.embedding_backend}/{settings.embedding_model}",
             "generation": settings.generation_config(), "laya": settings.laya_mode,
             "index_manifest": svc.manifest_info()})
