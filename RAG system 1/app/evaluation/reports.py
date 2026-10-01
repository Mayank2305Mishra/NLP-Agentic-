"""Human-readable benchmark report (markdown) from a machine-readable report."""
from __future__ import annotations


def render_markdown(report: dict) -> str:
    lat = report.get("latency", {})
    ret = report.get("retrieval_latency", {})
    gen = report.get("generation_latency", {})
    q = report.get("quality_proxies", {})
    lines = [
        f"# Benchmark report — {report.get('system')} (`{report.get('run_id')}`)",
        "",
        f"- Timestamp: {report.get('timestamp')}",
        f"- Task: {report.get('task')} | Split: {report.get('split')}",
        f"- Git: {report.get('git_commit')} | Source sha: {str(report.get('source_sha256'))[:12]}",
        f"- Embedding: {report.get('embedding_model')} dim={report.get('embedding_dimension')} "
        f"index={report.get('index_type')} chunks={report.get('num_chunks')}",
        f"- Generator: {report.get('generation', {}).get('backend')} "
        f"{report.get('generation', {}).get('model')} "
        f"temp={report.get('generation', {}).get('temperature')}",
        "",
        "## Latency (ms, warm)",
        f"- total: mean {lat.get('mean')} p50 {lat.get('p50')} p90 {lat.get('p90')} "
        f"p95 {lat.get('p95')} max {lat.get('max')} n={lat.get('count')}",
        f"- retrieval: mean {ret.get('mean')} p95 {ret.get('p95')}",
        f"- generation: mean {gen.get('mean')} p95 {gen.get('p95')}",
        "",
        "## Quality proxies (NOT official scores)",
        f"- success {q.get('success_rate')} | grounding {q.get('grounding_rate')} | "
        f"citation-valid {q.get('citation_validity_rate')} | "
        f"evidence-sufficient {q.get('evidence_sufficiency_rate')}",
        "",
        "> Do not present proxy metrics as the official hackathon score.",
    ]
    return "\n".join(lines)
