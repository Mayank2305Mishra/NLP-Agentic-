"""Report writers: machine-readable JSON + human-readable Markdown (§33, §34)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, report: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return target


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def markdown_summary(report: dict[str, Any]) -> str:
    lines: list[str] = []
    if report.get("kind") == "matrix":
        lines.append("# System 4 — Variant Comparison\n")
        rows = report.get("comparison", {}).get("rows", [])
        if rows:
            headers = list(rows[0].keys())
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join("---" for _ in headers) + "|")
            for row in rows:
                lines.append("| " + " | ".join(_fmt(row.get(h)) for h in headers) + " |")
        lines.append("")
        return "\n".join(lines)

    lines.append("# System 4 — Benchmark Report\n")
    lines.append(
        f"- system: `{report.get('system_version')}`  \n"
        f"- variant: `{report.get('variant')}`  \n"
        f"- questions: {report.get('questions')} × repeats {report.get('repeats')}  \n"
        f"- wall: {report.get('wall_ms')} ms\n"
    )
    retrieval = report.get("retrieval", {})
    if retrieval:
        lines.append("## Retrieval (§31)\n")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for key, value in retrieval.items():
            lines.append(f"| {key} | {_fmt(value)} |")
        lines.append("")
    quality = report.get("quality", {})
    if quality:
        lines.append("## Generation quality (§31)\n")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for key, value in quality.items():
            lines.append(f"| {key} | {_fmt(value)} |")
        lines.append("")
    latency = report.get("latency", {})
    if latency:
        lines.append("## Latency by stage (§32)\n")
        lines.append("| stage | mean ms | p50 | p95 | max |")
        lines.append("|---|---|---|---|---|")
        for stage, stats in latency.get("stages", {}).items():
            lines.append(
                f"| {stage} | {_fmt(stats.get('mean'))} | {_fmt(stats.get('p50'))} "
                f"| {_fmt(stats.get('p95'))} | {_fmt(stats.get('max'))} |"
            )
        lines.append("")
        lines.append(
            f"- avg context tokens: {_fmt(latency.get('context_tokens'))}  \n"
            f"- avg output tokens: {_fmt(latency.get('output_tokens'))}  \n"
            f"- generation tokens/s: {_fmt(latency.get('generation_tokens_per_second'))}  \n"
            f"- requests with stage errors: {_fmt(latency.get('error_requests'))}\n"
        )
    return "\n".join(lines)


def write(path_base: str | Path, report: dict[str, Any]) -> tuple[Path, Path]:
    """Write ``<path>.json`` and ``<path>.md``."""
    base = Path(path_base)
    json_path = write_json(base.with_suffix(".json"), report)
    md_path = base.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown_summary(report), encoding="utf-8")
    return json_path, md_path
