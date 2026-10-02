from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run = ROOT / "benchmarks" / "runs" / args.run_id
    metrics = json.loads((run / "aggregate_metrics.json").read_text())
    lines = [f"# Benchmark report: {args.run_id}", "", "## Retrieval and reliability", "", "| System | Conversation recall | MRR | Evidence precision | Contamination | Failures |", "|---|---:|---:|---:|---:|---:|"]
    for system, value in metrics.items():
        lines.append(f"| {system} | {value['conversation_recall']} | {value['mrr']} | {value['evidence_precision']} | {value['contamination_rate']} | {value['failures']} |")
    target = ROOT / "benchmarks" / "reports" / f"{args.run_id}.md"
    target.parent.mkdir(exist_ok=True)
    markdown = "\n".join(lines) + "\n"
    target.write_text(markdown)
    (target.with_suffix(".html")).write_text(
        "<html><body><pre>" + markdown.replace("&", "&amp;").replace("<", "&lt;") + "</pre></body></html>\n"
    )
    print(target)


if __name__ == "__main__":
    main()
