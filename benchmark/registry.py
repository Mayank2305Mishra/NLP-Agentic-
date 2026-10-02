from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_config(system: str) -> dict[str, Any]:
    path = ROOT / "benchmark" / "configs" / f"{system}.yaml"
    if not path.exists():
        raise ValueError(f"unknown system {system!r}; expected config at {path}")
    return json.loads(path.read_text())  # JSON is valid YAML; avoids a runner dependency.


def answer(system: str, question: dict[str, Any], timeout_s: float = 300) -> dict[str, Any]:
    config = load_config(system)
    command = config.get("command") or []
    if not command:
        raise RuntimeError(f"{system} is not configured; set its command in benchmark/configs/{system}.yaml")
    payload = {"query": question["query"], "task": question["task"], "config": config.get("settings", {})}
    proc = subprocess.run(command, input=json.dumps(payload), text=True, capture_output=True, timeout=timeout_s)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or f"adapter exited {proc.returncode}")
    result = json.loads(proc.stdout)
    required = {"answer", "evidence", "trace", "usage", "system_metadata"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"{system} adapter omitted required fields: {sorted(missing)}")
    return result
