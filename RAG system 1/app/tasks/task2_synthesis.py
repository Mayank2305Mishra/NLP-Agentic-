"""Task 2: synthesis grounded in Task-1 output + evidence (never from scratch)."""
from __future__ import annotations

from app.config import Settings
from app.domain.models import EvidencePack, Task1Result, Task2Result
from app.generation.base import GeneratorClient
from app.generation.parser import parse_task2, validate_task2_grounding
from app.generation.prompts import build_prompt_for_pack, prompt_version_for
from app.observability.trace import RequestTrace, Timer
from app.tasks.base import generate_with_repair


def run_task2(*, settings: Settings, client: GeneratorClient, query_text: str,
              pack: EvidencePack, task1: Task1Result,
              trace: RequestTrace) -> tuple[Task2Result, dict]:
    version = prompt_version_for(settings, "task2")
    with Timer(trace, "prompt_build"):
        prompt, prompt_file = build_prompt_for_pack(
            task="task2", version=version, query_text=query_text, pack=pack,
            task1_json=task1.model_dump_json())
    parsed, resp, grounding, vfail = generate_with_repair(
        client=client, prompt=prompt, parse_fn=parse_task2,
        ground_fn=validate_task2_grounding, pack=pack,
        request_id=trace.request_id, trace=trace)
    info = {"prompt_version": f"task2_{version}", "prompt_file": prompt_file,
            "model": resp.model, "http_status": resp.http_status,
            "generation_latency_ms": resp.latency_ms, "retries": resp.retries,
            "validation_failures": vfail, "grounding": grounding,
            "usage": getattr(resp, "usage", None)}
    return parsed, info
