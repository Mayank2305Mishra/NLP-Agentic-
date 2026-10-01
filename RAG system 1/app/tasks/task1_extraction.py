"""Task 1: causal rationale extraction (independently testable)."""
from __future__ import annotations

from app.config import Settings
from app.domain.errors import GroundingValidationError
from app.domain.models import EvidencePack, Task1Result
from app.generation.base import GeneratorClient
from app.generation.parser import parse_task1, validate_task1_grounding
from app.generation.prompts import build_prompt_for_pack, prompt_version_for
from app.observability.trace import RequestTrace, Timer
from app.tasks.base import generate_with_repair


def run_task1(*, settings: Settings, client: GeneratorClient, query_text: str,
              pack: EvidencePack, trace: RequestTrace) -> tuple[Task1Result, dict]:
    version = prompt_version_for(settings, "task1")
    with Timer(trace, "prompt_build"):
        prompt, prompt_file = build_prompt_for_pack(
            task="task1", version=version, query_text=query_text, pack=pack)
    parsed, resp, grounding, vfail = generate_with_repair(
        client=client, prompt=prompt, parse_fn=parse_task1,
        ground_fn=validate_task1_grounding, pack=pack,
        request_id=trace.request_id, trace=trace)
    with Timer(trace, "grounding_validation"):
        if parsed.insufficient_evidence and parsed.causal_rationales:
            raise GroundingValidationError("insufficient_evidence=true with non-empty rationales")
    info = {"prompt_version": f"task1_{version}", "prompt_file": prompt_file,
            "model": resp.model, "http_status": resp.http_status,
            "generation_latency_ms": resp.latency_ms, "retries": resp.retries,
            "validation_failures": vfail, "grounding": grounding,
            "usage": getattr(resp, "usage", None)}
    return parsed, info
