"""Task base: shared generation-with-one-repair-retry helper."""
from __future__ import annotations

from pathlib import Path

from app.domain.errors import GenerationParseError, GroundingValidationError
from app.generation.base import GeneratorClient
from app.observability.trace import RequestTrace
from app.retrieval.evidence import format_evidence_block

REPAIR_TEMPLATE = (Path(__file__).resolve().parent.parent.parent / "prompts" / "repair_v1.txt").read_text()


def _timed(trace: RequestTrace | None, stage: str, fn, *args, **kwargs):
    """Run fn while recording a trace stage; stage is closed even on failure."""
    if trace is None:
        return fn(*args, **kwargs)
    trace.start(stage)
    try:
        return fn(*args, **kwargs)
    finally:
        trace.end(stage)


def generate_with_repair(*, client: GeneratorClient, prompt: str, parse_fn,
                         ground_fn, pack, request_id: str,
                         trace: RequestTrace | None = None) -> tuple:
    """One generation call; on parse/grounding failure, one compact repair retry.

    The repair prompt carries the evidence pack: a quote-grounding error can
    only be fixed if the model can copy the verbatim source text again.
    Timing is split per stage so generation latency excludes parse/validation.
    """
    evidence_block = format_evidence_block(pack)
    allowed = sorted({c.chunk_id for c in pack.chunks})

    resp = _timed(trace, "generation_request", client.generate, prompt, request_id=request_id)
    validation_failures = 0
    try:
        parsed = _timed(trace, "output_parse", parse_fn, resp.text)
        grounding = _timed(trace, "grounding_validation", ground_fn, parsed, pack)
        return parsed, resp, grounding, validation_failures
    except (GenerationParseError, GroundingValidationError) as e:
        validation_failures = 1
        repair_prompt = (REPAIR_TEMPLATE.replace("{error}", f"{type(e).__name__}: {e}")
                         .replace("{allowed_ids}", ", ".join(allowed))
                         .replace("{evidence}", evidence_block)
                         .replace("{previous}", resp.text[:4000]))
        resp2 = _timed(trace, "generation_request", client.generate, repair_prompt,
                       request_id=request_id)
        parsed2 = _timed(trace, "output_parse", parse_fn, resp2.text)  # raises typed error if still invalid
        grounding2 = _timed(trace, "grounding_validation", ground_fn, parsed2, pack)
        return parsed2, resp2, grounding2, validation_failures
