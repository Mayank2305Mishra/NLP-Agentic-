"""Versioned prompt loading + rendering. Prompts live in prompts/, not in code."""
from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.domain.errors import ConfigurationError
from app.domain.models import EvidencePack
from app.retrieval.evidence import format_evidence_block

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def load_template(task: str, version: str) -> tuple[str, str]:
    name = f"{task}_{version}.txt"
    path = PROMPTS_DIR / name
    if not path.exists():
        raise ConfigurationError(f"Prompt file not found: {name}",
                                 context={"task": task, "version": version})
    return path.read_text(encoding="utf-8"), name


def prompt_version_for(settings: Settings, task: str) -> str:
    return settings.prompt_version_task1 if task == "task1" else settings.prompt_version_task2


def render_prompt(*, task: str, version: str, query_text: str,
                  evidence_block: str, task1_json: str = "") -> tuple[str, str]:
    template, filename = load_template(task, version)
    try:
        prompt = template.format(query=query_text, evidence_block=evidence_block,
                                 task1_json=task1_json)
    except KeyError as e:
        raise ConfigurationError(f"Prompt template {filename} has unknown placeholder: {e}") from e
    if "{" in prompt and ("query" in prompt or "evidence" in prompt):
        pass  # braces from transcript text are fine post-format
    return prompt, filename


def build_prompt_for_pack(*, task: str, version: str, query_text: str,
                          pack: EvidencePack, task1_json: str = "") -> tuple[str, str]:
    return render_prompt(task=task, version=version, query_text=query_text,
                         evidence_block=format_evidence_block(pack), task1_json=task1_json)
