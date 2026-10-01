"""Canonical Pydantic domain models — the internal application contracts."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Turn(BaseModel):
    speaker: str = "Unknown"
    text: str = ""


class TranscriptRecord(BaseModel):
    transcript_id: str
    time_of_interaction: str | None = None
    domain: str | None = None
    intent: str | None = None
    reason_for_call: str | None = None
    conversation: list[Turn] = Field(default_factory=list)


class NormalizedTurn(BaseModel):
    index: int
    speaker: str
    text: str  # verbatim original text
    retrieval_text: str  # normalized representation for embedding


class NormalizedTranscript(BaseModel):
    transcript_id: str
    domain: str | None = None
    intent: str | None = None
    reason_for_call: str | None = None
    turns: list[NormalizedTurn] = Field(default_factory=list)
    skipped_empty_turns: int = 0
    speaker_anomaly: bool = False  # single-speaker or unusual labels
    source_hash: str = ""


class Chunk(BaseModel):
    chunk_id: str  # {transcript_id}:chunk:{index}
    transcript_id: str
    chunk_index: int
    turn_start: int
    turn_end: int
    speakers: list[str] = Field(default_factory=list)
    domain: str | None = None
    intent: str | None = None
    reason_for_call: str | None = None
    text: str  # raw evidence text (verbatim, with metadata header)
    normalized_text: str  # retrieval representation
    source_hash: str = ""


class RetrievalQuery(BaseModel):
    text: str
    task: Literal["task1", "task2"] = "task1"
    domain: str | None = None
    intent: str | None = None
    top_k: int = 8


class EvidenceChunk(BaseModel):
    chunk_id: str
    transcript_id: str
    text: str
    score: float
    rank: int
    metadata: dict = Field(default_factory=dict)
    turn_start: int = 0
    turn_end: int = 0


class EvidencePack(BaseModel):
    chunks: list[EvidenceChunk] = Field(default_factory=list)
    truncated: bool = False
    context_tokens: int = 0
    dropped_chunk_ids: list[str] = Field(default_factory=list)


# --- Task outputs ---

class Task1Evidence(BaseModel):
    source_id: str
    turn_start: int = 0
    turn_end: int = 0
    quote: str = ""


class CausalRationale(BaseModel):
    cause: str
    effect: str
    mechanism: str = ""
    evidence: list[Task1Evidence] = Field(default_factory=list)
    strength: Literal["explicit", "supported", "uncertain"] = "supported"


class Task1Result(BaseModel):
    task: Literal["task1"] = "task1"
    causal_rationales: list[CausalRationale] = Field(default_factory=list)
    confidence: float = 0.0
    insufficient_evidence: bool = False


class Task2Result(BaseModel):
    task: Literal["task2"] = "task2"
    synthesis: str = ""
    key_causes: list[str] = Field(default_factory=list)
    business_effects: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class IndexManifest(BaseModel):
    source_file: str
    source_sha256: str
    embedding_model: str
    embedding_dimension: int
    embedding_normalize: bool = True
    embedding_backend: str = "hashing"
    chunking_config_hash: str
    chunking_config: dict = Field(default_factory=dict)
    index_type: str = "faiss"
    created_at: str = ""
    num_transcripts: int = 0
    num_chunks: int = 0
    app_version: str = "1.0.0"
