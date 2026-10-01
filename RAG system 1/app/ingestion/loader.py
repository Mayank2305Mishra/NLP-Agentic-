"""Read-only loader for the source transcript dataset.

Never modifies the source file. Supports the canonical
`final_transcript.json` name and the shipped `final_transcripts.json` name.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.domain.errors import DataNotFoundError, DataValidationError


def resolve_source_path(explicit: Path | str | None = None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise DataNotFoundError(f"Source dataset not found: {p}", context={"path": str(p)})
        return p
    root = Path(__file__).resolve().parent.parent.parent
    for name in ("final_transcript.json", "final_transcripts.json"):
        p = root / name
        if p.exists():
            return p
    raise DataNotFoundError(
        "Source dataset not found (looked for final_transcript.json / final_transcripts.json)",
        context={"root": str(root)},
    )


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_raw_records(path: Path | str | None = None) -> tuple[list[dict], str, Path]:
    """Return (records, sha256, resolved_path). Root must be a JSON array
    (or an object wrapper with a list value, e.g. {"transcripts": [...]}).

    Raises DataValidationError with line/offset diagnostics on malformed JSON,
    and on broken UTF-8 (no silent replacement).
    """
    resolved = resolve_source_path(path)
    try:
        raw_bytes = resolved.read_bytes()
    except FileNotFoundError as e:
        raise DataNotFoundError(f"Source dataset not found: {resolved}") from e
    try:
        raw_text = raw_bytes.decode("utf-8")  # strict: fail on broken UTF-8
    except UnicodeDecodeError as e:
        raise DataValidationError(
            f"Source file is not valid UTF-8: {e}",
            context={"path": str(resolved), "reason": str(e)},
        ) from e
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise DataValidationError(
            f"Malformed JSON in source file (line {e.lineno}, col {e.colno}): {e.msg}",
            context={"path": str(resolved), "lineno": e.lineno, "colno": e.colno, "pos": e.pos},
        ) from e
    if isinstance(data, dict):
        # Support an equivalent object wrapper: pick the first list value.
        lists = [v for v in data.values() if isinstance(v, list)]
        if len(lists) != 1:
            raise DataValidationError(
                "Source JSON object must wrap exactly one list of records",
                context={"path": str(resolved)},
            )
        data = lists[0]
    if not isinstance(data, list):
        raise DataValidationError(
            "Source JSON root must be an array of records",
            context={"path": str(resolved), "type": type(data).__name__},
        )
    return data, sha256_file(resolved), resolved
