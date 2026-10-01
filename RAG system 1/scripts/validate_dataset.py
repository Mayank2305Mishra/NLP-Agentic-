"""CLI: python -m scripts.validate_dataset — thin wrapper over app services."""
from __future__ import annotations

import json
import sys

from app.config import get_settings
from app.ingestion.loader import load_raw_records
from app.ingestion.validator import validate_records


def main() -> int:
    settings = get_settings()
    try:
        records, sha, resolved = load_raw_records(settings.source_path)
    except Exception as e:
        print(f"LOAD FAILED: {e}", file=sys.stderr)
        return 1
    try:
        usable, summary = validate_records(records)
    except Exception as e:
        print(f"VALIDATION FAILED: {e}", file=sys.stderr)
        return 1
    print(json.dumps({
        "source": str(resolved), "sha256": sha,
        "total": summary.total, "usable": summary.usable, "excluded": summary.excluded,
        "duplicate_ids": summary.duplicate_ids[:10],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
