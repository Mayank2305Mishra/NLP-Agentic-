"""Build deterministic System 4 indexes.

Usage:
    python scripts/build_index.py --source final_transcripts.json --output artifacts/index.pkl
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from system4.config.settings import Settings
from system4.ingestion.artifacts import save, source_hash
from system4.ingestion.loader import load_conversations
from system4.retrieval.index import LocalIndex
from system4.schema import SYSTEM_VERSION


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic System 4 local indexes")
    parser.add_argument("--source", required=True, help="path to final_transcripts.json")
    parser.add_argument("--output", required=True, help="path to write the index artifact")
    parser.add_argument("--hash-dimensions", type=int, default=None, help="override dense hash dimensions")
    args = parser.parse_args()

    started = time.perf_counter()
    settings = Settings(**({"hash_dimensions": args.hash_dimensions} if args.hash_dimensions else {}))
    stats: dict[str, int] = {}
    conversations = list(load_conversations(args.source, stats))
    index = LocalIndex.build(conversations, settings)
    elapsed = round(time.perf_counter() - started, 2)
    save(
        args.output,
        {
            "source_hash": source_hash(args.source),
            "index": index,
            "version": SYSTEM_VERSION,
            "settings_fingerprint": settings.fingerprint(),
            "conversation_count": len(conversations),
            "records_read": stats.get("records_read"),
            "records_skipped": stats.get("records_skipped"),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    print(
        f"indexed {len(conversations)} conversations "
        f"(skipped {stats.get('records_skipped', 0)} malformed) "
        f"in {elapsed}s -> {Path(args.output)}"
    )


if __name__ == "__main__":
    main()
