"""CLI: python -m scripts.build_index — atomic rebuild of the dense index."""
from __future__ import annotations

import argparse
import json
import sys

from app.config import get_settings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the dense vector index atomically.")
    ap.add_argument("--on-duplicate", default="fail", choices=["fail", "keep_first"])
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args(argv)
    settings = get_settings()
    from app.indexing.builder import build_index
    try:
        manifest = build_index(settings, on_duplicate=args.on_duplicate, progress=args.progress)
    except Exception as e:
        print(f"BUILD FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(json.dumps(manifest.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
