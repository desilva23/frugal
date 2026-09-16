"""Export the working cache as a committed fixture set.

The benchmark's headline claim is that anyone can reproduce it. That is only
true if the recorded responses are in the repository, so this trims the working
cache into something small enough to commit and writes it in the same layout,
which means ``--replay --cache benchmarks/fixtures`` just works.

Trimming truncates every result array to a little more than the depth the
planner considers. Nothing below that depth is ever read, so keeping it would
only make the repository larger.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".frugal-cache"
DESTINATION = ROOT / "benchmarks" / "fixtures"

#: A little above the planner's depth, so a change there does not immediately
#: invalidate the fixtures.
KEEP_PER_ARRAY = 12


def trim(payload: Any, depth: int = 0) -> Any:
    """Truncate result arrays, leaving structure and keys intact."""
    if isinstance(payload, dict):
        return {key: trim(value, depth + 1) for key, value in payload.items()}
    if isinstance(payload, list):
        return [trim(item, depth + 1) for item in payload[:KEEP_PER_ARRAY]]
    return payload


def main() -> int:
    if not SOURCE.is_dir():
        print(f"no cache at {SOURCE}; run the benchmark first", file=sys.stderr)
        return 1

    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)

    exported = 0
    for path in sorted(SOURCE.rglob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        record["response"] = trim(record.get("response", {}))
        target = DESTINATION / path.relative_to(SOURCE)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8"
        )
        exported += 1

    total = sum(p.stat().st_size for p in DESTINATION.rglob("*.json"))
    print(f"exported {exported} fixtures, {total / 1024 / 1024:.1f} MB -> {DESTINATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
