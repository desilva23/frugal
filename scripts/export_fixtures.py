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

#: Arrays that must be kept whole. Truncating a time series does not shorten it,
#: it changes what it says: cutting the India electric-vehicle series from 53
#: points to 12 turned "trending down 6%" into "trending up 456%", so a judge
#: replaying the fixtures would have read a different answer from the one the
#: live run gave. These are observations, not a page of results to paginate.
KEEP_WHOLE = frozenset({"timeline_data", "values", "interest_over_time", "timeline"})


def trim(payload: Any, depth: int = 0, *, whole: bool = False) -> Any:
    """Truncate result arrays, leaving structure, keys and series intact."""
    if isinstance(payload, dict):
        return {
            key: trim(value, depth + 1, whole=whole or key in KEEP_WHOLE)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        kept = payload if whole else payload[:KEEP_PER_ARRAY]
        return [trim(item, depth + 1, whole=whole) for item in kept]
    return payload


def _refuse(source: Path, destination: Path) -> str | None:
    """Why ``destination`` must not be deleted, or None if it may be.

    The destination is removed and rewritten, and it became a positional
    argument when the repository started carrying two recordings. Swapped
    arguments would then delete the live cache -- raw responses that were paid
    for and cannot be fetched again identically -- and a destination of
    "benchmarks" would delete the question sets and every results file.
    """
    src, dst = source.resolve(), destination.resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        return f"{destination} overlaps the source {source}"
    if not dst.name.startswith("fixtures"):
        return f"{destination} is not a fixtures directory (its name must start with 'fixtures')"
    if dst.exists():
        strays = [p.name for p in dst.iterdir() if not p.is_dir()]
        if strays:
            listed = ", ".join(sorted(strays)[:5])
            return f"{destination} holds files that are not fixtures: {listed}"
    return None


def main(argv: list[str] | None = None) -> int:
    """Export one cache as one fixture set.

    Takes an optional source and destination, because the repository carries two
    recordings: the benchmark is reported from both, and a reader should be able
    to replay either.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    source = Path(args[0]).resolve() if args else SOURCE
    destination = Path(args[1]).resolve() if len(args) > 1 else DESTINATION
    return export(source, destination)


def export(source: Path, destination: Path) -> int:
    """Replace ``destination`` with a trimmed copy of the cache at ``source``."""
    if not source.is_dir():
        print(f"no cache at {source}; run the benchmark first", file=sys.stderr)
        return 1

    problem = _refuse(source, destination)
    if problem:
        print(f"refusing to export: {problem}", file=sys.stderr)
        return 1

    if destination.exists():
        shutil.rmtree(destination)

    exported = 0
    for path in sorted(source.rglob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        record["response"] = trim(record.get("response", {}))
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8"
        )
        exported += 1

    total = sum(p.stat().st_size for p in destination.rglob("*.json"))
    print(f"exported {exported} fixtures, {total / 1024 / 1024:.1f} MB -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
