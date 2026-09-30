"""Compare the two committed recordings question by question.

The README leads with a number no results file carried: how many questions a
strategy answered in *both* recordings. It was computed once, by hand, and
typed in -- which made it the one figure in the repository that could go stale
without anything noticing, and the most prominent one at that. This computes it
from the two results files and writes it down, so a test can hold the README to
it.

    python scripts/compare_recordings.py            # writes benchmarks/recordings.json
    python scripts/compare_recordings.py OUT.json   # writes somewhere else
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIRST = ROOT / "benchmarks" / "results-first.json"
SECOND = ROOT / "benchmarks" / "results.json"
OUTPUT = ROOT / "benchmarks" / "recordings.json"


def verdicts(path: Path) -> dict[str, dict[str, bool]]:
    """Per strategy, whether each question was answered, from one results file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, bool]] = {}
    for outcome in payload["outcomes"]:
        out.setdefault(outcome["strategy"], {})[outcome["question_id"]] = outcome["found"]
    return out


def compare(first: Path = FIRST, second: Path = SECOND) -> dict[str, Any]:
    """How each strategy fared across the two recordings."""
    a, b = verdicts(first), verdicts(second)
    table: dict[str, Any] = {}
    for strategy in sorted(a):
        x, y = a[strategy], b[strategy]
        if set(x) != set(y):
            raise ValueError(f"{strategy}: the two recordings cover different questions")
        table[strategy] = {
            "questions": len(x),
            "first": sum(x.values()),
            "second": sum(y.values()),
            "both": sum(x[q] and y[q] for q in x),
            "never": sum(not x[q] and not y[q] for q in x),
            "flipped": sum(x[q] != y[q] for q in x),
        }
    return table


def main(argv: list[str] | None = None) -> int:
    output = Path(argv[0]) if argv else OUTPUT
    table = compare()
    output.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, row in table.items():
        print(f"{name:<14} first {row['first']:>3}  second {row['second']:>3}  "
              f"both {row['both']:>3}  flipped {row['flipped']:>3}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
