"""Draw the cost/recall frontier as an SVG, from both committed recordings.

The curve is the most useful thing this project measured and it was reachable
only as a markdown table some hundreds of lines into the README. A reader who
sees one picture sees this one.

Written by hand rather than with a plotting library, for two reasons: adding a
charting dependency to a search planner is not a trade worth making, and the
output has to be deterministic so that a test can assert the committed file
still matches the committed numbers.

Colours are mid-tones that read on both the light and dark GitHub themes, since
an SVG embedded in a README cannot ask which one is in use.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDINGS = (
    ("first recording", ROOT / "benchmarks" / "ablation-first.json"),
    ("second recording", ROOT / "benchmarks" / "ablation.json"),
)
OUTPUT = ROOT / "docs" / "frontier.svg"

W, H = 760, 430
LEFT, RIGHT, TOP, BOTTOM = 72, 40, 58, 64
X0, X1 = 0.7, 2.75
Y0, Y1 = 74.0, 101.0

INK = "#8b949e"
GRID = "#8b949e"
PLAN = "#2f81f7"
SECOND = "#3fb950"
FONT = "-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif"
ARMS = ("routed-1x1", "routed-2x1", "routed-3x1")


def x(value: float) -> float:
    return LEFT + (value - X0) / (X1 - X0) * (W - LEFT - RIGHT)


def y(value: float) -> float:
    return TOP + (Y1 - value) / (Y1 - Y0) * (H - TOP - BOTTOM)


def text(cx: float, cy: float, body: str, *, size: int = 12,
         anchor: str = "start", fill: str = INK, weight: str = "normal") -> str:
    return (
        f'<text x="{cx:.1f}" y="{cy:.1f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}" font-family="{FONT}">{body}</text>'
    )


def main(argv: list[str] | None = None) -> int:
    """Render the chart, to docs/frontier.svg or to the path given.

    The path argument exists for the test, which has to render somewhere other
    than the committed file: rendering over it and then comparing meant a stale
    chart failed once and passed on every re-run after, having quietly
    rewritten the file it was meant to be checking.
    """
    output = Path(argv[0]) if argv else OUTPUT
    series: list[tuple[str, list[tuple[float, int]]]] = []
    for label, path in RECORDINGS:
        data = json.loads(path.read_text(encoding="utf-8"))["ablation"]
        series.append(
            (label, [(data[a]["searches_per_question"], data[a]["answered"]) for a in ARMS])
        )

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" role="img" aria-label="Questions answered against searches spent, '
        f'in two recordings">'
    ]
    parts.append(text(LEFT, 26, "What each additional search buys", size=17, weight="600"))
    parts.append(
        text(
            LEFT, 44,
            "100 questions, recorded twice · every point replays from committed fixtures",
        )
    )

    for value in range(75, 101, 5):
        gy = y(value)
        parts.append(
            f'<line x1="{LEFT}" y1="{gy:.1f}" x2="{W - RIGHT}" y2="{gy:.1f}" '
            f'stroke="{GRID}" stroke-opacity="0.18" stroke-width="1"/>'
        )
        parts.append(text(LEFT - 10, gy + 4, str(value), size=11, anchor="end"))
    for tick in (1.0, 1.5, 2.0, 2.5):
        parts.append(text(x(tick), H - BOTTOM + 20, f"{tick:g}", size=11, anchor="middle"))
    parts.append(
        text((LEFT + W - RIGHT) / 2, H - 16, "searches per question", size=12, anchor="middle")
    )
    mid = (TOP + H - BOTTOM) / 2
    parts.append(
        f'<text x="20" y="{mid:.1f}" font-size="12" fill="{INK}" text-anchor="middle" '
        f'transform="rotate(-90 20 {mid:.1f})" font-family="{FONT}">questions answered</text>'
    )

    for (label, points), colour in zip(series, (PLAN, SECOND), strict=True):
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{x(px):.1f},{y(py):.1f}" for i, (px, py) in enumerate(points)
        )
        parts.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2.5"/>')
        for px, py in points:
            parts.append(f'<circle cx="{x(px):.1f}" cy="{y(py):.1f}" r="5" fill="{colour}"/>')
        end_x, end_y = points[-1]
        parts.append(
            text(x(end_x) + 10, y(end_y) + 4, f"{label} — {end_y}", size=11, fill=colour)
        )

    (one_a, two_a, three_a), (one_b, two_b, three_b) = (
        [answered for _, answered in points] for _, points in series
    )
    lo, hi = sorted((one_a, one_b))
    parts.append(text(x(1.0), y(lo) + 24, f"one engine: {lo} to {hi}", size=12, anchor="middle"))
    lo, hi = sorted((two_a, two_b))
    parts.append(
        text(x(2.0) - 12, y(hi) - 14, f"two engines, the default: {lo} to {hi}",
             size=12, anchor="end", fill=INK, weight="600")
    )
    gain = sorted((two_a - one_a, two_b - one_b))
    parts.append(
        text(x(1.5), y((one_a + two_a) / 2) + 34,
             f"the second search: +{gain[0]} to +{gain[1]}", size=12, anchor="middle")
    )
    extra = sorted((three_a - two_a, three_b - two_b))
    parts.append(
        text(x(2.26), y(min(three_a, three_b)) + 24,
             f"a third engine: +{extra[0]} to +{extra[1]}", size=12, anchor="middle")
    )

    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
