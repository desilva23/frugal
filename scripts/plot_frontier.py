"""Draw the cost/recall frontier as an SVG, from the committed results.

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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABLATION = ROOT / "benchmarks" / "ablation.json"
OUTPUT = ROOT / "docs" / "frontier.svg"

W, H = 760, 430
LEFT, RIGHT, TOP, BOTTOM = 72, 30, 58, 64
X0, X1 = 0.8, 4.4
Y0, Y1 = 74.0, 101.0

INK = "#8b949e"
GRID = "#8b949e"
PLAN = "#2f81f7"
FIXED = "#bc8cff"
BASE = "#768390"


def x(value: float) -> float:
    return LEFT + (value - X0) / (X1 - X0) * (W - LEFT - RIGHT)


def y(value: float) -> float:
    return TOP + (Y1 - value) / (Y1 - Y0) * (H - TOP - BOTTOM)


def text(cx: float, cy: float, body: str, *, size: int = 12,
         anchor: str = "start", fill: str = INK, weight: str = "normal") -> str:
    return (
        f'<text x="{cx:.1f}" y="{cy:.1f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">'
        f"{body}</text>"
    )


def main() -> int:
    data = json.loads(ABLATION.read_text(encoding="utf-8"))["ablation"]

    def point(arm: str) -> tuple[float, int]:
        s = data[arm]
        return s["searches_per_question"], s["answered"]

    frontier = [point(a) for a in ("routed-1x1", "routed-2x1", "routed-3x1", "routed-3x2")]
    fixed = {
        "web + shopping": point("fixed-shopping"),
        "web + news": point("fixed-news"),
        "web + scholar": point("fixed-scholar"),
    }
    naive_pt = point("naive")
    param_pt = point("parameterised")

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" role="img" aria-label="Questions answered against searches spent">'
    ]
    parts.append(
        text(LEFT, 26, "What each additional search buys", size=17, weight="600", fill=INK)
    )
    parts.append(
        text(LEFT, 44, "100 questions · every point reproduces from committed fixtures",
             size=12, fill=INK)
    )

    # grid and axes
    for value in range(75, 101, 5):
        gy = y(value)
        parts.append(
            f'<line x1="{LEFT}" y1="{gy:.1f}" x2="{W - RIGHT}" y2="{gy:.1f}" '
            f'stroke="{GRID}" stroke-opacity="0.18" stroke-width="1"/>'
        )
        parts.append(text(LEFT - 10, gy + 4, str(value), size=11, anchor="end"))
    for tick in (1, 2, 3, 4):
        tx = x(tick)
        parts.append(text(tx, H - BOTTOM + 20, str(tick), size=11, anchor="middle"))
    parts.append(
        text((LEFT + W - RIGHT) / 2, H - 16, "searches per question", size=12, anchor="middle")
    )
    parts.append(
        f'<text x="20" y="{(TOP + H - BOTTOM) / 2:.1f}" font-size="12" fill="{INK}" '
        f'text-anchor="middle" transform="rotate(-90 20 {(TOP + H - BOTTOM) / 2:.1f})" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">'
        f"questions answered</text>"
    )

    # the planned frontier
    path = " ".join(
        f"{'M' if i == 0 else 'L'}{x(sx):.1f},{y(sy):.1f}"
        for i, (sx, sy) in enumerate(frontier)
    )
    parts.append(f'<path d="{path}" fill="none" stroke="{PLAN}" stroke-width="2.5"/>')

    # baselines
    for label, (px, py), colour in (
        ("web search, verbatim", naive_pt, BASE),
        ("web search + parameters", param_pt, BASE),
    ):
        parts.append(
            f'<circle cx="{x(px):.1f}" cy="{y(py):.1f}" r="4.5" fill="{colour}"/>'
        )
        parts.append(text(x(px) + 10, y(py) + 4, f"{label} — {py}", size=11, fill=colour))

    # The planned points first, so a control landing on one shows as a ring
    # around it rather than hiding underneath.
    for i, (px, py) in enumerate(frontier):
        parts.append(f'<circle cx="{x(px):.1f}" cy="{y(py):.1f}" r="5" fill="{PLAN}"/>')
        if i == 1:
            parts.append(
                text(x(px) - 14, y(py) - 6, f"default — {py} answers, {px:.1f} searches",
                     size=12, anchor="end", fill=PLAN, weight="600")
            )
        elif i == 2:
            parts.append(text(x(px) + 9, y(py) - 6, f"{py}", size=11, fill=PLAN))
        elif i == 3:
            parts.append(
                text(x(px) - 8, y(py) - 16, f"{py} answers — and the last one",
                     size=11, anchor="end", fill=PLAN)
            )
            parts.append(
                text(x(px) - 8, y(py) - 3, "cost 184 searches by itself",
                     size=11, anchor="end", fill=PLAN)
            )

    # Fixed pairings: the control. Drawn after, and wide enough to encircle a
    # planned point it coincides with -- which one of them does exactly.
    planned_at = {(round(px, 2), py) for px, py in frontier}
    for label, (px, py) in fixed.items():
        coincides = (round(px, 2), py) in planned_at
        radius = 10 if coincides else 4.5
        parts.append(
            f'<circle cx="{x(px):.1f}" cy="{y(py):.1f}" r="{radius}" fill="none" '
            f'stroke="{FIXED}" stroke-width="2"/>'
        )
        offset = 15 if coincides else 10
        parts.append(
            text(x(px) + offset, y(py) + 4, f"{label} — {py}", size=11, fill=FIXED)
        )
        if coincides:
            parts.append(
                text(x(px) + offset, y(py) + 18,
                     "same point: choosing the engine earned nothing",
                     size=10, fill=FIXED)
            )

    parts.append("</svg>")
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
