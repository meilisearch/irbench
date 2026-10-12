"""Inline SVG line chart: QPS by concurrency step, one line per run. No JavaScript: hover
uses native <title> tooltips, and the step table under each chart carries the numbers."""

import math
from dataclasses import dataclass

from markupsafe import Markup, escape

WIDTH, HEIGHT = 640, 300
LEFT, RIGHT, TOP, BOTTOM = 64, 120, 16, 44


@dataclass(frozen=True)
class Point:
    concurrency: int
    qps: float
    valid: bool  # p99 and error rate within the scenario's limits (spec §5.7)
    note: str  # tooltip text


@dataclass(frozen=True)
class Series:
    label: str
    slot: int  # categorical slot 1..8, fixed per engine; 0 = folded "other"
    dashed: bool  # tuned track
    points: list[Point]


def nice_max(value: float) -> float:
    """Smallest 1, 2, 2.5 or 5 × 10^k at or above value."""
    if value <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(value))
    for m in (1, 2, 2.5, 5, 10):
        if value <= m * exp * (1 + 1e-9):
            return m * exp
    return 10 * exp


def qps_chart(title: str, series: list[Series], steps: list[int]) -> Markup | None:
    plotted = [(s, [p for p in s.points if p.concurrency in steps]) for s in series]
    plotted = [(s, pts) for s, pts in plotted if pts]
    if not plotted or not steps:
        return None
    ymax = nice_max(max(p.qps for _, pts in plotted for p in pts))
    pw, ph = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

    def x(c: int) -> float:
        return LEFT + steps.index(c) / max(len(steps) - 1, 1) * pw

    def y(v: float) -> float:
        return TOP + ph - v / ymax * ph

    out = [
        f'<svg class="chart" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        f'aria-label="{escape(title)}">'
    ]
    for i in range(5):
        v = ymax * i / 4
        out.append(
            f'<line class="grid" x1="{LEFT}" x2="{LEFT + pw}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
        )
        out.append(
            f'<text class="tick" x="{LEFT - 8}" y="{y(v) + 4:.1f}" '
            f'text-anchor="end">{v:,.0f}</text>'
        )
    for c in steps:
        out.append(
            f'<text class="tick" x="{x(c):.1f}" y="{TOP + ph + 18}" text-anchor="middle">{c}</text>'
        )
    mid = TOP + ph / 2
    out.append(
        f'<text class="axis-label" x="{LEFT + pw / 2:.1f}" y="{HEIGHT - 6}" '
        'text-anchor="middle">concurrent clients</text>'
    )
    out.append(
        f'<text class="axis-label" x="14" y="{mid:.1f}" transform="rotate(-90 14 '
        f'{mid:.1f})" text-anchor="middle">queries per second</text>'
    )

    direct = len(plotted) <= 4
    for s, pts in plotted:
        color = f"var(--series-{s.slot})" if s.slot else "var(--series-other)"
        if len(pts) > 1:
            coords = " ".join(f"{x(p.concurrency):.1f},{y(p.qps):.1f}" for p in pts)
            dashed = " dashed" if s.dashed else ""
            out.append(
                f'<polyline class="series{dashed}" points="{coords}" style="stroke:{color}"/>'
            )
        for p in pts:
            cls = "marker" if p.valid else "marker invalid"
            out.append(
                f'<circle class="{cls}" cx="{x(p.concurrency):.1f}" cy="{y(p.qps):.1f}" '
                f'r="4" style="--c:{color}"><title>{escape(s.label)}: '
                f"{escape(p.note)}</title></circle>"
            )
        if direct:
            last = pts[-1]
            out.append(
                f'<text class="direct" x="{x(last.concurrency) + 10:.1f}" '
                f'y="{y(last.qps) + 4:.1f}">{escape(s.label)}</text>'
            )
    out.append("</svg>")
    return Markup("".join(out))
