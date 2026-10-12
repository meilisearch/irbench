"""Static HTML report (spec §11). One page per scenario × hardware profile (results are
only compared within a profile, §8) and one per run. Plain HTML, no JavaScript."""

import shutil
from collections import defaultdict
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import jinja2

from irbench.publish import UNPUBLISHABLE_PROFILES as UNPUBLISHED_PROFILES
from irbench.report.charts import Point, Series, qps_chart
from irbench.report.data import (
    PERTURBATIONS,
    RunResult,
    ScenarioResults,
    Unreadable,
    group_key,
    load_results,
)

MARKER = ".irbench-report"
DEFAULT_DOCS_URL = "https://github.com/meilisearch/irbench/tree/main/docs"
BASE_GROUPS = ("keyword/keyword", "semantic/semantic", "hybrid/hybrid")
VECTOR_GROUPS = ("semantic/semantic", "hybrid/hybrid")
MAX_SLOTS = 8  # categorical palette size; later engines fold into "other" (slot 0)


@dataclass(frozen=True)
class Cell:
    text: str
    kind: str  # score | na | violated | missing | failed
    note: str = ""


@dataclass(frozen=True)
class SiteResult:
    pages: list[Path]
    unreadable: list[Unreadable]


def fmt_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def fmt_delta(value: float | None) -> str:
    return "—" if value is None else f"{value:+.3f}".replace("-", "−")


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}" if value >= 10 else f"{value:.2f}"


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    value = float(n)
    for unit in ("B", "KB", "MB"):
        if value < 1000:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} GB"


def fmt_group(group: str) -> str:
    mode, _, base = group.partition("/")
    return mode if mode == base else f"{mode} ({base})"


def fmt_g(value: float) -> str:
    """Like `fmt_ms` but with no fixed decimals — matches NO_VALID_STEP's `{p99:g}`."""
    return f"{value:g}"


def _safe_url(url: str | None) -> str | None:
    """Only http(s) URLs are safe to render as a live link; config files come from
    contributor PRs and could set e.g. a `javascript:` URL otherwise."""
    if isinstance(url, str) and url.lower().startswith(("http://", "https://")):
        return url
    return None


def engine_slots(scenarios: list[ScenarioResults]) -> dict[str, int]:
    """Color slot per engine, the same on every page (color follows the entity)."""
    engines = sorted({r.engine for s in scenarios for r in s.runs})
    return {e: i + 1 if i < MAX_SLOTS else 0 for i, e in enumerate(engines)}


def relevance_cell(run: RunResult, group: str) -> Cell:
    reason = run.na_reason(group)
    if reason:
        return Cell("N/A", "na", reason)
    value = run.ndcg(group)
    if value is None:
        return Cell("—", "missing", "not run")
    violation = run.violation(group)
    if violation:
        return Cell(
            fmt_score(value),
            "violated",
            f"constraint violated: {violation}; this score does not measure the mode",
        )
    return Cell(fmt_score(value), "score")


def _delta(value: float | None, base: float) -> float | None:
    return None if value is None else value - base


def perturbation_rows(s: ScenarioResults) -> list[dict[str, Any]]:
    rows = []
    for run in s.runs:
        if run.failure:
            continue
        for group in BASE_GROUPS:
            original = run.ndcg(group, "original")
            if original is None:
                continue
            rows.append(
                {
                    "run": run,
                    "group": group,
                    "original": original,
                    "deltas": {p: _delta(run.ndcg(group, p), original) for p in PERTURBATIONS[1:]},
                }
            )
    return rows


def model_comparison(s: ScenarioResults) -> list[dict[str, Any]]:
    if len(s.models) < 2:
        return []
    tables = []
    for group in VECTOR_GROUPS:
        by_label: dict[str, dict[str, Cell]] = defaultdict(dict)
        for run in s.runs:
            if not run.failure:
                by_label[run.label][run.model] = relevance_cell(run, group)
        rows = [
            {
                "label": label,
                "cells": [cells.get(m, Cell("—", "missing", "not run")) for m in s.models],
            }
            for label, cells in sorted(by_label.items())
        ]
        tables.append({"group": group, "rows": rows})
    return tables


def na_notes(s: ScenarioResults) -> list[dict[str, Any]]:
    notes: dict[tuple[str, str], list[str]] = defaultdict(list)
    for run in s.runs:
        for group in run.groups:
            reason = run.na_reason(group)
            if reason and group not in notes[(run.label, reason)]:
                notes[(run.label, reason)].append(group)
    return [
        {"label": label, "reason": reason, "groups": groups}
        for (label, reason), groups in sorted(notes.items())
    ]


NO_VALID_STEP = "no concurrency step met p99 ≤ {p99:g} ms and ≤ {err:.1%} errors"


def _latency_cell(run: RunResult, group: str) -> Cell:
    lat = run.perf["latency"].get(group)
    if lat is None:
        return Cell("N/A", "na", run.na_reason(group) or "not measured")
    note = f"{lat['count']} requests, {lat['errors']} errors" if lat.get("count") else ""
    return Cell(f"{fmt_ms(lat['p50_ms'])} / {fmt_ms(lat['p99_ms'])}", "score", note)


def _qps_cell(run: RunResult, group: str) -> Cell:
    if group in run.perf["skipped_groups"]:
        return Cell("N/A", "na", run.na_reason(group) or "the engine does not support this mode")
    t = run.perf["throughput"].get(group)
    if t is None:
        return Cell("—", "missing", "not measured")
    if t["max_qps"] is None:
        limits = run.perf["limits"]
        return Cell(
            "none",
            "violated",
            NO_VALID_STEP.format(p99=limits["p99_ms"], err=limits["max_error_rate"]),
        )
    return Cell(
        f"{t['max_qps']:,.0f}", "score", f"at {t['max_qps_concurrency']} concurrent clients"
    )


def _series(run: RunResult, group: str, slots: dict[str, int]) -> Series:
    points = [
        Point(
            st["concurrency"],
            st["qps"],
            st["valid"],
            f"{st['concurrency']} clients: {st['qps']:,.0f} QPS, p99 {fmt_ms(st['p99_ms'])} ms, "
            f"errors {st['error_rate']:.2%}" + ("" if st["valid"] else " (invalid step)"),
        )
        for st in run.perf["throughput"][group]["steps"]
    ]
    return Series(run.label, slots.get(run.engine, 0), run.track != "default", points)


def perf_view(s: ScenarioResults, slots: dict[str, int]) -> dict[str, Any]:
    measured = [r for r in s.runs if r.perf is not None and not r.failure]
    latency_groups = sorted(
        {g for r in measured for g in r.perf["latency"] if g != "all"}, key=group_key
    )
    throughput_groups = sorted(
        {g for r in measured for g in (*r.perf["throughput"], *r.perf["skipped_groups"])},
        key=group_key,
    )
    charts = []
    for model in s.models:
        runs = [r for r in measured if r.model == model]
        for group in throughput_groups:
            with_data = [r for r in runs if group in r.perf["throughput"]]
            series = [_series(r, group, slots) for r in with_data]
            steps = sorted({p.concurrency for se in series for p in se.points})
            svg = qps_chart(
                f"{fmt_group(group)}: queries per second by concurrency, {model}", series, steps
            )
            if svg is None:
                continue
            table = [
                {
                    "concurrency": c,
                    "cells": [
                        next(
                            (
                                f"{p.qps:,.0f}" + ("" if p.valid else " (invalid)")
                                for p in se.points
                                if p.concurrency == c
                            ),
                            "—",
                        )
                        for se in series
                    ],
                }
                for c in steps
            ]
            charts.append(
                {"group": group, "model": model, "svg": svg, "series": series, "table": table}
            )
    return {
        "runs": measured,
        # Limits are the same for every run of a scenario; take them from the first
        # measured run (the leading run referred to in the note below is per-run steps,
        # not this list).
        "limits": measured[0].perf["limits"] if measured else None,
        "latency": [
            {"group": g, "cells": [_latency_cell(r, g) for r in measured]} for g in latency_groups
        ],
        "max_qps": [
            {"group": g, "cells": [_qps_cell(r, g) for r in measured]} for g in throughput_groups
        ],
        "charts": charts,
        "indexing": [r for r in s.runs if r.index and not r.failure],
        "unmeasured": [r.label for r in s.runs if r.perf is None and not r.failure],
    }


def scenario_view(s: ScenarioResults, slots: dict[str, int]) -> dict[str, Any]:
    return {
        "columns": s.groups,
        "matrix": [{"run": r, "cells": [relevance_cell(r, g) for g in s.groups]} for r in s.runs],
        "na_notes": na_notes(s),
        "perturbations": perturbation_rows(s),
        "models": model_comparison(s),
        "comparable": s.hardware not in UNPUBLISHED_PROFILES,
        "perf": perf_view(s, slots),
    }


def run_view(run: RunResult) -> dict[str, Any]:
    rows = []
    for group in run.groups:
        reason = run.na_reason(group)
        all_metrics = None
        if run.scores is not None and not reason:
            mode, _, base = group.partition("/")
            all_metrics = next(
                (
                    g
                    for g in run.scores["relevance"]
                    if (g["mode"], g["base"], g["perturbation"]) == (mode, base, "all")
                ),
                None,
            )
        rows.append(
            {
                "group": group,
                "na": reason,
                "violation": run.violation(group),
                "n": all_metrics and all_metrics["n"],
                "metrics": all_metrics and all_metrics["metrics"],
                "by_perturbation": {p: run.ndcg(group, p) for p in PERTURBATIONS},
            }
        )
    config = run.manifest.get("config", {})
    return {
        "relevance": rows,
        "checks": sorted(
            (run.scores or {}).get("checks", []),
            key=lambda c: group_key(f"{c['mode']}/{c['base']}"),
        ),
        "diff": config.get("diff_vs_default", {}),
        "credits": [
            {"name": c["name"], "safe_url": _safe_url(c.get("url"))}
            for c in config.get("credits", [])
        ],
        "stages": run.manifest.get("stages", {}),
        "perf": [
            {"group": g, "latency": _latency_cell(run, g), "max_qps": _qps_cell(run, g)}
            # "all" mixes modes and has no group_key; it stays out, as in summarize_perf.
            for g in sorted((k for k in run.perf["latency"] if k != "all"), key=group_key)
        ]
        if run.perf and not run.failure
        else [],
    }


def _env() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.PackageLoader("irbench.report", "templates"),
        autoescape=True,
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update(
        score=fmt_score, delta=fmt_delta, ms=fmt_ms, bytes=fmt_bytes, group=fmt_group, g=fmt_g
    )
    return env


def _prepare_out(out: Path) -> None:
    if out.exists():
        if not out.is_dir():
            raise FileExistsError(f"{out} exists and is not a directory; choose another --out")
        if not (out / MARKER).exists() and any(out.iterdir()):
            raise FileExistsError(
                f"{out} is not empty and was not written by `irbench report`; choose another --out"
            )
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / MARKER).write_text("written by `irbench report`; replaced on every build\n")


def build_site(results_dir: Path, out: Path, docs_url: str = DEFAULT_DOCS_URL) -> SiteResult:
    scenarios, unreadable = load_results(results_dir)
    _prepare_out(out)
    env = _env()
    slots = engine_slots(scenarios)
    pages: list[Path] = []

    def write(rel: str, template: str, **context: Any) -> None:
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        html = env.get_template(template).render(
            docs_url=docs_url, root="../" * rel.count("/"), **context
        )
        path.write_text(html, encoding="utf-8")
        pages.append(path)

    write("index.html", "index.html.j2", scenarios=scenarios, unreadable=unreadable)
    for s in scenarios:
        write(f"{s.slug}.html", "scenario.html.j2", s=s, view=scenario_view(s, slots))
        for run in s.runs:
            write(f"runs/{run.run_id}.html", "run.html.j2", run=run, s=s, view=run_view(run))
    style = resources.files("irbench.report").joinpath("static/style.css")
    (out / "style.css").write_text(style.read_text(encoding="utf-8"), encoding="utf-8")
    return SiteResult(pages, unreadable)
