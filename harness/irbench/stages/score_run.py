from typing import Any

from irbench.io import read_json, read_jsonl, read_run, write_json
from irbench.score.checks import CheckSummary, TaskCheck, summarize_checks
from irbench.score.relevance import score_relevance
from irbench.stages.context import RunContext


def score_stage(ctx: RunContext) -> dict[str, Any]:
    run_dir = ctx.run_dir
    na: dict[str, str] = read_json(run_dir / "na.json")
    supported = [t for t in ctx.tasks if t.task_id not in na]
    run = read_run(run_dir / "run.trec")
    unknown = sorted(set(run) - {t.task_id for t in supported})
    if unknown:
        raise ValueError(f"run.trec has results for unknown or unsupported tasks: {unknown[:5]}")

    not_applicable: dict[str, str] = {}
    for task in ctx.tasks:
        if task.task_id in na:
            not_applicable.setdefault(f"{task.mode}:{task.base}", na[task.task_id])

    checks = [TaskCheck.model_validate(r) for r in read_jsonl(run_dir / "checks.jsonl")]
    summaries = summarize_checks(checks)
    report = {
        "run_id": ctx.spec.run_id,
        "scenario": ctx.spec.scenario,
        "engine": ctx.spec.engine,
        "model": ctx.spec.model,
        "track": ctx.spec.track,
        "relevance": [g.model_dump() for g in score_relevance(supported, ctx.qrels, run)],
        "checks": [s.model_dump() for s in summaries],
        "constraint_violations": _constraint_violations(summaries),
        "not_applicable": not_applicable,
        "errors": sum(1 for e in read_jsonl(run_dir / "errors.jsonl") if e["request"] == "search"),
    }
    write_json(run_dir / "scores.json", report)
    return report


def _constraint_violations(summaries: list[CheckSummary]) -> dict[str, str]:
    """Groups whose results break the mode's contract: their nDCG does not measure that mode."""
    out: dict[str, str] = {}
    for s in summaries:
        broken = []
        if s.filter_violations:
            broken.append(f"{s.filter_violations} filter violations")
        if s.order_violations:
            broken.append(f"{s.order_violations} order violations")
        if broken:
            out[f"{s.mode}:{s.base}"] = ", ".join(broken)
    return out
