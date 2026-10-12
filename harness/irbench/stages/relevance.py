import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from irbench.engines.base import (
    EngineAdapter,
    EngineError,
    Request,
    SearchResult,
    SqlRequest,
    call,
    load_request,
)
from irbench.io import write_jsonl, write_run
from irbench.models import QueryTask, RunResults
from irbench.score.checks import TaskCheck, check_task
from irbench.stages.context import RunContext

REQUEST_ERRORS = (httpx.HTTPError, EngineError, ValueError, KeyError, TypeError)


@dataclass(frozen=True)
class RelevanceResult:
    n_tasks: int
    n_errors: int


def execute(client: Any, request: Request | SqlRequest, timeout: float) -> Any:
    if isinstance(request, SqlRequest):
        return client.execute(request.sql, request.params, timeout_s=timeout)
    return call(client, request, timeout)


def run_request(
    client: Any,
    adapter: EngineAdapter,
    task: QueryTask,
    request: Request | SqlRequest,
    timeout: float,
) -> SearchResult:
    return adapter.parse(task, execute(client, request, timeout))


def _load(path: Path) -> Request | SqlRequest:
    return load_request(json.loads(path.read_text(encoding="utf-8")))


def _error(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


def relevance_stage(ctx: RunContext) -> RelevanceResult:
    """Execute each rendered request once, sequentially; score inputs go to the run dir."""
    scenario, adapter = ctx.scenario, ctx.adapter
    timeout = scenario.request_timeout_s
    results: RunResults = {}
    checks: list[TaskCheck] = []
    errors: list[dict[str, str]] = []

    with ctx.client() as client:
        for task in ctx.tasks:
            path = ctx.rendered_path(task)
            if not path.exists():  # unsupported: listed in na.json
                continue
            try:
                found = run_request(client, adapter, task, _load(path), timeout)
            except REQUEST_ERRORS as e:
                errors.append({"task_id": task.task_id, "request": "search", "error": _error(e)})
                checks.append(
                    TaskCheck(task_id=task.task_id, mode=task.mode, base=task.base, error=_error(e))
                )
                continue
            found = SearchResult(ids=found.ids[: scenario.top_k], facets=found.facets)
            check_ids, check_error = None, None
            if task.facets:
                try:
                    check_request = _load(ctx.rendered_path(task, check=True))
                    check_ids = run_request(client, adapter, task, check_request, timeout).ids
                except REQUEST_ERRORS as e:
                    check_error = _error(e)
                    errors.append(
                        {"task_id": task.task_id, "request": "check", "error": check_error}
                    )
            results[task.task_id] = found.ids
            check = check_task(task, found, check_ids, ctx.docs_by_id, scenario.facet_verify_limit)
            check.check_error = check_error
            checks.append(check)

    write_run(
        ctx.run_dir / "run.trec", {t: ids for t, ids in results.items() if ids}, ctx.spec.run_id
    )
    write_jsonl(ctx.run_dir / "checks.jsonl", (c.model_dump() for c in checks))
    write_jsonl(ctx.run_dir / "errors.jsonl", errors)
    n_failed = sum(1 for e in errors if e["request"] == "search")
    return RelevanceResult(n_tasks=len(checks), n_errors=n_failed)
