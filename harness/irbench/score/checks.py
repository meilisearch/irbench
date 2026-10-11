from collections import defaultdict

from pydantic import BaseModel

from irbench.engines.base import SearchResult
from irbench.models import QueryTask
from irbench.score.constraints import (
    Docs,
    expected_facets,
    facet_accuracy,
    filter_violations,
    order_violations,
)


class TaskCheck(BaseModel):
    task_id: str
    mode: str
    base: str
    error: str | None = None
    filter_violations: int | None = None
    order_violations: int | None = None
    facet_accuracy: float | None = None
    facet_verifiable: bool | None = None


class CheckSummary(BaseModel):
    mode: str
    base: str
    n: int
    errors: int
    filter_violations: int | None = None
    order_violations: int | None = None
    facet_accuracy: float | None = None
    facet_verified: int | None = None
    facet_unverifiable: int | None = None


def check_task(
    task: QueryTask,
    result: SearchResult,
    check_ids: list[str] | None,
    docs_by_id: Docs,
    verify_limit: int,
) -> TaskCheck:
    check = TaskCheck(task_id=task.task_id, mode=task.mode, base=task.base)
    if task.filter is not None:
        check.filter_violations = filter_violations(result.ids, task.filter, docs_by_id)
    if task.sort is not None:
        check.order_violations = order_violations(result.ids, task.sort, docs_by_id)
    if task.facets:
        unique = list(dict.fromkeys(check_ids)) if check_ids is not None else None
        check.facet_verifiable = unique is not None and len(unique) < verify_limit
        if check.facet_verifiable:
            reported = result.facets or {}
            scores = [
                facet_accuracy(
                    {str(k): int(v) for k, v in reported.get(name, {}).items()},
                    expected_facets(unique, name, docs_by_id),
                )
                for name in task.facets
            ]
            check.facet_accuracy = sum(scores) / len(scores)
    return check


def summarize_checks(checks: list[TaskCheck]) -> list[CheckSummary]:
    groups: dict[tuple[str, str], list[TaskCheck]] = defaultdict(list)
    for c in checks:
        groups[(c.mode, c.base)].append(c)
    summaries = []
    for (mode, base), items in sorted(groups.items()):
        ok = [c for c in items if c.error is None]
        s = CheckSummary(mode=mode, base=base, n=len(items), errors=len(items) - len(ok))
        fv = [c.filter_violations for c in ok if c.filter_violations is not None]
        if fv:
            s.filter_violations = sum(fv)
        ov = [c.order_violations for c in ok if c.order_violations is not None]
        if ov:
            s.order_violations = sum(ov)
        faceted = [c for c in ok if c.facet_verifiable is not None]
        if faceted:
            verified = [c.facet_accuracy for c in faceted if c.facet_verifiable]
            s.facet_verified = len(verified)
            s.facet_unverifiable = len(faceted) - len(verified)
            s.facet_accuracy = sum(verified) / len(verified) if verified else None
        summaries.append(s)
    return summaries
