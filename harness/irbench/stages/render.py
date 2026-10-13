import shutil
from dataclasses import dataclass

from irbench.io import write_json
from irbench.stages.context import RunContext


@dataclass(frozen=True)
class RenderResult:
    n_rendered: int
    n_unsupported: int


def render_stage(ctx: RunContext) -> RenderResult:
    """Write every supported task as a request file; unsupported ones go to na.json.

    rendered/ holds the top_k requests (relevance, and perf at top_k) plus facet checks;
    each other page size gets rendered-p<N>/ with limit = depth = N and no checks.
    """
    for stale in [ctx.rendered_dir, *ctx.run_dir.glob("rendered-p*")]:
        if stale.exists():
            shutil.rmtree(stale)
    adapter, scenario = ctx.adapter, ctx.scenario
    extra_sizes = [n for n in scenario.page_sizes if n != scenario.top_k]
    unsupported: dict[str, str] = {}
    rendered = 0
    for task in ctx.tasks:
        reason = adapter.unsupported_reason(task)
        if reason:
            unsupported[task.task_id] = reason
            continue
        vector = ctx.query_vector(task)
        request = adapter.render(task, vector, scenario.top_k, depth=scenario.top_k)
        write_json(ctx.rendered_path(task), request.to_json(), indent=None)
        if task.facets:
            check = adapter.render(
                task, vector, scenario.facet_verify_limit, facets=False, depth=scenario.top_k
            )
            write_json(ctx.rendered_path(task, check=True), check.to_json(), indent=None)
        for n in extra_sizes:
            page = adapter.render(task, vector, n, depth=n)
            write_json(ctx.rendered_path(task, page_size=n), page.to_json(), indent=None)
        rendered += 1
    write_json(ctx.run_dir / "na.json", unsupported)
    return RenderResult(n_rendered=rendered, n_unsupported=len(unsupported))
