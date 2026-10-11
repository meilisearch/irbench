import shutil
from dataclasses import dataclass

from irbench.io import write_json
from irbench.stages.context import RunContext


@dataclass(frozen=True)
class RenderResult:
    n_rendered: int
    n_unsupported: int


def render_stage(ctx: RunContext) -> RenderResult:
    """Write every supported task as a request file; unsupported ones go to na.json."""
    if ctx.rendered_dir.exists():
        shutil.rmtree(ctx.rendered_dir)
    adapter, scenario = ctx.adapter, ctx.scenario
    unsupported: dict[str, str] = {}
    rendered = 0
    for task in ctx.tasks:
        reason = adapter.unsupported_reason(task)
        if reason:
            unsupported[task.task_id] = reason
            continue
        vector = ctx.query_vector(task)
        request = adapter.render(task, vector, scenario.top_k)
        write_json(ctx.rendered_path(task), request.to_json(), indent=None)
        if task.facets:
            check = adapter.render(task, vector, scenario.facet_verify_limit, facets=False)
            write_json(ctx.rendered_path(task, check=True), check.to_json(), indent=None)
        rendered += 1
    write_json(ctx.run_dir / "na.json", unsupported)
    return RenderResult(n_rendered=rendered, n_unsupported=len(unsupported))
