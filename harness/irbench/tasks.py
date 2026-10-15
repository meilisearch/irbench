import random
from dataclasses import dataclass, field
from typing import Any

from irbench.filters import field_values, matches
from irbench.models import BASE_MODES, Filter, Qrels, Query, QueryTask, Sort
from irbench.scenario import FilterTemplate, Scenario


@dataclass
class TaskSet:
    tasks: list[QueryTask] = field(default_factory=list)
    qrels: Qrels = field(default_factory=dict)
    skipped_queries: list[str] = field(default_factory=list)


def _choose_filter(
    templates: list[FilterTemplate], positives: list[dict[str, Any]], rng: random.Random
) -> Filter | None:
    candidates: list[Filter] = []
    for t in templates:
        if t.op == "eq":
            values = sorted({str(v) for d in positives for v in field_values(d, t.field)})
            candidates += [Filter(field=t.field, op="eq", value=v) for v in values]
        else:
            for v in t.values or []:
                f = Filter(field=t.field, op=t.op, value=float(v))
                if any(matches(d, f) for d in positives):
                    candidates.append(f)
    return rng.choice(candidates) if candidates else None


def build_tasks(
    scenario: Scenario, variants: list[Query], qrels: Qrels, docs_by_id: dict[str, dict[str, Any]]
) -> TaskSet:
    seed = scenario.perturbations.seed
    result = TaskSet()
    filters: dict[str, Filter | None] = {}
    sorts: dict[str, Sort] = {}

    def add(query: Query, mode: str, base: str, grades: dict[str, int], **extra: Any) -> None:
        task = QueryTask(
            task_id=f"{query.qid}:{mode}:{base}",
            qid=query.qid,
            variant_of=query.variant_of,
            perturbation=query.perturbation,
            text=query.text,
            mode=mode,
            base=base,
            group=query.group,
            **extra,
        )
        result.tasks.append(task)
        result.qrels[task.task_id] = grades

    for query in variants:
        source = query.variant_of
        grades = qrels.get(source, {})
        positive_ids = [d for d, g in grades.items() if g > 0]
        if not positive_ids:
            if query.perturbation == "original":
                result.skipped_queries.append(source)
            continue
        positives = [docs_by_id[d] for d in sorted(positive_ids) if d in docs_by_id]

        for mode in scenario.modes:
            if mode in BASE_MODES:
                add(query, mode, mode, dict(grades))
                continue
            for base in scenario.structured_bases:
                if mode == "filtered":
                    if source not in filters:
                        rng = random.Random(f"{seed}:{source}:filter")
                        filters[source] = _choose_filter(scenario.filter_templates, positives, rng)
                    f = filters[source]
                    if f is None:
                        continue
                    kept = {
                        d: g
                        for d, g in grades.items()
                        if d in docs_by_id and matches(docs_by_id[d], f)
                    }
                    add(query, mode, base, kept, filter=f)
                elif mode == "sorted":
                    if source not in sorts:
                        rng = random.Random(f"{seed}:{source}:sort")
                        sorts[source] = rng.choice(scenario.sort_templates)
                    add(query, mode, base, dict(grades), sort=sorts[source])
                else:
                    add(query, mode, base, dict(grades), facets=list(scenario.index_fields.facets))
    return result
