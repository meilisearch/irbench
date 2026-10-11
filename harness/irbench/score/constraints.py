from collections import Counter
from typing import Any

from irbench.filters import field_values, matches
from irbench.models import Filter, Sort

Docs = dict[str, dict[str, Any]]


def filter_violations(ids: list[str], f: Filter, docs_by_id: Docs) -> int:
    return sum(1 for d in ids if d not in docs_by_id or not matches(docs_by_id[d], f))


def _sort_value(doc: dict[str, Any] | None, field: str) -> Any:
    if doc is None:
        return None
    values = field_values(doc, field)
    return values[0] if values else None


def order_violations(ids: list[str], sort: Sort, docs_by_id: Docs) -> int:
    values = [_sort_value(docs_by_id.get(d), sort.field) for d in ids]
    violations = 0
    for a, b in zip(values, values[1:]):
        if a is None:
            violations += b is not None
        elif b is not None and (a > b if sort.direction == "asc" else a < b):
            violations += 1
    return violations


def expected_facets(ids: list[str], field: str, docs_by_id: Docs) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for d in ids:
        if d in docs_by_id:
            counts.update(str(v) for v in field_values(docs_by_id[d], field))
    return dict(counts)


def facet_accuracy(reported: dict[str, int], expected: dict[str, int]) -> float:
    keys = set(reported) | set(expected)
    if not keys:
        return 1.0
    return sum(1 for k in keys if reported.get(k, 0) == expected.get(k, 0)) / len(keys)
