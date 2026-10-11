from typing import Any

from irbench.models import Filter


def field_values(doc: dict[str, Any], field: str) -> list[Any]:
    value = doc.get(field)
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def matches(doc: dict[str, Any], f: Filter) -> bool:
    values = field_values(doc, f.field)
    if f.op == "eq":
        return f.value in values
    numeric = [v for v in values if isinstance(v, int | float)]
    if f.op == "gte":
        return any(v >= f.value for v in numeric)
    return any(v <= f.value for v in numeric)
