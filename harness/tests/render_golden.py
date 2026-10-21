"""Golden renders: every adapter's requests at (limit=top_k, depth=top_k) and its facet checks.

Regenerate ONLY when a request shape change is intended:
    uv run python harness/tests/render_golden.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from adapter_helpers import make_adapter, make_task  # noqa: E402
from irbench.models import Filter, Sort  # noqa: E402

ENGINES = ["meilisearch", "elasticsearch", "qdrant", "postgres", "postgres-paradedb", "typesense"]
GOLDEN_DIR = Path(__file__).parent / "fixtures" / "render_golden"
V = [0.1, 0.2, 0.3]


def cases():
    eq = Filter(field="product_class", op="eq", value="Beds")
    sort = Sort(field="average_rating", direction="desc")
    for base in ("keyword", "semantic", "hybrid"):
        yield f"{base}", make_task(base)
    for base in ("keyword", "semantic"):
        yield f"filtered-{base}", make_task("filtered", base, filter=eq)
        yield f"sorted-{base}", make_task("sorted", base, sort=sort)
        yield f"faceted-{base}", make_task("faceted", base, facets=["product_class"])


def render_all(engine: str, **kwargs) -> dict:
    adapter = make_adapter(engine, n_docs=1000)
    top_k = adapter.ctx.scenario.top_k
    verify = adapter.ctx.scenario.facet_verify_limit
    out = {}
    for name, task in cases():
        vector = None if task.base == "keyword" else V
        for label, limit, facets in (("main", top_k, True), ("check", verify, False)):
            if label == "check" and not task.facets:
                continue
            try:
                req = adapter.render(task, vector, limit, facets, **kwargs)
            except ValueError as e:  # e.g. Qdrant does not render faceted tasks
                out[f"{name}/{label}"] = {"error": str(e)}
                continue
            out[f"{name}/{label}"] = req.to_json()
    return out


if __name__ == "__main__":
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for engine in ENGINES:
        path = GOLDEN_DIR / f"{engine}.json"
        path.write_text(json.dumps(render_all(engine), indent=1, sort_keys=True) + "\n")
        print(f"wrote {path}")
