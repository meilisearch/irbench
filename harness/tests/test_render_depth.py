import json

import pytest
from render_golden import ENGINES, GOLDEN_DIR, V, render_all

from adapter_helpers import make_adapter, make_task


@pytest.mark.parametrize("engine", ENGINES)
def test_requests_at_top_k_are_unchanged(engine):
    golden = json.loads((GOLDEN_DIR / f"{engine}.json").read_text())
    assert render_all(engine) == golden
    assert render_all(engine, depth=100) == golden


def _body(engine, task, limit, depth, vector=V):
    return make_adapter(engine, n_docs=1000).render(task, vector, limit, depth=depth).body


def test_meilisearch_page_of_20():
    assert _body("meilisearch", make_task("hybrid"), 20, 20)["limit"] == 20


def test_elasticsearch_knn_k_follows_depth():
    body = _body("elasticsearch", make_task("hybrid"), 20, 20)
    assert body["size"] == 20 and body["knn"]["k"] == 20


def test_qdrant_pools_follow_depth():
    hyb = _body("qdrant", make_task("hybrid"), 20, 20)
    assert hyb["limit"] == 20
    keyword_side, dense_side = hyb["prefetch"]
    assert dense_side["limit"] == 20 and keyword_side["limit"] == 20
    assert {leaf["limit"] for leaf in keyword_side["prefetch"]} == {20}
    kw = _body("qdrant", make_task("keyword"), 20, 20, vector=None)
    assert {leaf["limit"] for leaf in kw["prefetch"]} == {20}
    from irbench.models import Sort

    sort = Sort(field="average_rating", direction="desc")
    sem_sorted = _body("qdrant", make_task("sorted", "semantic", sort=sort), 20, 20)
    assert sem_sorted["prefetch"]["limit"] == 20


@pytest.mark.parametrize("engine", ["postgres", "postgres-paradedb"])
def test_postgres_hybrid_pools_follow_depth(engine):
    req = make_adapter(engine, n_docs=1000).render(make_task("hybrid"), V, 20, depth=20)
    ints = [p for p in req.params if isinstance(p, int)]
    # the RRF side limits (depth) and the outer limit are 20; no top_k (100) left anywhere
    assert 20 in ints and 100 not in ints


@pytest.mark.parametrize("engine", ["postgres", "postgres-paradedb"])
def test_postgres_semantic_sorted_pool_follows_depth(engine):
    from irbench.models import Sort

    sort = Sort(field="average_rating", direction="desc")
    task = make_task("sorted", "semantic", sort=sort)
    req = make_adapter(engine, n_docs=1000).render(task, V, 20, depth=20)
    ints = [p for p in req.params if isinstance(p, int)]
    assert 20 in ints and 100 not in ints
