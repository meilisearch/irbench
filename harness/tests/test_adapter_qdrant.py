import importlib.util
import uuid

import numpy as np
import pytest

from adapter_helpers import SCENARIO, make_adapter, make_task
from conftest import REPO_ROOT
from irbench.engines.config import load_engine_config
from irbench.models import Filter, Sort

V = [0.1, 0.2, 0.3]


def _load_point_id():
    path = REPO_ROOT / "engines" / "qdrant" / "adapter.py"
    spec = importlib.util.spec_from_file_location("qdrant_adapter_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.point_id


def _body(task, vector=V, limit=100, facets=True):
    return make_adapter("qdrant").render(task, vector, limit, facets).body


def test_point_ids():
    point_id = _load_point_id()
    assert point_id("42") == 42
    assert point_id("sku-1") == str(uuid.uuid5(uuid.NAMESPACE_URL, "sku-1"))


def test_config_declares_only_what_the_adapter_cannot_do():
    track = load_engine_config(REPO_ROOT, "qdrant", SCENARIO).track("default")
    assert track.unsupported_reason(make_task("faceted", "semantic", facets=["product_class"]))
    assert track.unsupported_reason(make_task("faceted", "keyword", facets=["product_class"]))
    for mode, base in [
        ("keyword", "keyword"),
        ("semantic", "semantic"),
        ("hybrid", "hybrid"),
        ("filtered", "keyword"),
        ("filtered", "semantic"),
        ("sorted", "keyword"),
        ("sorted", "semantic"),
    ]:
        assert track.unsupported_reason(make_task(mode, base)) is None, (mode, base)
    assert list(track.unsupported) == ["faceted"]
    assert track.synonyms


def test_collection_and_payload_indexes():
    adapter = make_adapter("qdrant")
    assert adapter.collection_body() == {
        "vectors": {"dense": {"size": 3, "distance": "Cosine"}},
        "sparse_vectors": {
            name: {"modifier": "idf"}
            for name in ["name", "product_class", "category", "features", "description"]
        },
    }
    assert adapter.payload_indexes() == {
        "average_rating": "float",
        "category": "keyword",
        "product_class": "keyword",
        "review_count": "float",
    }


def test_index_config_is_merged_into_the_collection():
    body = make_adapter("qdrant", {"index": {"hnsw_config": {"m": 32}}}).collection_body()
    assert body["hnsw_config"] == {"m": 32}


def test_points_carry_the_dense_vector_and_bm25_text_per_field():
    adapter = make_adapter("qdrant")
    sent = []

    class Client:
        def request(self, method, path, **kwargs):
            sent.append((method, path, kwargs["json"]))
            return type("R", (), {"status_code": 200, "content": b"{}", "json": lambda s: {}})()

    doc = {
        "id": "7",
        "name": "Oak table",
        "product_class": "Tables",
        "category": "",
        "features": ["solid", "oak"],
        "average_rating": 4.5,
    }
    adapter.load_documents(Client(), [doc], np.array([V]))
    ((method, path, body),) = sent
    assert (method, path) == ("PUT", "/collections/idx/points?wait=true")
    (point,) = body["points"]
    assert point["id"] == 7 and point["payload"] == doc
    bm25 = lambda text: {"text": text, "model": "qdrant/bm25"}  # noqa: E731
    assert point["vector"] == {
        "dense": V,
        "name": bm25("Oak table"),
        "product_class": bm25("Tables"),
        "features": bm25("solid, oak"),
    }


def _formula(limit=100, filter=None):
    names = ["name", "product_class", "category", "features", "description"]
    leaves = []
    for name in names:
        leaf = {
            "query": {"text": "grey sofa", "model": "qdrant/bm25"},
            "using": name,
            "limit": 100,
        }
        if filter:
            leaf["filter"] = filter
        leaves.append(leaf)
    weights = [5, 4, 3, 2, 1]
    return {
        "prefetch": leaves,
        "query": {
            "formula": {
                "sum": [{"mult": [w, f"$score[{i}]"]} for i, w in enumerate(weights)],
            },
            "defaults": {f"$score[{i}]": 0 for i in range(5)},
        },
        "limit": limit,
    }


def test_semantic_and_filtered_requests():
    assert _body(make_task("semantic")) == {
        "query": V,
        "using": "dense",
        "limit": 100,
        "with_payload": ["id"],
    }
    eq = Filter(field="product_class", op="eq", value="Beds")
    assert _body(make_task("filtered", "semantic", filter=eq))["filter"] == {
        "must": [{"key": "product_class", "match": {"value": "Beds"}}]
    }
    gte = Filter(field="average_rating", op="gte", value=4)
    assert _body(make_task("filtered", "semantic", filter=gte))["filter"] == {
        "must": [{"key": "average_rating", "range": {"gte": 4.0}}]
    }


def test_sorted_request_reorders_the_nearest_neighbours():
    s = make_task("sorted", "semantic", sort=Sort(field="review_count", direction="desc"))
    assert _body(s) == {
        "prefetch": {"query": V, "using": "dense", "limit": 100},
        "query": {"formula": "review_count", "defaults": {"review_count": -1e18}},
        "limit": 100,
        "with_payload": ["id"],
    }


def test_ascending_sort_negates_and_missing_values_still_sort_last():
    s = make_task("sorted", "semantic", sort=Sort(field="average_rating", direction="asc"))
    assert _body(s)["query"] == {
        "formula": {"neg": "average_rating"},
        "defaults": {"average_rating": 1e18},
    }


def test_keyword_request_weights_fields_by_priority():
    assert _body(make_task("keyword"), None) == {**_formula(), "with_payload": ["id"]}


def test_hybrid_request_fuses_keyword_and_dense_natively():
    body = _body(make_task("hybrid"))
    assert body == {
        "prefetch": [
            _formula(),
            {"query": V, "using": "dense", "limit": 100},
        ],
        "query": {"fusion": "rrf"},
        "limit": 100,
        "with_payload": ["id"],
    }


def test_filtered_keyword_filters_every_leaf_and_the_top_level():
    eq = Filter(field="product_class", op="eq", value="Beds")
    f = {"must": [{"key": "product_class", "match": {"value": "Beds"}}]}
    body = _body(make_task("filtered", "keyword", filter=eq), None)
    assert body == {**_formula(filter=f), "filter": f, "with_payload": ["id"]}


def test_filtered_hybrid_filters_both_branches():
    eq = Filter(field="product_class", op="eq", value="Beds")
    f = {"must": [{"key": "product_class", "match": {"value": "Beds"}}]}
    body = _body(make_task("filtered", "hybrid", filter=eq))
    assert body["filter"] == f
    keyword, dense = body["prefetch"]
    assert dense["filter"] == f
    assert all(leaf["filter"] == f for leaf in keyword["prefetch"])


def test_sorted_keyword_prefetches_the_keyword_ranking_then_orders():
    s = make_task("sorted", "keyword", sort=Sort(field="review_count", direction="desc"))
    assert _body(s, None) == {
        "prefetch": _formula(),
        "query": {"formula": "review_count", "defaults": {"review_count": -1e18}},
        "limit": 100,
        "with_payload": ["id"],
    }


def test_search_params_are_merged_on_every_base():
    track = {"search": {"keyword": {"params": {"x": 1}}, "hybrid": {"params": {"y": 2}}}}
    adapter = make_adapter("qdrant", track)
    assert adapter.render(make_task("keyword"), None, 100).body["params"] == {"x": 1}
    assert adapter.render(make_task("hybrid"), V, 100).body["params"] == {"y": 2}


def test_faceted_tasks_are_refused():
    with pytest.raises(ValueError, match="faceted"):
        _body(make_task("faceted", "keyword", facets=["product_class"]), None)
    _body(make_task("faceted", "keyword", facets=["product_class"]), None, facets=False)


def test_parse_returns_original_ids_from_payload():
    payload = {"result": {"points": [{"id": 7, "score": 0.9, "payload": {"id": "7"}}]}}
    assert make_adapter("qdrant").parse(make_task("semantic"), payload).ids == ["7"]
