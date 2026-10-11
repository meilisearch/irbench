import importlib.util
import uuid

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


def test_config_declares_what_the_adapter_cannot_do():
    track = load_engine_config(REPO_ROOT, "qdrant", SCENARIO).track("default")
    assert track.unsupported_reason(make_task("keyword"))
    assert track.unsupported_reason(make_task("filtered", "keyword"))
    assert track.unsupported_reason(make_task("hybrid"))
    assert track.unsupported_reason(make_task("faceted", "semantic", facets=["product_class"]))
    assert track.unsupported_reason(make_task("semantic")) is None
    assert track.unsupported_reason(make_task("filtered", "semantic")) is None
    assert track.unsupported_reason(make_task("sorted", "semantic")) is None


def test_collection_and_payload_indexes():
    adapter = make_adapter("qdrant")
    assert adapter.collection_body() == {"vectors": {"size": 3, "distance": "Cosine"}}
    assert adapter.payload_indexes() == {
        "average_rating": "float",
        "category": "keyword",
        "product_class": "keyword",
        "review_count": "float",
    }


def test_semantic_and_filtered_requests():
    assert _body(make_task("semantic")) == {"query": V, "limit": 100, "with_payload": ["id"]}
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
        "prefetch": {"query": V, "limit": 100},
        "query": {"order_by": {"key": "review_count", "direction": "desc"}},
        "limit": 100,
        "with_payload": ["id"],
    }


def test_keyword_tasks_are_refused():
    with pytest.raises(ValueError, match="semantic"):
        _body(make_task("keyword"), None)


def test_parse_returns_original_ids_from_payload():
    payload = {"result": {"points": [{"id": 7, "score": 0.9, "payload": {"id": "7"}}]}}
    assert make_adapter("qdrant").parse(make_task("semantic"), payload).ids == ["7"]
