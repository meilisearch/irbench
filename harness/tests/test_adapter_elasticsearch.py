import json

from adapter_helpers import SCENARIO, make_adapter, make_task
from conftest import REPO_ROOT
from irbench.engines.config import load_engine_config
from irbench.models import Filter, Sort

V = [0.1, 0.2, 0.3]
FIELDS = ["name^5", "product_class^4", "category^3", "features^2", "description^1"]


def _body(task, vector=None, limit=100, facets=True, track=None):
    return make_adapter("elasticsearch", track).render(task, vector, limit, facets).body


def test_config_loads_and_declares_nothing_unsupported():
    config = load_engine_config(REPO_ROOT, "elasticsearch", SCENARIO)
    assert config.track("default").unsupported == {}


def test_mappings_follow_the_scenario():
    mappings = make_adapter("elasticsearch").mappings()
    props = mappings["properties"]
    assert props["name"] == {"type": "text", "search_analyzer": "irbench_search"}
    assert props["product_class"]["fields"] == {"raw": {"type": "keyword"}}
    assert props["category"]["fields"] == {"raw": {"type": "keyword"}}
    assert props["average_rating"] == {"type": "float"}
    assert props["vector"] == {
        "type": "dense_vector",
        "dims": 3,
        "index": True,
        "similarity": "cosine",
    }
    assert mappings["dynamic"] is False


def test_index_body_loads_synonyms_and_takes_overrides():
    track = {"index": {"settings": {"refresh_interval": "30s"}}}
    settings = make_adapter("elasticsearch", track).index_body()["settings"]
    assert settings["number_of_replicas"] == 0 and settings["refresh_interval"] == "30s"
    syn = settings["analysis"]["filter"]["irbench_synonyms"]
    assert syn["type"] == "synonym_graph" and "grey, gray" in syn["synonyms"]
    analyzer = settings["analysis"]["analyzer"]["irbench_search"]
    assert analyzer["filter"] == ["lowercase", "irbench_synonyms"]


def test_keyword_semantic_hybrid_requests():
    kw = _body(make_task("keyword"))
    assert kw == {
        "size": 100,
        "_source": False,
        "query": {"multi_match": {"query": "grey sofa", "fields": FIELDS}},
    }
    sem = _body(make_task("semantic"), V)
    assert "query" not in sem
    assert sem["knn"] == {"field": "vector", "query_vector": V, "k": 100}
    hyb = _body(make_task("hybrid"), V)
    assert hyb["query"]["multi_match"]["query"] == "grey sofa" and hyb["knn"]["k"] == 100


def test_filters_on_keyword_and_vector_bases():
    eq = Filter(field="product_class", op="eq", value="Beds")
    kw = _body(make_task("filtered", "keyword", filter=eq))
    assert kw["query"]["bool"]["filter"] == [{"term": {"product_class.raw": "Beds"}}]
    gte = Filter(field="average_rating", op="gte", value=4.5)
    sem = _body(make_task("filtered", "semantic", filter=gte), V)
    assert sem["knn"]["filter"] == [{"range": {"average_rating": {"gte": 4.5}}}]


def test_sort_and_facets():
    sort = Sort(field="average_rating", direction="desc")
    assert _body(make_task("sorted", "keyword", sort=sort))["sort"] == [
        {"average_rating": {"order": "desc"}}
    ]
    f = make_task("faceted", "semantic", facets=["product_class"])
    assert _body(f, V)["aggs"] == {
        "product_class": {"terms": {"field": "product_class.raw", "size": 1000}}
    }
    check = _body(f, V, limit=1000, facets=False)
    assert "aggs" not in check and check["size"] == 1000 and check["knn"]["k"] == 100


def test_parse_ids_and_aggregations():
    adapter = make_adapter("elasticsearch")
    payload = {
        "hits": {"hits": [{"_id": "9"}, {"_id": "2"}]},
        "aggregations": {"product_class": {"buckets": [{"key": "Beds", "doc_count": 3}]}},
    }
    result = adapter.parse(make_task("faceted", "keyword", facets=["product_class"]), payload)
    assert result.ids == ["9", "2"] and result.facets == {"product_class": {"Beds": 3}}
    assert adapter.parse(make_task("keyword"), {"hits": {"hits": []}}).facets is None


def test_no_synonyms_creates_no_synonym_filter():
    body = make_adapter("elasticsearch", synonyms=[]).index_body()
    assert "irbench_synonyms" not in json.dumps(body)
