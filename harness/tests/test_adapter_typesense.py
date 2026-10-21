import pytest

from adapter_helpers import SCENARIO, make_adapter, make_task
from conftest import REPO_ROOT
from irbench.engines.base import EngineError
from irbench.engines.config import load_engine_config
from irbench.models import Filter, Sort

V = [0.1, 0.2, 0.3]
QUERY_BY = "name,product_class,category,features,description"


def _search(task, vector=None, limit=100, facets=True, track=None, depth=None):
    body = make_adapter("typesense", track).render(task, vector, limit, facets, depth=depth).body
    assert len(body["searches"]) == 1
    return body["searches"][0]


def test_config_loads_and_declares_nothing_unsupported():
    config = load_engine_config(REPO_ROOT, "typesense", SCENARIO)
    assert config.track("default").unsupported == {}


def test_collection_follows_the_scenario():
    adapter = make_adapter("typesense")
    fields = {f["name"]: f for f in adapter.collection_body()["fields"]}
    assert fields["name"] == {"name": "name", "type": "string*", "optional": True, "facet": False}
    assert fields["product_class"]["facet"] is True
    # Searchable fields keep Typesense's tokenization; exact-only fields keep their symbols.
    assert "symbols_to_index" not in fields["product_class"]
    assert fields["average_rating"] == {
        "name": "average_rating",
        "type": "float",
        "optional": True,
        "facet": False,
    }
    assert fields["vector"] == {"name": "vector", "type": "float[]", "num_dim": 3}
    assert adapter.collection_body()["synonym_sets"] == ["idx_synonyms"]
    groups = adapter.synonyms_body()["items"]
    assert {"id": "g0", "synonyms": groups[0]["synonyms"]} == groups[0]
    assert any(set(g["synonyms"]) == {"grey", "gray"} for g in groups)


def test_index_options_reach_searchable_and_named_fields():
    track = {
        "index": {
            "searchable_fields": {"stem": True},
            "fields": {"vector": {"hnsw_params": {"M": 32}}},
            "token_separators": ["-"],
        }
    }
    body = make_adapter("typesense", track).collection_body()
    fields = {f["name"]: f for f in body["fields"]}
    assert fields["description"]["stem"] is True and "stem" not in fields["average_rating"]
    assert fields["vector"]["hnsw_params"] == {"M": 32}
    assert body["token_separators"] == ["-"]
    with pytest.raises(ValueError, match="unknown fields"):
        make_adapter("typesense", {"index": {"fields": {"nope": {}}}}).collection_body()


def test_exact_match_fields_keep_symbols():
    adapter = make_adapter("typesense")
    adapter.ctx.scenario.index_fields.filterable.append("tags")
    tags = {f["name"]: f for f in adapter.collection_body()["fields"]}["tags"]
    assert {"-", "#", ".", "&"} <= set(tags["symbols_to_index"])
    assert "`" not in tags["symbols_to_index"]


def test_no_synonyms_means_no_synonym_set():
    assert "synonym_sets" not in make_adapter("typesense", synonyms=[]).collection_body()


def test_keyword_semantic_hybrid_requests():
    kw = _search(make_task("keyword"))
    assert kw == {
        "collection": "idx",
        "include_fields": "id",
        "q": "grey sofa",
        "query_by": QUERY_BY,
        "query_by_weights": "5,4,3,2,1",
        "per_page": 100,
    }
    sem = _search(make_task("semantic"), V)
    assert sem["q"] == "*" and "query_by" not in sem
    assert sem["vector_query"] == "vector:([0.1,0.2,0.3], k:100)"
    hyb = _search(make_task("hybrid"), V, limit=20, depth=20)
    assert hyb["q"] == "grey sofa" and hyb["vector_query"].endswith("k:20)")
    assert hyb["per_page"] == 20


def test_filters_sort_and_facets():
    eq = Filter(field="product_class", op="eq", value="Beds")
    assert _search(make_task("filtered", "keyword", filter=eq))["filter_by"] == (
        "product_class:=`Beds`"
    )
    gte = Filter(field="average_rating", op="gte", value=4.5)
    assert _search(make_task("filtered", "semantic", filter=gte), V)["filter_by"] == (
        "average_rating:>=4.5"
    )
    sort = Sort(field="average_rating", direction="desc")
    assert _search(make_task("sorted", "keyword", sort=sort))["sort_by"] == (
        "average_rating(missing_values: last):desc"
    )
    f = _search(make_task("faceted", "keyword", facets=["product_class"]))
    assert f["facet_by"] == "product_class" and f["max_facet_values"] == 1000


def test_backtick_in_a_filter_value_is_refused():
    bad = Filter(field="product_class", op="eq", value="a`b")
    with pytest.raises(ValueError, match="backtick"):
        _search(make_task("filtered", "keyword", filter=bad))


def test_fetches_above_250_hits_span_pages_in_one_request():
    task = make_task("faceted", "keyword", facets=["product_class"])
    body = make_adapter("typesense").render(task, None, 1000, False).body
    pages = body["searches"]
    assert [s["page"] for s in pages] == [1, 2, 3, 4]
    assert {s["per_page"] for s in pages} == {250} and "facet_by" not in pages[0]


def test_search_params_merge_per_base():
    track = {"search": {"keyword": {"drop_tokens_threshold": 0}}}
    assert _search(make_task("keyword"), track=track)["drop_tokens_threshold"] == 0
    assert "drop_tokens_threshold" not in _search(make_task("semantic"), V, track=track)


def test_vector_query_options_go_inside_the_vector_query():
    track = {"search": {"hybrid": {"vector_query_options": {"alpha": 0.9}, "num_typos": 1}}}
    hyb = _search(make_task("hybrid"), V, track=track)
    assert hyb["vector_query"] == "vector:([0.1,0.2,0.3], k:100, alpha:0.9)"
    assert hyb["num_typos"] == 1 and "vector_query_options" not in hyb


def test_parse_concatenates_pages_and_reads_facets():
    adapter = make_adapter("typesense")
    payload = {
        "results": [
            {
                "hits": [{"document": {"id": "9"}}, {"document": {"id": "2"}}],
                "facet_counts": [
                    {"field_name": "product_class", "counts": [{"value": "Beds", "count": 3}]}
                ],
            },
            {"hits": [{"document": {"id": "4"}}]},
        ]
    }
    result = adapter.parse(make_task("faceted", "keyword", facets=["product_class"]), payload)
    assert result.ids == ["9", "2", "4"] and result.facets == {"product_class": {"Beds": 3}}
    assert adapter.parse(make_task("keyword"), payload).facets is None


def test_parse_raises_on_an_error_inside_a_200_response():
    payload = {"results": [{"code": 422, "error": "Only upto 250 hits can be fetched per page."}]}
    with pytest.raises(EngineError, match="422"):
        make_adapter("typesense").parse(make_task("keyword"), payload)
