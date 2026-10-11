import pytest

from adapter_helpers import SCENARIO, make_adapter, make_task
from conftest import REPO_ROOT
from irbench.engines.config import load_engine_config
from irbench.models import Filter, Sort

V = [0.1, 0.2, 0.3]
SEARCHABLE = ["name", "product_class", "category", "features", "description"]


def _body(task, vector=None, limit=100, facets=True, track=None):
    return make_adapter("meilisearch", track).render(task, vector, limit, facets).body


def test_config_sorts_strictly_and_declares_nothing_unsupported():
    track = load_engine_config(REPO_ROOT, "meilisearch", SCENARIO).track("default")
    assert track.unsupported == {}
    assert track.index["rankingRules"][0] == "sort"
    assert "rankingRules" in track.justification


def test_index_settings_follow_the_scenario():
    s = make_adapter("meilisearch").index_settings()
    assert s["searchableAttributes"] == SEARCHABLE
    assert s["filterableAttributes"] == [
        "average_rating",
        "category",
        "product_class",
        "review_count",
    ]
    assert s["sortableAttributes"] == ["average_rating", "review_count"]
    assert s["synonyms"]["grey"] == ["gray"] and s["synonyms"]["3 seater"] == ["three seater"]
    assert s["faceting"] == {"maxValuesPerFacet": 1000}
    assert s["embedders"] == {"default": {"source": "userProvided", "dimensions": 3}}


def test_index_settings_take_config_overrides():
    track = {"index": {"typoTolerance": {"enabled": False}}}
    s = make_adapter("meilisearch", track).index_settings()
    assert s["typoTolerance"] == {"enabled": False}
    assert s["searchableAttributes"][0] == "name"


def test_keyword_semantic_hybrid_requests():
    kw = _body(make_task("keyword"))
    assert kw == {"q": "grey sofa", "limit": 100, "attributesToRetrieve": ["id"]}
    sem = _body(make_task("semantic"), V)
    assert "q" not in sem
    assert sem["vector"] == V and sem["hybrid"] == {"embedder": "default", "semanticRatio": 1.0}
    hyb = _body(make_task("hybrid"), V)
    assert hyb["q"] == "grey sofa" and hyb["vector"] == V
    assert hyb["hybrid"] == {"embedder": "default"}
    with pytest.raises(ValueError, match="query vector"):
        _body(make_task("semantic"), None)


def test_filter_expressions_quote_strings_and_format_numbers():
    f = Filter(field="product_class", op="eq", value='Kid\'s "Beds"')
    assert _body(make_task("filtered", "keyword", filter=f))["filter"] == (
        'product_class = "Kid\'s \\"Beds\\""'
    )
    f = Filter(field="average_rating", op="gte", value=4)
    assert _body(make_task("filtered", "semantic", filter=f), V)["filter"] == (
        "average_rating >= 4.0"
    )


def test_sort_and_facets():
    s = make_task("sorted", "keyword", sort=Sort(field="review_count", direction="desc"))
    assert _body(s)["sort"] == ["review_count:desc"]
    f = make_task("faceted", "keyword", facets=["product_class"])
    assert _body(f)["facets"] == ["product_class"]
    check = _body(f, limit=1000, facets=False)
    assert "facets" not in check and check["limit"] == 1000


def test_search_overrides_apply_to_their_base_only():
    track = {"search": {"hybrid": {"hybrid": {"semanticRatio": 0.7}}}}
    hybrid = _body(make_task("hybrid"), V, track=track)["hybrid"]
    assert hybrid == {"embedder": "default", "semanticRatio": 0.7}
    assert "hybrid" not in _body(make_task("keyword"), track=track)


def test_parse_ids_and_facets():
    adapter = make_adapter("meilisearch")
    payload = {
        "hits": [{"id": "7"}, {"id": "3"}],
        "facetDistribution": {"product_class": {"Beds": 2}},
    }
    result = adapter.parse(make_task("faceted", "keyword", facets=["product_class"]), payload)
    assert result.ids == ["7", "3"] and result.facets == {"product_class": {"Beds": 2}}
