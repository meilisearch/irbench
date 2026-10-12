import pytest

from adapter_helpers import SCENARIO, make_task
from conftest import REPO_ROOT
from irbench.engines.base import IndexContext, SqlRequest, load_adapter_class
from irbench.engines.config import load_engine_config
from irbench.models import Filter, Sort
from irbench.scenario import load_scenario
from irbench.terms import load_term_groups

ENGINE = "postgres"


@pytest.fixture
def adapter():
    scenario = load_scenario(REPO_ROOT, SCENARIO)
    cls = load_adapter_class(REPO_ROOT, ENGINE)
    config = load_engine_config(REPO_ROOT, ENGINE, SCENARIO).track("default")
    return cls(
        IndexContext(
            scenario=scenario,
            config=config,
            index="irbench_t",
            dim=3,
            synonyms=load_term_groups(scenario.synonyms_path),
            n_docs=1000,
        )
    )


def placeholders_are_text(r: SqlRequest) -> bool:
    import re

    return all(m.group(1) for m in re.finditer(r"\$\d+(::text)?", r.sql))


def test_keyword_request(adapter):
    r = adapter.render(make_task("keyword", text="grey sofa"), None, 100)
    assert isinstance(r, SqlRequest) and placeholders_are_text(r)
    assert "websearch_to_tsquery('irbench_ecommerce_product_search', $1::text)" in r.sql
    assert "ts_rank(" in r.sql and "LIMIT $2::text::int" in r.sql
    assert r.params == ["grey sofa", 100]


def test_query_text_is_a_parameter_not_sql(adapter):
    text = "sofa'; DROP TABLE docs; --"
    r = adapter.render(make_task("keyword", text=text), None, 100)
    assert text not in r.sql and r.params[0] == text


def test_semantic_request(adapter):
    r = adapter.render(make_task("semantic"), [0.1, 0.2, 0.3], 100)
    assert "embedding <=> $1::text::vector" in r.sql
    assert r.params == [[0.1, 0.2, 0.3], 100]


def test_hybrid_is_rrf_in_one_statement(adapter):
    r = adapter.render(make_task("hybrid", text="sofa"), [0.1, 0.2, 0.3], 100)
    assert "FULL OUTER JOIN" in r.sql and "60 +" in r.sql
    assert r.params[:2] == ["sofa", [0.1, 0.2, 0.3]]


def test_filters(adapter):
    eq = adapter.render(
        make_task(
            "filtered",
            "keyword",
            text="sofa",
            filter=Filter(field="product_class", op="eq", value="Sofas"),
        ),
        None,
        100,
    )
    assert "product_class__terms @> ARRAY[$" in eq.sql and "Sofas" in eq.params
    rng = adapter.render(
        make_task(
            "filtered",
            "keyword",
            text="sofa",
            filter=Filter(field="average_rating", op="gte", value=4.0),
        ),
        None,
        100,
    )
    assert "average_rating >= $" in rng.sql and "::text::float8" in rng.sql and 4.0 in rng.params


def test_sorted_keyword_sorts_every_match(adapter):
    r = adapter.render(
        make_task(
            "sorted", "keyword", text="sofa", sort=Sort(field="average_rating", direction="desc")
        ),
        None,
        100,
    )
    assert "ORDER BY average_rating DESC NULLS LAST, id" in r.sql
    assert "embedding" not in r.sql


def test_sorted_semantic_sorts_top_k(adapter):
    r = adapter.render(
        make_task("sorted", "semantic", sort=Sort(field="review_count", direction="asc")),
        [0.1, 0.2, 0.3],
        100,
    )
    assert "ORDER BY review_count ASC NULLS LAST, id" in r.sql and "embedding <=>" in r.sql


def test_faceted_returns_one_row_with_ids_and_facets(adapter):
    r = adapter.render(
        make_task("faceted", "keyword", text="sofa", facets=["product_class"]), None, 100
    )
    assert "json_agg" in r.sql and "unnest(product_class__terms)" in r.sql
    found = adapter.parse(
        make_task("faceted", "keyword", text="sofa", facets=["product_class"]),
        [{"ids": ["2", "1"], "facets": {"product_class": {"Sofas": 2}}}],
    )
    assert found.ids == ["2", "1"] and found.facets == {"product_class": {"Sofas": 2}}


def test_facet_check_fetch_is_a_plain_query(adapter):
    task = make_task("faceted", "keyword", text="sofa", facets=["product_class"])
    r = adapter.render(task, None, 1000, facets=False)
    assert "json_agg" not in r.sql and r.params[-1] == 1000
    assert adapter.parse(task, [{"id": "1"}, {"id": "2"}]).ids == ["1", "2"]


def test_synonym_file_matches_scenario_synonyms(adapter):
    path = REPO_ROOT / "engines/postgres/docker/tsearch/irbench_ecommerce_product_search.syn"
    expected = adapter.synonym_file()
    assert path.read_text(encoding="utf-8") == expected, (
        f"{path} is out of date; replace its content with:\n{expected}"
    )


def test_synonym_file_skips_multiword_groups(adapter):
    text = adapter.synonym_file()
    assert "gray grey" in text.splitlines()
    assert not any(" seater" in line for line in text.splitlines())
