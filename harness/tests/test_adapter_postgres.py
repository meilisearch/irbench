import pytest

from adapter_helpers import SCENARIO, make_adapter, make_task
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


def test_thesaurus_file_matches_scenario_synonyms(adapter):
    path = REPO_ROOT / "engines/postgres/docker/tsearch/irbench_ecommerce_product_search.ths"
    expected = adapter.thesaurus_file()
    assert path.read_text(encoding="utf-8") == expected, (
        f"{path} is out of date; replace its content with:\n{expected}"
    )


def test_thesaurus_maps_every_member_to_the_first(adapter):
    lines = adapter.thesaurus_file().splitlines()
    assert "gray : grey" in lines and "children : kids" in lines
    assert not any("seater" in line or "drawer" in line for line in lines)  # digit groups dropped


def test_ddl_maps_hyphenated_tokens_through_the_thesaurus(adapter):
    ddl = adapter.setup_ddl()
    assert "TEMPLATE = thesaurus" in ddl and "Dictionary = english_stem" in ddl
    for token in ("asciihword", "hword_asciipart", "hword_part", "hword"):
        assert token in ddl


def test_settings_reset_then_apply_on_current_database(adapter):
    class Recorder:
        sql = ""

        def script(self, sql):
            self.sql = sql

    rec = Recorder()
    adapter.apply_settings(rec)
    assert "RESET ALL" in rec.sql and "current_database()" in rec.sql
    assert "irbench" not in rec.sql.replace("current_database", "")
    assert "SET hnsw.ef_search = 100;" in rec.sql


def test_faceted_and_sorted_never_carry_wide_columns(adapter):
    v = [0.1, 0.2, 0.3]
    asc = Sort(field="average_rating", direction="asc")
    cases = [
        (make_task("faceted", "keyword", text="sofa", facets=["product_class"]), None),
        (make_task("faceted", "semantic", facets=["product_class"]), v),
        (make_task("faceted", "hybrid", text="sofa", facets=["product_class"]), v),
        (make_task("sorted", "keyword", text="sofa", sort=asc), None),
        (make_task("sorted", "semantic", sort=asc), v),
        (make_task("sorted", "hybrid", text="sofa", sort=asc), v),
    ]
    for task, vec in cases:
        sql = adapter.render(task, vec, 100).sql
        assert "SELECT *" not in sql and "d.*" not in sql and ", tsv" not in sql, sql
        assert "SELECT embedding" not in sql, sql


def test_sorted_keyword_has_no_inner_sort(adapter):
    sort = Sort(field="average_rating", direction="desc")
    r = adapter.render(make_task("sorted", "keyword", text="sofa", sort=sort), None, 100)
    assert r.sql.count("ORDER BY") == 1 and r.params == ["sofa", 100]


def test_faceted_semantic_counts_without_order_or_limit(adapter):
    task = make_task("faceted", "semantic", facets=["product_class"])
    r = adapter.render(task, [0.1, 0.2, 0.3], 100)
    m = r.sql.split(", hits AS")[0]
    assert "ORDER BY" not in m and "LIMIT" not in m
    assert "ORDER BY embedding <=> $1::text::vector" in r.sql
    assert r.params == [[0.1, 0.2, 0.3], 100, adapter.ctx.scenario.max_facet_values]


def _settings_script(adapter, settings):
    class Recorder:
        sql = ""

        def script(self, sql):
            self.sql = sql

    adapter.ctx.config.index["settings"] = settings
    rec = Recorder()
    adapter.apply_settings(rec)
    return rec.sql


def test_settings_accept_core_names(adapter):
    sql = _settings_script(adapter, {"plan_cache_mode": "force_custom_plan"})
    assert "SET plan_cache_mode = force_custom_plan;" in sql


@pytest.mark.parametrize(
    "name,value",
    [
        ("plan_cache_mode; DROP TABLE x", "on"),
        ("Plan_Cache", "on"),
        ("a.b.c", "1"),
        ("plan_cache_mode", "x; DROP TABLE y"),
        ("plan_cache_mode", "a b"),
        ("hnsw.ef_search", "1'; --"),
    ],
)
def test_settings_reject_unsafe_names_and_values(adapter, name, value):
    with pytest.raises(ValueError, match="unsafe setting"):
        _settings_script(adapter, {name: value})


def test_facet_buckets_are_capped_at_the_scenario_max(adapter):
    cap = adapter.ctx.scenario.max_facet_values
    task = make_task("faceted", "keyword", text="sofa", facets=["product_class", "category"])
    r = adapter.render(task, None, 100)
    n = r.params.index(cap) + 1
    assert r.params.count(cap) == 1
    assert r.sql.count(f"ORDER BY c DESC, v LIMIT ${n}::text::int") == 2


def test_no_synonyms_creates_no_thesaurus():
    ddl = make_adapter("postgres", synonyms=[]).setup_ddl()
    assert "CREATE TEXT SEARCH DICTIONARY" not in ddl
    assert "_ths," not in ddl


def test_load_documents_strips_nul_bytes(adapter):
    # CQADupStack posts contain U+0000, which PostgreSQL text columns reject.
    import numpy as np

    class Client:
        rows: list = []

        def copy(self, table, names, rows):
            self.rows.extend(rows)

    client = Client()
    doc = {"id": "1", "name": "a\x00b", "product_class": ["x\x00y"]}
    adapter.load_documents(client, [doc], np.array([[0.1, 0.2, 0.3]]))
    values = [x for v in client.rows[0] for x in (v if isinstance(v, list) else [v])]
    assert not any(isinstance(v, str) and "\x00" in v for v in values)
    assert "ab" in values and "xy" in values
