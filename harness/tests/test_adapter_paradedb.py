import pytest

from adapter_helpers import SCENARIO, make_adapter, make_task
from conftest import REPO_ROOT
from irbench.engines.base import IndexContext, SqlRequest, load_adapter_class
from irbench.engines.config import load_engine_config
from irbench.models import Filter, Sort
from irbench.scenario import load_scenario

ENGINE = "postgres-paradedb"


@pytest.fixture
def adapter():
    scenario = load_scenario(REPO_ROOT, SCENARIO)
    cls = load_adapter_class(REPO_ROOT, ENGINE)
    config = load_engine_config(REPO_ROOT, ENGINE, SCENARIO).track("default")
    return cls(
        IndexContext(
            scenario=scenario, config=config, index="irbench_t", dim=3, synonyms=[], n_docs=1000
        )
    )


def test_keyword_is_bm25_with_field_boosts_in_scenario_order(adapter):
    r = adapter.render(make_task("keyword", text="grey sofa"), None, 100)
    assert isinstance(r, SqlRequest)
    for weight, field in zip(
        [5, 4, 3, 2, 1], ["name", "product_class", "category", "features", "description"]
    ):
        assert f"{field} ||| $1::text::pdb.boost({weight})" in r.sql
    assert "pdb.score(id)" in r.sql and r.params[0] == "grey sofa"


def test_bm25_index_ddl(adapter):
    ddl = adapter.text_index_ddl()
    assert "USING bm25 (id, name, product_class, category, features, description)" in ddl
    assert "key_field" not in ddl


def test_setup_creates_both_extensions(adapter):
    ddl = adapter.setup_ddl()
    assert "pg_search" in ddl and "vector" in ddl


def test_synonyms_declared_unsupported(adapter):
    assert adapter.ctx.config.synonyms


def test_keyword_orders_by_score_only_so_topk_can_push_down(adapter):
    r = adapter.render(make_task("keyword", text="sofa"), None, 100)
    assert "ORDER BY score DESC LIMIT" in r.sql
    assert r.sql.endswith("m ORDER BY score DESC")
    assert "score DESC, id" not in r.sql


def test_filtered_keyword_orders_by_score_only(adapter):
    task = make_task(
        "keyword", text="sofa", filter=Filter(field="product_class", op="eq", value="Sofa")
    )
    r = adapter.render(task, None, 100)
    assert "score DESC, id" not in r.sql and "ORDER BY score DESC LIMIT" in r.sql


def test_faceted_keyword_hits_order_by_score_only(adapter):
    r = adapter.render(
        make_task("faceted", "keyword", text="sofa", facets=["product_class"]), None, 100
    )
    assert "ORDER BY score DESC) AS o" in r.sql and "score DESC, id" not in r.sql


def test_hybrid_keyword_side_orders_by_score_only_but_fusion_keeps_id(adapter):
    r = adapter.render(make_task("hybrid", text="sofa"), [0.1, 0.2, 0.3], 100)
    assert "ORDER BY pdb.score(id) DESC LIMIT" in r.sql
    assert "pdb.score(id) DESC, id" not in r.sql
    assert "ORDER BY score DESC, id LIMIT" in r.sql  # RRF fusion stays total


def test_sorted_keyword_keeps_id_tiebreak(adapter):
    sort = Sort(field="average_rating", direction="desc")
    r = adapter.render(make_task("sorted", "keyword", text="sofa", sort=sort), None, 100)
    assert "ORDER BY average_rating DESC NULLS LAST, id" in r.sql


def test_no_synonyms_matches_the_default_setup():
    assert (
        make_adapter("postgres-paradedb", synonyms=[]).setup_ddl()
        == make_adapter("postgres-paradedb").setup_ddl()
    )
