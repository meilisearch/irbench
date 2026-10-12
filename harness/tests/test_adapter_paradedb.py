import pytest

from adapter_helpers import SCENARIO, make_task
from conftest import REPO_ROOT
from irbench.engines.base import IndexContext, SqlRequest, load_adapter_class
from irbench.engines.config import load_engine_config
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
