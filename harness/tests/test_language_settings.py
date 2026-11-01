"""Language settings are config keys: each adapter carries one through its existing channel."""

from dataclasses import replace

import pytest

from adapter_helpers import make_adapter, make_task
from conftest import REPO_ROOT
from irbench.engines.base import EngineAdapter, load_adapter_class

ENGINES = ["meilisearch", "elasticsearch", "typesense", "qdrant", "postgres", "postgres-paradedb"]
LANGUAGE_KEYS = {
    "meilisearch": ("index.localizedAttributes",),
    "elasticsearch": ("index.analyzer",),
    "typesense": ("index.searchable_fields.locale",),
    "qdrant": ("index.bm25_options",),
    "postgres": ("index.text_search_config",),
    "postgres-paradedb": ("index.tokenizer",),
}


def test_base_adapter_has_no_language_keys():
    assert EngineAdapter.language_keys == ()


@pytest.mark.parametrize("engine", ENGINES)
def test_every_adapter_names_its_language_keys(engine):
    assert load_adapter_class(REPO_ROOT, engine).language_keys == LANGUAGE_KEYS[engine]


def test_es_analyzer_key():
    adapter = make_adapter("elasticsearch", {"index": {"analyzer": "cjk"}}, synonyms=[])
    body = adapter.index_body()
    props = body["mappings"]["properties"]
    for name in adapter.ctx.scenario.index_fields.searchable:
        assert props[name]["analyzer"] == props[name]["search_analyzer"] == "cjk"
    assert "analyzer" not in body["settings"]
    assert "analyzer" not in body
    assert "irbench_search" not in body["settings"]["analysis"].get("analyzer", {})


def test_es_analyzer_key_keeps_the_synonym_analyzer_for_search():
    adapter = make_adapter("elasticsearch", {"index": {"analyzer": "cjk"}})
    body = adapter.index_body()
    props = body["mappings"]["properties"]
    for name in adapter.ctx.scenario.index_fields.searchable:
        assert props[name]["analyzer"] == "cjk"
        assert props[name]["search_analyzer"] == "irbench_search"
    assert "irbench_search" in body["settings"]["analysis"]["analyzer"]
    assert "analyzer" not in body["settings"] and "analyzer" not in body


def test_es_without_the_key_has_no_index_analyzer():
    props = make_adapter("elasticsearch").mappings()["properties"]
    assert all("analyzer" not in p for p in props.values())


def test_pg_text_search_config_key():
    ddl = make_adapter("postgres", {"index": {"text_search_config": "spanish"}}).setup_ddl()
    assert "COPY = spanish" in ddl and "COPY = english" not in ddl


def test_pg_default_config_is_english():
    assert "COPY = english" in make_adapter("postgres").setup_ddl()


def test_pg_thesaurus_follows_the_config_language():
    track = {"index": {"text_search_config": "spanish"}}
    ddl = make_adapter("postgres", track, synonyms=[["gris", "plomo"]]).setup_ddl()
    assert "Dictionary = spanish_stem" in ddl and "english_stem" not in ddl
    ddl = make_adapter("postgres", synonyms=[["grey", "gray"]]).setup_ddl()
    assert "Dictionary = english_stem" in ddl


def test_pg_text_search_config_is_an_identifier():
    adapter = make_adapter("postgres", {"index": {"text_search_config": "x; DROP TABLE y"}})
    with pytest.raises(ValueError, match="unsafe SQL identifier"):
        adapter.setup_ddl()


@pytest.mark.parametrize(
    "expr", ["pdb.lindera(japanese)", "pdb.simple('stemmer=spanish', 'ascii_folding=true')"]
)
def test_paradedb_tokenizer_key(expr):
    # pg_search 0.26: `(<column>::pdb.<tokenizer>(<args>))` inside the index column list, see
    # https://www.paradedb.com/docs/documentation/tokenizers/available-tokenizers/lindera and
    # https://www.paradedb.com/docs/documentation/token-filters/stemming
    adapter = make_adapter("postgres-paradedb", {"index": {"tokenizer": expr}})
    ddl = adapter.text_index_ddl()
    fields = adapter.ctx.scenario.index_fields.searchable
    assert ddl == (
        f"CREATE INDEX ON {adapter.table} USING bm25 (id, "
        + ", ".join(f"({f}::{expr})" for f in fields)
        + ")"
    )


def test_paradedb_without_the_key_keeps_bare_columns():
    adapter = make_adapter("postgres-paradedb")
    fields = ", ".join(adapter.ctx.scenario.index_fields.searchable)
    assert adapter.text_index_ddl().endswith(f"USING bm25 (id, {fields})")


def test_paradedb_tokenizer_is_not_free_text():
    adapter = make_adapter("postgres-paradedb", {"index": {"tokenizer": "x); DROP TABLE y; --"}})
    with pytest.raises(ValueError, match="tokenizer"):
        adapter.text_index_ddl()


def test_meili_typesense_qdrant_pass_through():
    meili = make_adapter(
        "meilisearch",
        {
            "index": {"localizedAttributes": [{"attributePatterns": ["*"], "locales": ["jpn"]}]},
            "search": {"keyword": {"locales": ["jpn"]}},
        },
    )
    assert meili.index_settings()["localizedAttributes"] == [
        {"attributePatterns": ["*"], "locales": ["jpn"]}
    ]
    assert meili.render(make_task("keyword"), None, 10).body["locales"] == ["jpn"]

    ts = make_adapter("typesense", {"index": {"searchable_fields": {"locale": "ja"}}})
    searchable = ts.ctx.scenario.index_fields.searchable
    fields = {f["name"]: f for f in ts.fields()}
    assert all(fields[name]["locale"] == "ja" for name in searchable)

    qdrant = make_adapter("qdrant", {"index": {"bm25_options": {"language": "spanish"}}})
    doc = {name: "sofá gris" for name in qdrant.ctx.scenario.index_fields.searchable}
    sparse = qdrant._sparse_vectors(doc)
    assert sparse and all(v["options"] == {"language": "spanish"} for v in sparse.values())
    assert "bm25_options" not in qdrant.collection_body()


@pytest.mark.parametrize("expr", ["pdb.simple(--)", "pdb.simple('a'); DROP TABLE t", "english"])
def test_paradedb_tokenizer_rejects_comments_and_statements(expr):
    adapter = make_adapter("postgres-paradedb", {"index": {"tokenizer": expr}})
    with pytest.raises(ValueError, match="tokenizer"):
        adapter.text_index_ddl()


def test_pgroonga_render():
    adapter = make_adapter(
        "postgres", {"index": {"text_search": "pgroonga"}}, synonyms=[["grey", "gray"]]
    )
    setup = adapter.setup_ddl()
    assert "CREATE EXTENSION IF NOT EXISTS pgroonga" in setup and "vector" in setup
    assert "TEXT SEARCH" not in setup
    assert adapter.extra_columns() == []
    fields = adapter.ctx.scenario.index_fields.searchable
    ddl = adapter.text_index_ddl()
    assert ddl == (
        f"CREATE INDEX {adapter.pgroonga_index} ON {adapter.table} USING pgroonga "
        f"((ARRAY[{', '.join(fields)}]))"
    )
    request = adapter.render(make_task("keyword", text="ズボン"), None, 10)
    weights = ", ".join(str(len(fields) - i) for i in range(len(fields)))
    assert f"ARRAY[{', '.join(fields)}] &@~ pgroonga_condition($1::text, " in request.sql
    assert f"weights => ARRAY[{weights}]" in request.sql
    assert f"index_name => '{adapter.pgroonga_index}'" in request.sql
    assert "pgroonga_score(tableoid, ctid)" in request.sql
    assert "tsv" not in request.sql and "ts_rank" not in request.sql
    assert request.params[0] == "ズボン"


def test_pgroonga_index_name_fits_an_identifier():
    adapter = make_adapter("postgres", {"index": {"text_search": "pgroonga"}})
    adapter.ctx = replace(adapter.ctx, index="irbench_" + "x" * 80)
    assert len(adapter.pgroonga_index) <= 63


def test_pgroonga_rejects_options_it_cannot_honour():
    adapter = make_adapter(
        "postgres",
        {"index": {"text_search": "pgroonga"}, "search": {"keyword": {"match": "any"}}},
    )
    with pytest.raises(ValueError, match="pgroonga"):
        adapter.render(make_task("keyword"), None, 10)
    adapter = make_adapter(
        "postgres",
        {"index": {"text_search": "pgroonga"}, "search": {"keyword": {"rank": "ts_rank_cd"}}},
    )
    with pytest.raises(ValueError, match="pgroonga"):
        adapter.render(make_task("keyword"), None, 10)


def test_pg_unknown_text_search_is_rejected():
    adapter = make_adapter("postgres", {"index": {"text_search": "bm25"}})
    with pytest.raises(ValueError, match="text_search"):
        adapter.setup_ddl()


def test_pg_tsvector_is_the_default_text_search():
    for track in (None, {"index": {"text_search": "tsvector"}}):
        adapter = make_adapter("postgres", track)
        assert "USING gin (tsv)" in adapter.text_index_ddl()
        assert "pgroonga" not in adapter.setup_ddl()
        assert "websearch_to_tsquery" in adapter.render(make_task("keyword"), None, 10).sql


@pytest.mark.parametrize("engine", ENGINES)
def test_report_keys_default_to_language_keys(engine):
    cls = load_adapter_class(REPO_ROOT, engine)
    if engine == "postgres":
        # a PGroonga run inherits `text_search_config` from the default block: the report
        # must also show `text_search`, which lint does not require
        assert cls.report_keys() == ("index.text_search", "index.text_search_config")
        assert cls.language_keys == ("index.text_search_config",)
    else:
        assert cls.report_keys() == LANGUAGE_KEYS[engine]


def _settings_sql(adapter):
    class Recorder:
        sql = ""

        def script(self, sql):
            self.sql = sql

    rec = Recorder()
    adapter.apply_settings(rec)
    return rec.sql


def test_pgroonga_turns_sequential_scans_off_by_default():
    sql = _settings_sql(make_adapter("postgres", {"index": {"text_search": "pgroonga"}}))
    assert "ALTER DATABASE %I SET enable_seqscan = off" in sql
    assert sql.endswith("\nSET enable_seqscan = off;")


def test_pgroonga_seqscan_default_yields_to_the_config():
    track = {"index": {"text_search": "pgroonga", "settings": {"enable_seqscan": "on"}}}
    sql = _settings_sql(make_adapter("postgres", track))
    assert "enable_seqscan = on" in sql and "enable_seqscan = off" not in sql


def test_tsvector_sets_no_planner_default():
    for track in (None, {"index": {"text_search": "tsvector"}}):
        assert "enable_seqscan" not in _settings_sql(make_adapter("postgres", track))
