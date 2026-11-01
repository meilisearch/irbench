"""The tuned Chinese images (spec 2026-10-31 Chinese title search §5.2): each plugin is
installed and splits an unspaced Chinese query into words."""

import numpy as np
import pytest

from adapter_helpers import make_adapter, make_task
from conftest import REPO_ROOT
from irbench.engines.base import engine_base_url, load_adapter_class
from irbench.runtime.docker import DockerRuntime, wait_ready
from irbench.runtime.hardware import load_hardware
from irbench.stages.context import default_client
from irbench.stages.relevance import run_request

QUERY = "纯山楂酱"
TITLE = "护肚子肚围婴儿冬季纯棉加厚绒肚兜宝宝保暖新生儿裹腹护肚脐神器"


@pytest.mark.docker
def test_es_zh_analyzers():
    cls = load_adapter_class(REPO_ROOT, "elasticsearch")
    runtime = DockerRuntime(
        REPO_ROOT, "elasticsearch", load_hardware(REPO_ROOT, "local"), compose="compose.zh.yaml"
    )
    runtime.up()
    try:
        client = default_client(engine_base_url(cls, "localhost"))
        wait_ready(client, cls.health_path, 180)
        for analyzer in ("smartcn", "icu_analyzer", "ik_smart", "ik_max_word"):
            analyzed = client.post("/_analyze", json={"analyzer": analyzer, "text": QUERY}).json()
            tokens = [t["token"] for t in analyzed["tokens"]]
            assert len(tokens) >= 2, (analyzer, tokens)
    finally:
        runtime.down()


@pytest.mark.docker
def test_pg_zh_parsers():
    cls = load_adapter_class(REPO_ROOT, "postgres")
    # match: any, as a segmenter may cut the query differently from the title (护肚 / 护肚子).
    track = {"index": {"text_search_config": "zhparser"}, "search": {"keyword": {"match": "any"}}}
    adapter = make_adapter("postgres", track, dim=3, synonyms=[])
    runtime = DockerRuntime(
        REPO_ROOT, "postgres", load_hardware(REPO_ROOT, "local"), compose="compose.zh.yaml"
    )
    runtime.up()
    client = default_client(engine_base_url(cls, "localhost"))
    try:
        wait_ready(client, cls.health_path, 180)
        adapter.create_index(client)
        extensions = {r["extname"] for r in client.execute("SELECT extname FROM pg_extension")}
        assert {"zhparser", "vector"} <= extensions
        client.execute("CREATE EXTENSION IF NOT EXISTS pgroonga")
        lexemes = client.execute(
            f"SELECT unnest(tsvector_to_array(to_tsvector('{adapter.ts_config}', '{QUERY}'))) AS l"
        )
        assert len(lexemes) >= 2, lexemes
        field = adapter.ctx.scenario.index_fields.searchable[0]
        adapter.load_documents(client, [{"id": "1", field: TITLE}], np.eye(1, 3, dtype=np.float32))
        adapter.wait_until_indexed(client, 1, 60)
        task = make_task("keyword", text="宝宝护肚")
        request = adapter.render(task, None, 10)
        assert run_request(client, adapter, task, request, 30).ids == ["1"]
    finally:
        client.close()
        runtime.down()
