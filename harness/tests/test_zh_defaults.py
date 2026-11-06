"""Each engine's default-track Chinese setting (spec 2026-10-31 Chinese title search §5.1)
exists on the pinned engine version and finds a title from a query cut differently."""

import numpy as np
import pytest

from adapter_helpers import make_task
from conftest import REPO_ROOT
from irbench.engines.base import IndexContext, engine_base_url, load_adapter_class
from irbench.engines.config import load_engine_config
from irbench.runtime.docker import DockerRuntime, wait_ready
from irbench.runtime.hardware import load_hardware
from irbench.scenario import load_scenario
from irbench.stages.context import default_client
from irbench.stages.relevance import run_request

SCENARIO = "ecommerce/title-search-zh"
TITLE = "护肚子肚围婴儿冬季纯棉加厚绒肚兜宝宝保暖新生儿裹腹护肚脐神器"
OTHER = "撒哈拉X500多硬盘位机箱支持9/12/14盘 双U服务器机箱"
QUERY = "宝宝护肚"
ENGINES = ["meilisearch", "elasticsearch", "typesense", "qdrant", "postgres-paradedb", "postgres"]


@pytest.mark.docker
@pytest.mark.parametrize("engine", ENGINES)
def test_default_finds_title(engine):
    cls = load_adapter_class(REPO_ROOT, engine)
    scenario = load_scenario(REPO_ROOT, SCENARIO)
    config = load_engine_config(REPO_ROOT, engine, SCENARIO).track("default")
    adapter = cls(
        IndexContext(scenario=scenario, config=config, index="zh_default", dim=3, synonyms=[])
    )
    docs = [{"id": "1", "title": TITLE}, {"id": "2", "title": OTHER}]
    vectors = np.eye(2, 3, dtype=np.float32)
    runtime = DockerRuntime(REPO_ROOT, engine, load_hardware(REPO_ROOT, "local"))
    runtime.up()
    client = default_client(engine_base_url(cls, "localhost"))
    try:
        wait_ready(client, cls.health_path, 180)
        adapter.create_index(client)
        adapter.load_documents(client, docs, vectors)
        adapter.wait_until_indexed(client, len(docs), 120)
        task = make_task("keyword", text=QUERY)
        ids = run_request(client, adapter, task, adapter.render(task, None, 10), 30).ids
        if engine == "postgres":
            # No Chinese configuration: `simple` keeps the unspaced title as one token.
            assert ids == []
        else:
            assert ids[:1] == ["1"], ids
    finally:
        client.close()
        runtime.down()
