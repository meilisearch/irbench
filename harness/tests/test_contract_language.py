"""Language keys (`stop_words`, `stemming`) work on the real engines: the settings are
accepted and change what matches. Needs Docker: `uv run pytest -m docker`."""

import pytest

from adapter_helpers import SCENARIO, make_task
from conftest import REPO_ROOT
from contract_fixture import contract_docs
from irbench.engines.base import IndexContext, engine_base_url, load_adapter_class
from irbench.engines.config import TrackConfig, deep_merge, load_engine_config
from irbench.runtime.docker import DockerRuntime, wait_ready
from irbench.runtime.hardware import load_hardware
from irbench.scenario import load_scenario
from irbench.stages.context import default_client
from irbench.stages.relevance import run_request
from irbench.terms import load_term_groups

pytestmark = pytest.mark.docker

# Meilisearch has no stemming: only its stop words are checked.
LANGUAGE = {
    "meilisearch": {"stop_words": "english"},
    "elasticsearch": {"stop_words": "english", "stemming": "english"},
    "postgres-paradedb": {"stop_words": "english", "stemming": "english"},
    "typesense": {"stop_words": "english", "stemming": "english"},
}


@pytest.fixture(scope="module", params=sorted(LANGUAGE))
def engine(request):
    name = request.param
    cls = load_adapter_class(REPO_ROOT, name)
    scenario = load_scenario(REPO_ROOT, SCENARIO)
    docs, vectors = contract_docs()
    default = load_engine_config(REPO_ROOT, name, SCENARIO).track("default").model_dump()
    track = TrackConfig.model_validate(deep_merge(default, {"index": LANGUAGE[name]}))
    adapter = cls(
        IndexContext(
            scenario=scenario,
            config=track,
            index="irbench_contract_language",
            dim=vectors.shape[1],
            synonyms=load_term_groups(scenario.synonyms_path),
            n_docs=len(docs),
        )
    )
    runtime = DockerRuntime(REPO_ROOT, name, load_hardware(REPO_ROOT, "local"))
    runtime.up()
    client = default_client(engine_base_url(cls, "localhost"))
    try:
        wait_ready(client, cls.health_path, 180)
        adapter.create_index(client)
        adapter.load_documents(client, docs, vectors)
        adapter.wait_until_indexed(client, len(docs), 300)
        yield name, adapter, client
    finally:
        client.close()
        runtime.down()


def _ids(engine, text):
    _, adapter, client = engine
    task = make_task("keyword", text=text)
    return run_request(client, adapter, task, adapter.render(task, None, 100, False), 30).ids


def test_stop_words_are_not_matched(engine):
    # Every description reads "a <color> <item> for the home": without stop word removal
    # "for" matches all 99 non-empty descriptions.
    assert _ids(engine, "for") == []


def test_content_words_still_match(engine):
    assert _ids(engine, "the white table")[0] == "42"


def test_stemming_matches_inflected_forms(engine):
    name, _, _ = engine
    if "stemming" not in LANGUAGE[name]:
        pytest.skip(f"{name} has no stemming")
    ids = _ids(engine, "tables")
    assert ids and ids[0] in {str(i) for i in range(2, 100, 10)}  # the ten tables
