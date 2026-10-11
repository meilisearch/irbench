"""Adapter contract: every engine indexes the fixture and passes known-answer
queries for every mode it claims. Needs Docker: `uv run pytest -m docker`."""

from dataclasses import dataclass

import httpx
import numpy as np
import pytest

from adapter_helpers import SCENARIO, make_task
from conftest import REPO_ROOT
from contract_fixture import COLORS, ITEMS, contract_docs, rating
from irbench.engines.base import EngineAdapter, IndexContext, load_adapter_class
from irbench.engines.config import load_engine_config
from irbench.models import Filter, Sort
from irbench.runtime.docker import DockerRuntime, wait_healthy
from irbench.runtime.hardware import load_hardware
from irbench.scenario import load_scenario
from irbench.score.checks import check_task
from irbench.stages.relevance import run_request
from irbench.terms import load_term_groups

pytestmark = pytest.mark.docker

ENGINES = ["meilisearch", "elasticsearch", "qdrant"]
INDEX = "irbench_contract"
TOP_K = 100
VERIFY_LIMIT = 1000


@dataclass
class Engine:
    adapter: EngineAdapter
    client: httpx.Client
    docs: list[dict]
    docs_by_id: dict[str, dict]
    vectors: np.ndarray

    def index(self):
        self.adapter.create_index(self.client)
        self.adapter.load_documents(self.client, self.docs, self.vectors)
        self.adapter.wait_until_indexed(self.client, len(self.docs), 300)

    def search(self, task, vector=None, limit=TOP_K, facets=True):
        reason = self.adapter.unsupported_reason(task)
        if reason:
            pytest.skip(f"{self.adapter.name}: {task.mode}:{task.base} N/A — {reason}")
        request = self.adapter.render(task, vector, limit, facets)
        return run_request(self.client, self.adapter, task, request, 30)

    def vec(self, doc_id: str) -> list[float]:
        return self.vectors[int(doc_id)].tolist()


@pytest.fixture(scope="module", params=ENGINES)
def engine(request):
    name = request.param
    cls = load_adapter_class(REPO_ROOT, name)
    scenario = load_scenario(REPO_ROOT, SCENARIO)
    docs, vectors = contract_docs()
    adapter = cls(
        IndexContext(
            scenario=scenario,
            config=load_engine_config(REPO_ROOT, name, SCENARIO).track("default"),
            index=INDEX,
            dim=vectors.shape[1],
            synonyms=load_term_groups(scenario.synonyms_path),
        )
    )
    runtime = DockerRuntime(REPO_ROOT, name, load_hardware(REPO_ROOT, "local"))
    runtime.up()
    client = httpx.Client(base_url=f"http://localhost:{cls.host_port}", timeout=60)
    try:
        wait_healthy(client, cls.health_path, 180)
        e = Engine(adapter, client, docs, {d["id"]: d for d in docs}, vectors)
        e.index()
        yield e
    finally:
        client.close()
        runtime.down()


def _ids_where(pred) -> set[str]:
    return {str(i) for i in range(100) if pred(i)}


def test_reindex_replaces_previous_index(engine):
    engine.index()  # second time into an existing index
    result = engine.search(make_task("semantic"), engine.vec("0"))
    assert len(result.ids) == len(set(result.ids)) == 100
    assert set(result.ids) == set(engine.docs_by_id)


def test_keyword_known_answer(engine):
    assert engine.search(make_task("keyword", text="white table")).ids[0] == "42"


def test_keyword_loads_shared_synonyms(engine):
    ids = engine.search(make_task("keyword", text="gray")).ids
    assert ids and set(ids) <= _ids_where(lambda i: COLORS[i // 10] == "grey")


def test_semantic_known_answer(engine):
    assert engine.search(make_task("semantic"), engine.vec("42")).ids[0] == "42"


def test_hybrid_known_answer(engine):
    result = engine.search(make_task("hybrid", text="white table"), engine.vec("42"))
    assert result.ids[0] == "42"


def test_filtered_keyword_returns_exactly_the_matching_docs(engine):
    f = Filter(field="average_rating", op="gte", value=3.0)
    task = make_task("filtered", "keyword", text="table", filter=f)
    result = engine.search(task)
    assert check_task(task, result, None, engine.docs_by_id, VERIFY_LIMIT).filter_violations == 0
    expected = _ids_where(lambda i: ITEMS[i % 10] == "table" and (rating(i) or 0) >= 3.0)
    assert set(result.ids) == expected


def test_filtered_semantic_prefilters(engine):
    f = Filter(field="product_class", op="eq", value="Sofas")
    task = make_task("filtered", "semantic", filter=f)
    result = engine.search(task, engine.vec("50"))
    assert result.ids[0] == "50"
    assert check_task(task, result, None, engine.docs_by_id, VERIFY_LIMIT).filter_violations == 0
    assert set(result.ids) == _ids_where(lambda i: ITEMS[i % 10] == "sofa")


def test_sorted_keyword_orders_and_puts_missing_values_last(engine):
    sort = Sort(field="average_rating", direction="desc")
    task = make_task("sorted", "keyword", text="chair", sort=sort)
    result = engine.search(task)
    assert check_task(task, result, None, engine.docs_by_id, VERIFY_LIMIT).order_violations == 0
    assert set(result.ids) == _ids_where(lambda i: ITEMS[i % 10] == "chair")


def test_sorted_keyword_orders_across_relevance_buckets(engine):
    # "grey chair": doc 53 matches both words, other grey docs and chairs match one.
    sort = Sort(field="review_count", direction="desc")
    task = make_task("sorted", "keyword", text="grey chair", sort=sort)
    result = engine.search(task)
    assert len(result.ids) > 1
    assert check_task(task, result, None, engine.docs_by_id, VERIFY_LIMIT).order_violations == 0


def test_sorted_semantic_orders(engine):
    task = make_task("sorted", "semantic", sort=Sort(field="review_count", direction="desc"))
    result = engine.search(task, engine.vec("42"))
    assert result.ids
    assert check_task(task, result, None, engine.docs_by_id, VERIFY_LIMIT).order_violations == 0


@pytest.mark.parametrize("base", ["keyword", "semantic"])
def test_facet_counts_match_the_result_set(engine, base):
    task = make_task("faceted", base, text="red", facets=["product_class"])
    vector = engine.vec("0") if base == "semantic" else None
    result = engine.search(task, vector)
    check_ids = engine.search(task, vector, limit=VERIFY_LIMIT, facets=False).ids
    check = check_task(task, result, check_ids, engine.docs_by_id, VERIFY_LIMIT)
    assert check.facet_verifiable is True
    assert check.facet_accuracy == 1.0, (result.facets, check_ids)
