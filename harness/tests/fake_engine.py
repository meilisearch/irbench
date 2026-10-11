"""An in-memory engine behind httpx.MockTransport, for stage and pipeline tests."""

import json
import shutil
from typing import Any

import httpx
import yaml

from conftest import REPO_ROOT
from irbench.engines.base import EngineAdapter, Request, SearchResult, call, poll
from irbench.runs import RunSpec
from irbench.score.constraints import expected_facets
from irbench.stages.embed import embed_scenario
from irbench.stages.prepare import prepare
from test_embed_cache import FakeEmbedder
from test_prepare import fixture_client, make_root

SCENARIO = "ecommerce/product-search"
FAKE_SPEC = RunSpec(scenario=SCENARIO, engine="fake", model="fake")
FAKE_CONFIG = {"default": {"unsupported": {"hybrid": "fake engine has no hybrid"}}}


class FakeAdapter(EngineAdapter):
    name = "fake"
    host_port = 9999
    health_path = "/health"
    data_dir = "/data"

    def create_index(self, client):
        call(client, Request("PUT", f"/indexes/{self.ctx.index}"))

    def load_documents(self, client, docs, vectors):
        for start in range(0, len(docs), 2):
            ids = [d["id"] for d in docs[start : start + 2]]
            call(client, Request("POST", f"/indexes/{self.ctx.index}/docs", body={"ids": ids}))

    def wait_until_indexed(self, client, n_docs, timeout_s):
        def ready():
            count = call(client, Request("GET", f"/indexes/{self.ctx.index}/count"))["count"]
            return count == n_docs

        poll(ready, timeout_s, "fake indexing", interval_s=0)

    def render(self, task, vector, limit, facets=True):
        body = {
            "task_id": task.task_id,
            "limit": limit,
            "facets": facets and bool(task.facets),
            "check": bool(task.facets) and not facets,
            "has_vector": vector is not None,
            **self.search_params(task),
        }
        return Request("POST", f"/indexes/{self.ctx.index}/search", body=body)

    def parse(self, task, payload):
        return SearchResult(ids=payload["ids"], facets=payload.get("facets"))


class FakeServer:
    """Answers searches from a {task_id: [doc ids]} table."""

    def __init__(
        self,
        answers=None,
        facets=None,
        fail=(),
        slow=(),
        fail_checks=False,
        fail_index=False,
        drop_docs=False,
    ):
        self.answers: dict[str, list[str]] = answers or {}
        self.facets: dict[str, dict[str, dict[str, int]]] = facets or {}
        self.fail, self.slow = set(fail), set(slow)
        self.fail_checks, self.fail_index, self.drop_docs = fail_checks, fail_index, drop_docs
        self.docs: set[str] = set()

    def client_factory(self, base_url: str) -> httpx.Client:
        return httpx.Client(base_url=base_url, transport=httpx.MockTransport(self))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body: Any = json.loads(request.content) if request.content else None
        if path == "/health":
            return httpx.Response(200)
        if request.method == "PUT":
            if self.fail_index:
                return httpx.Response(500, text="index creation failed")
            self.docs = set()
            return httpx.Response(200, json={})
        if path.endswith("/docs"):
            if not self.drop_docs:
                self.docs.update(body["ids"])
            return httpx.Response(200, json={})
        if path.endswith("/count"):
            return httpx.Response(200, json={"count": len(self.docs)})
        if path.endswith("/search"):
            tid = body["task_id"]
            if tid in self.slow:
                raise httpx.ReadTimeout("engine too slow", request=request)
            if tid in self.fail or (body["check"] and self.fail_checks):
                return httpx.Response(500, text="search failed")
            out: dict[str, Any] = {"ids": self.answers.get(tid, [])[: body["limit"]]}
            if body["facets"]:
                out["facets"] = self.facets.get(tid, {})
            return httpx.Response(200, json=out)
        return httpx.Response(404)


class FakeRuntime:
    def __init__(self):
        self.calls: list[str] = []

    def up(self):
        self.calls.append("up")

    def down(self):
        self.calls.append("down")

    def image_info(self):
        return {"image": "fake:1", "digest": "fake@sha256:0"}

    def memory_peak_bytes(self):
        return 1024

    def disk_bytes(self, path):
        return 2048


def fake_root(tmp_path, wands_raw, config=None):
    """Mini WANDS prepared and embedded with the fake model, plus an `engines/fake` folder."""
    root = make_root(tmp_path, wands_raw)
    prepare(root, SCENARIO, client=fixture_client(wands_raw))
    embed_scenario(root, SCENARIO, "fake", embedder=FakeEmbedder())
    engine_dir = root / "engines" / "fake"
    (engine_dir / "configs" / "ecommerce").mkdir(parents=True)
    (engine_dir / "adapter.py").write_text(
        "from fake_engine import FakeAdapter as Adapter  # noqa: F401\n"
    )
    (engine_dir / "configs" / "ecommerce" / "product-search.yaml").write_text(
        yaml.safe_dump(config or FAKE_CONFIG)
    )
    shutil.copytree(REPO_ROOT / "hardware", root / "hardware")
    return root


def set_scenario(root, **changes):
    path = root / "scenarios" / SCENARIO / "scenario.yaml"
    raw = yaml.safe_load(path.read_text())
    raw.update(changes)
    path.write_text(yaml.safe_dump(raw))


def perfect_answers(ctx):
    """Every task answered with its relevant docs, best grade first."""
    answers, facets = {}, {}
    for task in ctx.tasks:
        grades = ctx.qrels[task.task_id]
        ids = [d for d, g in sorted(grades.items(), key=lambda kv: (-kv[1], kv[0])) if g > 0]
        answers[task.task_id] = ids
        if task.facets:
            facets[task.task_id] = {
                name: expected_facets(ids, name, ctx.docs_by_id) for name in task.facets
            }
    return answers, facets
