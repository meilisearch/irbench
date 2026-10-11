import json

import httpx
import pytest

from irbench.engines.base import (
    EngineAdapter,
    EngineError,
    Request,
    call,
    index_name,
    load_adapter_class,
    poll,
    require_vector,
)
from irbench.models import QueryTask

TOY = """
from irbench.engines.base import EngineAdapter


class Adapter(EngineAdapter):
    name = "toy"
    host_port = 1
    health_path = "/"
    data_dir = "/d"

    def create_index(self, client): pass
    def load_documents(self, client, docs, vectors): pass
    def wait_until_indexed(self, client, n_docs, timeout_s): pass
    def render(self, task, vector, limit, facets=True): pass
    def parse(self, task, payload): pass
"""


def _engine_file(root, engine, source):
    path = root / "engines" / engine / "adapter.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)


def test_request_json_roundtrip():
    r = Request("POST", "/i/search", body={"q": "sofa"})
    assert r.to_json() == {
        "protocol": "http",
        "method": "POST",
        "path": "/i/search",
        "headers": {},
        "body": {"q": "sofa"},
    }
    assert Request.from_json(json.loads(json.dumps(r.to_json()))) == r
    raw = Request(
        "POST", "/_bulk", content="{}\n", headers={"Content-Type": "application/x-ndjson"}
    )
    assert "body" not in raw.to_json()
    assert Request.from_json(raw.to_json()) == raw


def test_call_sends_json_or_raw_content_and_raises_on_http_error():
    seen = []

    def handler(req):
        seen.append((req.headers.get("content-type"), req.content))
        if req.url.path == "/bad":
            return httpx.Response(400, text="nope")
        if req.url.path == "/empty":
            return httpx.Response(204)
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(base_url="http://e", transport=httpx.MockTransport(handler))
    assert call(client, Request("POST", "/j", body={"a": 1})) == {"ok": True}
    call(
        client,
        Request(
            "POST", "/n", content='{"x":1}\n', headers={"Content-Type": "application/x-ndjson"}
        ),
    )
    assert seen[0][0] == "application/json" and json.loads(seen[0][1]) == {"a": 1}
    assert seen[1] == ("application/x-ndjson", b'{"x":1}\n')
    assert call(client, Request("DELETE", "/empty")) is None
    with pytest.raises(EngineError, match="GET /bad -> 400: nope"):
        call(client, Request("GET", "/bad"))


def test_poll_returns_when_ready_and_times_out():
    state = iter([False, False, True])
    poll(lambda: next(state), timeout_s=5, what="x", interval_s=0)
    with pytest.raises(TimeoutError, match="waiting for thing"):
        poll(lambda: False, timeout_s=0, what="thing", interval_s=0)


def test_index_name_is_engine_safe():
    assert index_name("ecommerce/product-search", "bge-small-en-v1.5") == (
        "irbench_ecommerce_product_search_bge_small_en_v1_5"
    )


def test_require_vector():
    task = QueryTask(
        task_id="1:semantic:semantic",
        qid="1",
        variant_of="1",
        perturbation="original",
        text="x",
        mode="semantic",
        base="semantic",
    )
    assert require_vector(task, [0.5]) == [0.5]
    with pytest.raises(ValueError, match="1:semantic:semantic"):
        require_vector(task, None)


def test_load_adapter_class(tmp_path):
    _engine_file(tmp_path, "toy", TOY)
    cls = load_adapter_class(tmp_path, "toy")
    assert issubclass(cls, EngineAdapter) and cls.name == "toy"


def test_load_adapter_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope"):
        load_adapter_class(tmp_path, "nope")
    _engine_file(tmp_path, "bare", "x = 1\n")
    with pytest.raises(TypeError, match="Adapter"):
        load_adapter_class(tmp_path, "bare")
    _engine_file(tmp_path, "other", TOY)
    with pytest.raises(ValueError, match="toy"):
        load_adapter_class(tmp_path, "other")
