"""Qdrant adapter. Parity notes: engines/qdrant/README.md."""

import uuid
from typing import Any

import httpx
import numpy as np

from irbench.engines.base import (
    EngineAdapter,
    EngineError,
    Request,
    SearchResult,
    call,
    poll,
    require_vector,
)
from irbench.engines.config import deep_merge
from irbench.models import Filter, QueryTask

BATCH_SIZE = 256


def point_id(doc_id: str) -> int | str:
    return int(doc_id) if doc_id.isdigit() else str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


class Adapter(EngineAdapter):
    name = "qdrant"
    host_port = 16333
    health_path = "/readyz"
    data_dir = "/qdrant/storage"

    def collection_body(self) -> dict[str, Any]:
        body = {"vectors": {"size": self.ctx.dim, "distance": "Cosine"}}
        return deep_merge(body, self.ctx.config.index)

    def payload_indexes(self) -> dict[str, str]:
        f = self.ctx.scenario.index_fields
        names = sorted(set(f.filterable) | set(f.facets) | set(f.sortable))
        return {name: "float" if name in f.numeric else "keyword" for name in names}

    def create_index(self, client: httpx.Client) -> None:
        c = self.ctx.index
        response = Request("DELETE", f"/collections/{c}").send(client)
        if response.status_code not in (200, 404):
            raise EngineError(
                f"DELETE /collections/{c} -> {response.status_code}: {response.text[:500]}"
            )
        call(client, Request("PUT", f"/collections/{c}", body=self.collection_body()))
        for name, schema in self.payload_indexes().items():
            body = {"field_name": name, "field_schema": schema}
            call(client, Request("PUT", f"/collections/{c}/index?wait=true", body=body))

    def load_documents(
        self, client: httpx.Client, docs: list[dict[str, Any]], vectors: np.ndarray
    ) -> None:
        c, id_field = self.ctx.index, self.ctx.scenario.id_field
        for start in range(0, len(docs), BATCH_SIZE):
            points = [
                {"id": point_id(str(doc[id_field])), "vector": vectors[i].tolist(), "payload": doc}
                for i, doc in enumerate(docs[start : start + BATCH_SIZE], start=start)
            ]
            call(
                client,
                Request("PUT", f"/collections/{c}/points?wait=true", body={"points": points}),
            )

    def wait_until_indexed(self, client: httpx.Client, n_docs: int, timeout_s: float) -> None:
        c = self.ctx.index

        def ready() -> bool:
            info = call(client, Request("GET", f"/collections/{c}"))["result"]
            if info["status"] != "green":
                return False
            count_request = Request("POST", f"/collections/{c}/points/count", body={"exact": True})
            return call(client, count_request)["result"]["count"] == n_docs

        poll(ready, timeout_s, f"qdrant collection {c} to be green with {n_docs} points")

    def _filter(self, f: Filter) -> dict[str, Any]:
        if f.op == "eq":
            return {"must": [{"key": f.field, "match": {"value": f.value}}]}
        return {"must": [{"key": f.field, "range": {f.op: float(f.value)}}]}

    def render(
        self, task: QueryTask, vector: list[float] | None, limit: int, facets: bool = True
    ) -> Request:
        if task.base != "semantic" or (facets and task.facets):
            raise ValueError(
                "qdrant adapter renders semantic, filtered and sorted tasks on the semantic base "
                f"only; {task.task_id} must be declared unsupported in the config"
            )
        vector = require_vector(task, vector)
        query_filter = self._filter(task.filter) if task.filter is not None else None
        body: dict[str, Any]
        if task.sort is not None:
            prefetch: dict[str, Any] = {"query": vector, "limit": self.ctx.scenario.top_k}
            if query_filter:
                prefetch["filter"] = query_filter
            body = {
                "prefetch": prefetch,
                "query": {"order_by": {"key": task.sort.field, "direction": task.sort.direction}},
            }
        else:
            body = {"query": vector}
            if query_filter:
                body["filter"] = query_filter
        body["limit"] = limit
        body["with_payload"] = [self.ctx.scenario.id_field]
        body = deep_merge(body, self.search_params(task))
        return Request("POST", f"/collections/{self.ctx.index}/points/query", body=body)

    def parse(self, task: QueryTask, payload: Any) -> SearchResult:
        id_field = self.ctx.scenario.id_field
        return SearchResult(ids=[str(p["payload"][id_field]) for p in payload["result"]["points"]])
