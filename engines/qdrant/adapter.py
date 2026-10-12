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
DENSE = "dense"
BM25 = "qdrant/bm25"


def point_id(doc_id: str) -> int | str:
    return int(doc_id) if doc_id.isdigit() else str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


class Adapter(EngineAdapter):
    name = "qdrant"
    host_port = 16333
    health_path = "/readyz"
    data_dir = "/qdrant/storage"

    def collection_body(self) -> dict[str, Any]:
        body = {
            "vectors": {DENSE: {"size": self.ctx.dim, "distance": "Cosine"}},
            # One BM25 sparse vector per searchable field; the server tokenizes and stems.
            "sparse_vectors": {
                name: {"modifier": "idf"} for name in self.ctx.scenario.index_fields.searchable
            },
        }
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
                {
                    "id": point_id(str(doc[id_field])),
                    "vector": {DENSE: vectors[i].tolist(), **self._sparse_vectors(doc)},
                    "payload": doc,
                }
                for i, doc in enumerate(docs[start : start + BATCH_SIZE], start=start)
            ]
            call(
                client,
                Request("PUT", f"/collections/{c}/points?wait=true", body={"points": points}),
            )

    def _sparse_vectors(self, doc: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in self.ctx.scenario.index_fields.searchable:
            value = doc.get(name)
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            text = "" if value is None else str(value).strip()
            if text:
                out[name] = {"text": text, "model": BM25}
        return out

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

    def _keyword(
        self, text: str, limit: int, query_filter: dict[str, Any] | None
    ) -> dict[str, Any]:
        """BM25 per searchable field, combined by a weighted sum: weight n - i for field i of n.

        Same priority as the Elasticsearch adapter's `name^5 ... description^1` boosts. Qdrant
        formulas have no max, so this sums the fields where Elasticsearch takes the best one.
        """
        fields = self.ctx.scenario.index_fields.searchable
        n = len(fields)
        leaves = []
        for name in fields:
            leaf: dict[str, Any] = {
                "query": {"text": text, "model": BM25},
                "using": name,
                "limit": self.ctx.scenario.top_k,
            }
            if query_filter:
                leaf["filter"] = query_filter
            leaves.append(leaf)
        return {
            "prefetch": leaves,
            "query": {
                "formula": {"sum": [{"mult": [n - i, f"$score[{i}]"]} for i in range(n)]},
                "defaults": {f"$score[{i}]": 0 for i in range(n)},
            },
            "limit": limit,
        }

    def _dense(
        self, vector: list[float], limit: int, query_filter: dict[str, Any] | None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"query": vector, "using": DENSE, "limit": limit}
        if query_filter:
            body["filter"] = query_filter
        return body

    def _retrieve(
        self,
        task: QueryTask,
        vector: list[float] | None,
        limit: int,
        query_filter: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """The query for the task's retrieval base; the filter is applied inside every branch."""
        top_k = self.ctx.scenario.top_k
        if task.base == "keyword":
            return self._keyword(task.text, limit, query_filter)
        vector = require_vector(task, vector)
        if task.base == "semantic":
            return self._dense(vector, limit, query_filter)
        return {  # hybrid: native reciprocal rank fusion of the keyword and dense rankings
            "prefetch": [
                self._keyword(task.text, top_k, query_filter),
                self._dense(vector, top_k, query_filter),
            ],
            "query": {"fusion": "rrf"},
            "limit": limit,
        }

    def render(
        self, task: QueryTask, vector: list[float] | None, limit: int, facets: bool = True
    ) -> Request:
        if facets and task.facets:
            raise ValueError(
                "qdrant adapter does not render faceted tasks; "
                f"{task.task_id} must be declared unsupported in the config"
            )
        query_filter = self._filter(task.filter) if task.filter is not None else None
        top_k = self.ctx.scenario.top_k
        body: dict[str, Any]
        if task.sort is not None:
            body = {
                "prefetch": self._retrieve(task, vector, top_k, query_filter),
                "query": {"order_by": {"key": task.sort.field, "direction": task.sort.direction}},
            }
        else:
            body = self._retrieve(task, vector, limit, query_filter)
        if query_filter:
            body["filter"] = query_filter
        body["limit"] = limit
        body["with_payload"] = [self.ctx.scenario.id_field]
        body = deep_merge(body, self.search_params(task))
        return Request("POST", f"/collections/{self.ctx.index}/points/query", body=body)

    def parse(self, task: QueryTask, payload: Any) -> SearchResult:
        id_field = self.ctx.scenario.id_field
        return SearchResult(ids=[str(p["payload"][id_field]) for p in payload["result"]["points"]])
