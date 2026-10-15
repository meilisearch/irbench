"""Meilisearch adapter. Parity notes: engines/meilisearch/README.md."""

import time
from typing import Any

import httpx
import numpy as np

from irbench.engines.base import (
    EngineAdapter,
    EngineError,
    IndexContext,
    Request,
    SearchResult,
    call,
    poll,
    require_vector,
)
from irbench.engines.config import deep_merge
from irbench.models import Filter, QueryTask

EMBEDDER = "default"
BATCH_SIZE = 1000
SETUP_TIMEOUT_S = 600


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def filter_expression(f: Filter) -> str:
    if f.op == "eq":
        return f"{f.field} = {_quote(str(f.value))}"
    return f"{f.field} {'>=' if f.op == 'gte' else '<='} {float(f.value)}"


def synonyms_setting(groups: list[list[str]]) -> dict[str, list[str]]:
    return {term: [t for t in group if t != term] for group in groups for term in group}


class Adapter(EngineAdapter):
    name = "meilisearch"
    host_port = 17700
    health_path = "/health"
    data_dir = "/meili_data"

    def __init__(self, ctx: IndexContext):
        super().__init__(ctx)
        self._pending: list[int] = []

    def index_settings(self) -> dict[str, Any]:
        s = self.ctx.scenario
        f = s.index_fields
        settings = {
            "searchableAttributes": list(f.searchable),
            "filterableAttributes": sorted(set(f.filterable) | set(f.facets)),
            "sortableAttributes": list(f.sortable),
            "synonyms": synonyms_setting(self.ctx.synonyms),
            "faceting": {"maxValuesPerFacet": s.max_facet_values},
            "embedders": {EMBEDDER: {"source": "userProvided", "dimensions": self.ctx.dim}},
        }
        return deep_merge(settings, self.ctx.config.index)

    def _wait(self, client: httpx.Client, uid: int, timeout_s: float) -> None:
        def done() -> bool:
            task = call(client, Request("GET", f"/tasks/{uid}"))
            if task["status"] in ("failed", "canceled"):
                raise EngineError(f"meilisearch task {uid} {task['status']}: {task.get('error')}")
            return task["status"] == "succeeded"

        poll(done, timeout_s, f"meilisearch task {uid}", interval_s=0.2)

    def create_index(self, client: httpx.Client) -> None:
        idx = self.ctx.index
        self._pending = []
        existing = Request("GET", f"/indexes/{idx}").send(client)
        if existing.status_code == 200:
            deleted = call(client, Request("DELETE", f"/indexes/{idx}"))
            self._wait(client, deleted["taskUid"], SETUP_TIMEOUT_S)
        elif existing.status_code != 404:
            raise EngineError(
                f"GET /indexes/{idx} -> {existing.status_code}: {existing.text[:500]}"
            )
        body = {"uid": idx, "primaryKey": self.ctx.scenario.id_field}
        self._wait(
            client, call(client, Request("POST", "/indexes", body=body))["taskUid"], SETUP_TIMEOUT_S
        )
        settings = Request("PATCH", f"/indexes/{idx}/settings", body=self.index_settings())
        self._wait(client, call(client, settings)["taskUid"], SETUP_TIMEOUT_S)

    def load_documents(
        self, client: httpx.Client, docs: list[dict[str, Any]], vectors: np.ndarray
    ) -> None:
        for start in range(0, len(docs), BATCH_SIZE):
            batch = [
                {**doc, "_vectors": {EMBEDDER: vectors[i].tolist()}}
                for i, doc in enumerate(docs[start : start + BATCH_SIZE], start=start)
            ]
            task = call(client, Request("POST", f"/indexes/{self.ctx.index}/documents", body=batch))
            self._pending.append(task["taskUid"])

    def wait_until_indexed(self, client: httpx.Client, n_docs: int, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        for uid in self._pending:
            self._wait(client, uid, max(0.0, deadline - time.monotonic()))
        stats = call(client, Request("GET", f"/indexes/{self.ctx.index}/stats"))
        if stats["numberOfDocuments"] != n_docs:
            raise EngineError(
                f"meilisearch holds {stats['numberOfDocuments']} documents, expected {n_docs}"
            )

    def render(
        self,
        task: QueryTask,
        vector: list[float] | None,
        limit: int,
        facets: bool = True,
        *,
        depth: int | None = None,
    ) -> Request:
        body: dict[str, Any] = {}
        if task.base in ("keyword", "hybrid"):
            body["q"] = task.text
        if task.base in ("semantic", "hybrid"):
            body["vector"] = require_vector(task, vector)
            body["hybrid"] = {"embedder": EMBEDDER}
            if task.base == "semantic":
                body["hybrid"]["semanticRatio"] = 1.0
        body["limit"] = limit
        body["attributesToRetrieve"] = [self.ctx.scenario.id_field]
        if task.filter is not None:
            body["filter"] = filter_expression(task.filter)
        if task.sort is not None:
            body["sort"] = [f"{task.sort.field}:{task.sort.direction}"]
        if facets and task.facets:
            body["facets"] = list(task.facets)
        body = deep_merge(body, self.search_params(task))
        return Request("POST", f"/indexes/{self.ctx.index}/search", body=body)

    def parse(self, task: QueryTask, payload: Any) -> SearchResult:
        id_field = self.ctx.scenario.id_field
        return SearchResult(
            ids=[str(hit[id_field]) for hit in payload["hits"]],
            facets=payload.get("facetDistribution"),
        )
