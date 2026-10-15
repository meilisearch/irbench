"""Elasticsearch adapter. Parity notes: engines/elasticsearch/README.md."""

import json
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

VECTOR_FIELD = "vector"
SYNONYMS_FILTER = "irbench_synonyms"
SEARCH_ANALYZER = "irbench_search"
BATCH_SIZE = 1000


class Adapter(EngineAdapter):
    name = "elasticsearch"
    host_port = 19200
    health_path = "/_cluster/health?wait_for_status=yellow&timeout=1s"
    data_dir = "/usr/share/elasticsearch/data"

    def _exact_fields(self) -> set[str]:
        f = self.ctx.scenario.index_fields
        return (set(f.filterable) | set(f.facets)) - set(f.numeric)

    def exact_field(self, name: str) -> str:
        return f"{name}.raw" if name in self.ctx.scenario.index_fields.searchable else name

    def mappings(self) -> dict[str, Any]:
        f = self.ctx.scenario.index_fields
        exact = self._exact_fields()
        props: dict[str, Any] = {}
        for name in f.searchable:
            props[name] = {"type": "text", "search_analyzer": SEARCH_ANALYZER}
            if name in exact:
                props[name]["fields"] = {"raw": {"type": "keyword"}}
        for name in sorted(exact - set(f.searchable)):
            props[name] = {"type": "keyword"}
        for name in f.numeric:
            props[name] = {"type": "float"}
        props[VECTOR_FIELD] = {
            "type": "dense_vector",
            "dims": self.ctx.dim,
            "index": True,
            "similarity": "cosine",
        }
        return {"dynamic": False, "properties": props}

    def index_body(self) -> dict[str, Any]:
        filters = ["lowercase"]
        analysis: dict[str, Any] = {}
        if self.ctx.synonyms:
            analysis["filter"] = {
                SYNONYMS_FILTER: {
                    "type": "synonym_graph",
                    "synonyms": [", ".join(group) for group in self.ctx.synonyms],
                }
            }
            filters.append(SYNONYMS_FILTER)
        analysis["analyzer"] = {
            SEARCH_ANALYZER: {"type": "custom", "tokenizer": "standard", "filter": filters}
        }
        body = {
            "settings": {"number_of_replicas": 0, "analysis": analysis},
            "mappings": self.mappings(),
        }
        return deep_merge(body, self.ctx.config.index)

    def create_index(self, client: httpx.Client) -> None:
        idx = self.ctx.index
        response = Request("DELETE", f"/{idx}").send(client)
        if response.status_code not in (200, 404):
            raise EngineError(f"DELETE /{idx} -> {response.status_code}: {response.text[:500]}")
        call(client, Request("PUT", f"/{idx}", body=self.index_body()))

    def load_documents(
        self, client: httpx.Client, docs: list[dict[str, Any]], vectors: np.ndarray
    ) -> None:
        idx, id_field = self.ctx.index, self.ctx.scenario.id_field
        for start in range(0, len(docs), BATCH_SIZE):
            lines = []
            for i, doc in enumerate(docs[start : start + BATCH_SIZE], start=start):
                lines.append(json.dumps({"index": {"_index": idx, "_id": str(doc[id_field])}}))
                lines.append(json.dumps({**doc, VECTOR_FIELD: vectors[i].tolist()}))
            bulk = Request(
                "POST",
                "/_bulk",
                content="\n".join(lines) + "\n",
                headers={"Content-Type": "application/x-ndjson"},
            )
            result = call(client, bulk)
            if result.get("errors"):
                first = next(
                    item["index"]["error"] for item in result["items"] if item["index"].get("error")
                )
                raise EngineError(f"bulk indexing failed: {first}")

    def wait_until_indexed(self, client: httpx.Client, n_docs: int, timeout_s: float) -> None:
        idx = self.ctx.index
        call(client, Request("POST", f"/{idx}/_refresh"))

        def ready() -> bool:
            return call(client, Request("GET", f"/{idx}/_count"))["count"] == n_docs

        poll(ready, timeout_s, f"{idx} to hold {n_docs} documents")

    def _text_query(self, text: str) -> dict[str, Any]:
        fields = self.ctx.scenario.index_fields.searchable
        n = len(fields)
        boosted = [f"{name}^{n - i}" for i, name in enumerate(fields)]
        return {"multi_match": {"query": text, "fields": boosted}}

    def _filter_clause(self, f: Filter) -> dict[str, Any]:
        if f.op == "eq":
            return {"term": {self.exact_field(f.field): f.value}}
        return {"range": {f.field: {f.op: float(f.value)}}}

    def render(
        self,
        task: QueryTask,
        vector: list[float] | None,
        limit: int,
        facets: bool = True,
        *,
        depth: int | None = None,
    ) -> Request:
        scenario = self.ctx.scenario
        depth = self.pool_depth(depth)
        body: dict[str, Any] = {"size": limit, "_source": False}
        filters = [self._filter_clause(task.filter)] if task.filter is not None else []
        if task.base in ("keyword", "hybrid"):
            query = self._text_query(task.text)
            body["query"] = {"bool": {"must": [query], "filter": filters}} if filters else query
        if task.base in ("semantic", "hybrid"):
            knn: dict[str, Any] = {
                "field": VECTOR_FIELD,
                "query_vector": require_vector(task, vector),
                "k": depth,
            }
            if filters:
                knn["filter"] = filters
            body["knn"] = knn
        if task.sort is not None:
            body["sort"] = [{task.sort.field: {"order": task.sort.direction}}]
        if facets and task.facets:
            body["aggs"] = {
                name: {
                    "terms": {"field": self.exact_field(name), "size": scenario.max_facet_values}
                }
                for name in task.facets
            }
        body = deep_merge(body, self.search_params(task))
        return Request("POST", f"/{self.ctx.index}/_search", body=body)

    def parse(self, task: QueryTask, payload: Any) -> SearchResult:
        aggs = payload.get("aggregations")
        facets = None
        if aggs is not None:
            facets = {
                name: {str(b["key"]): int(b["doc_count"]) for b in agg["buckets"]}
                for name, agg in aggs.items()
            }
        return SearchResult(ids=[hit["_id"] for hit in payload["hits"]["hits"]], facets=facets)
