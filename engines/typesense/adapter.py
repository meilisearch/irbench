"""Typesense adapter. Parity notes: engines/typesense/README.md."""

import json
import string
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

API_KEY = "irbench"  # local benchmark only; set in docker/compose.yaml
HEADERS = {"X-TYPESENSE-API-KEY": API_KEY}
VECTOR_FIELD = "vector"
BATCH_SIZE = 1000
# Typesense returns at most 250 hits per page; larger fetches (the facet check) span pages.
MAX_PER_PAGE = 250
# Typesense drops symbols from indexed strings, so `tags:=webserver` would also match
# `web-server`. Exact-match fields keep every symbol (the backtick cannot be quoted anyway).
EXACT_SYMBOLS = [c for c in string.punctuation if c != "`"]
# Config keys under `index` that the adapter applies itself instead of merging them into the
# collection body.
SEARCHABLE_FIELDS = "searchable_fields"  # merged into every searchable field definition
FIELDS = "fields"  # {name: {...}} merged into that field's definition
STOPWORDS = "stopwords"  # {set name: [words]} created as stopword sets
# Search config key the adapter applies itself: {option: value} appended inside `vector_query`
# (e.g. `alpha`, the vector side's weight in hybrid rank fusion).
VECTOR_QUERY_OPTIONS = "vector_query_options"


def _quote(value: str) -> str:
    # Backticks quote a filter value; a backtick inside the value cannot be escaped.
    if "`" in value:
        raise ValueError(f"typesense filter values cannot contain a backtick: {value!r}")
    return f"`{value}`"


def filter_expression(f: Filter) -> str:
    if f.op == "eq":
        return f"{f.field}:={_quote(str(f.value))}"
    return f"{f.field}:{'>=' if f.op == 'gte' else '<='}{float(f.value)}"


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


class Adapter(EngineAdapter):
    name = "typesense"
    host_port = 18108
    health_path = "/health"
    data_dir = "/data"

    @property
    def synonym_set(self) -> str:
        return f"{self.ctx.index}_synonyms"

    def _index_config(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        config = dict(self.ctx.config.index)
        searchable = config.pop(SEARCHABLE_FIELDS, {})
        fields = config.pop(FIELDS, {})
        config.pop(STOPWORDS, None)
        return config, searchable, fields

    def fields(self) -> list[dict[str, Any]]:
        f = self.ctx.scenario.index_fields
        _, searchable_opts, field_opts = self._index_config()
        facets = set(f.facets)
        out: list[dict[str, Any]] = []
        # `string*` takes a string or a list of strings; every field is optional because the
        # datasets leave some values empty.
        for name in f.searchable:
            field = {"name": name, "type": "string*", "optional": True, "facet": name in facets}
            out.append(deep_merge(field, searchable_opts))
        for name in sorted((set(f.filterable) | facets) - set(f.searchable) - set(f.numeric)):
            out.append(
                {
                    "name": name,
                    "type": "string*",
                    "optional": True,
                    "facet": name in facets,
                    "symbols_to_index": EXACT_SYMBOLS,
                }
            )
        for name in sorted((set(f.numeric) | set(f.sortable)) - set(f.searchable)):
            out.append({"name": name, "type": "float", "optional": True, "facet": name in facets})
        out.append({"name": VECTOR_FIELD, "type": "float[]", "num_dim": self.ctx.dim})
        unknown = set(field_opts) - {field["name"] for field in out}
        if unknown:
            raise ValueError(f"index.fields names unknown fields: {sorted(unknown)}")
        return [deep_merge(field, field_opts.get(field["name"], {})) for field in out]

    def collection_body(self) -> dict[str, Any]:
        rest, _, _ = self._index_config()
        body: dict[str, Any] = {"name": self.ctx.index, "fields": self.fields()}
        if self.ctx.synonyms:
            body["synonym_sets"] = [self.synonym_set]
        return deep_merge(body, rest)

    def synonyms_body(self) -> dict[str, Any]:
        return {
            "items": [
                {"id": f"g{i}", "synonyms": list(group)}
                for i, group in enumerate(self.ctx.synonyms)
            ]
        }

    def create_index(self, client: httpx.Client) -> None:
        c = self.ctx.index
        response = Request("DELETE", f"/collections/{c}", headers=HEADERS).send(client)
        if response.status_code not in (200, 404):
            raise EngineError(
                f"DELETE /collections/{c} -> {response.status_code}: {response.text[:500]}"
            )
        if self.ctx.synonyms:
            path = f"/synonym_sets/{self.synonym_set}"
            call(client, Request("PUT", path, body=self.synonyms_body(), headers=HEADERS))
        for name, words in self.ctx.config.index.get(STOPWORDS, {}).items():
            body = {"stopwords": list(words), "locale": "en"}
            call(client, Request("PUT", f"/stopwords/{name}", body=body, headers=HEADERS))
        call(client, Request("POST", "/collections", body=self.collection_body(), headers=HEADERS))

    def load_documents(
        self, client: httpx.Client, docs: list[dict[str, Any]], vectors: np.ndarray
    ) -> None:
        id_field = self.ctx.scenario.id_field
        for start in range(0, len(docs), BATCH_SIZE):
            lines = []
            for i, doc in enumerate(docs[start : start + BATCH_SIZE], start=start):
                body = {k: v for k, v in doc.items() if v is not None}
                body["id"] = str(doc[id_field])
                body[VECTOR_FIELD] = vectors[i].tolist()
                lines.append(json.dumps(body))
            response = Request(
                "POST",
                f"/collections/{self.ctx.index}/documents/import?action=create",
                content="\n".join(lines),
                headers={**HEADERS, "Content-Type": "text/plain"},
            ).send(client, timeout=600)
            if response.status_code >= 400:
                raise EngineError(f"import -> {response.status_code}: {response.text[:500]}")
            for line in response.text.splitlines():
                result = json.loads(line)
                if not result.get("success"):
                    raise EngineError(f"typesense import failed: {line[:500]}")

    def wait_until_indexed(self, client: httpx.Client, n_docs: int, timeout_s: float) -> None:
        # Imports are synchronous; this confirms the count.
        def ready() -> bool:
            info = call(client, Request("GET", f"/collections/{self.ctx.index}", headers=HEADERS))
            return info["num_documents"] == n_docs

        poll(ready, timeout_s, f"{self.ctx.index} to hold {n_docs} documents")

    def _search(
        self, task: QueryTask, vector: list[float] | None, depth: int, facets: bool
    ) -> dict[str, Any]:
        scenario = self.ctx.scenario
        searchable = scenario.index_fields.searchable
        n = len(searchable)
        s: dict[str, Any] = {"collection": self.ctx.index, "include_fields": "id"}
        if task.base in ("keyword", "hybrid"):
            s["q"] = task.text
            s["query_by"] = ",".join(searchable)
            # Field priority as weights n - i, the same boosts the other engines get.
            s["query_by_weights"] = ",".join(str(n - i) for i in range(n))
        else:
            s["q"] = "*"
        params = dict(self.search_params(task))
        options = params.pop(VECTOR_QUERY_OPTIONS, {})
        if task.base in ("semantic", "hybrid"):
            literal = vector_literal(require_vector(task, vector))
            extra = "".join(f", {k}:{v}" for k, v in options.items())
            s["vector_query"] = f"{VECTOR_FIELD}:({literal}, k:{depth}{extra})"
        if task.filter is not None:
            s["filter_by"] = filter_expression(task.filter)
        if task.sort is not None:
            # Documents without the field go last, as the scenario's order check expects.
            s["sort_by"] = f"{task.sort.field}(missing_values: last):{task.sort.direction}"
        if facets and task.facets:
            s["facet_by"] = ",".join(task.facets)
            s["max_facet_values"] = scenario.max_facet_values
        return deep_merge(s, params)

    def render(
        self,
        task: QueryTask,
        vector: list[float] | None,
        limit: int,
        facets: bool = True,
        *,
        depth: int | None = None,
    ) -> Request:
        search = self._search(task, vector, self.pool_depth(depth), facets)
        if limit <= MAX_PER_PAGE:
            searches = [{**search, "per_page": limit}]
        else:  # one request, consecutive pages; parse concatenates their hits
            pages = -(-limit // MAX_PER_PAGE)
            searches = [
                {**search, "per_page": MAX_PER_PAGE, "page": p} for p in range(1, pages + 1)
            ]
        return Request("POST", "/multi_search", body={"searches": searches}, headers=HEADERS)

    def parse(self, task: QueryTask, payload: Any) -> SearchResult:
        # multi_search answers 200 even when a search fails; the error is in the result.
        ids: list[str] = []
        facets = None
        for i, result in enumerate(payload["results"]):
            if "error" in result:
                raise EngineError(
                    f"typesense search failed: {result.get('code')} {result['error']}"
                )
            ids += [str(hit["document"]["id"]) for hit in result["hits"]]
            if i == 0 and task.facets and "facet_counts" in result:
                facets = {
                    fc["field_name"]: {str(c["value"]): int(c["count"]) for c in fc["counts"]}
                    for fc in result["facet_counts"]
                }
        return SearchResult(ids=ids, facets=facets)
