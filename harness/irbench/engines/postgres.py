"""Shared Postgres adapter (spec 2026-10-12 §4-§5). Subclasses provide keyword search only."""

import re
from abc import abstractmethod
from typing import Any

import numpy as np

from irbench.engines.base import EngineAdapter, SearchResult, SqlRequest, require_vector
from irbench.models import Filter, QueryTask

COPY_BATCH = 2000
RRF_K = 60


class Sql:
    """Collects params; every placeholder is `$n::text`, optionally cast further."""

    def __init__(self) -> None:
        self.params: list[Any] = []

    def p(self, value: Any, cast: str | None = None) -> str:
        self.params.append(value)
        ph = f"${len(self.params)}::text"
        return f"{ph}::{cast}" if cast else ph


def ident(name: str) -> str:
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
        raise ValueError(f"unsafe SQL identifier {name!r}")
    return name


class PostgresAdapter(EngineAdapter):
    protocol = "postgres"
    health_path = ""
    data_dir = "/var/lib/postgresql/data"

    # --- schema ---------------------------------------------------------------
    @property
    def table(self) -> str:
        # ctx.index embeds the model id (e.g. "bge-small-en-v1.5"): make it an identifier.
        return ident(re.sub(r"[^a-z0-9_]+", "_", self.ctx.index.lower()))

    @property
    def searchable(self) -> list[str]:
        return [ident(f) for f in self.ctx.scenario.index_fields.searchable]

    @property
    def numeric(self) -> list[str]:
        return [ident(f) for f in self.ctx.scenario.index_fields.numeric]

    @property
    def term_fields(self) -> list[str]:
        """Non-numeric filter/facet fields, stored as text[] in `<field>__terms`."""
        f = self.ctx.scenario.index_fields
        names = sorted((set(f.filterable) | set(f.facets)) - set(f.numeric))
        return [ident(n) for n in names]

    def columns(self) -> list[tuple[str, str]]:
        cols = [("id", "text PRIMARY KEY")]
        cols += [(f, "text") for f in self.searchable]
        cols += [(f, "double precision") for f in self.numeric]
        cols += [(f"{f}__terms", "text[]") for f in self.term_fields]
        cols += [("embedding", f"vector({int(self.ctx.dim)})")]
        return cols

    def create_index(self, client: Any) -> None:
        cols = ",\n  ".join(f"{n} {t}" for n, t in self.columns() + self.extra_columns())
        client.script(
            f"{self.setup_ddl()}\n"
            f"DROP TABLE IF EXISTS {self.table};\n"
            f"CREATE TABLE {self.table} (\n  {cols}\n);"
        )
        self.apply_settings(client)

    def apply_settings(self, client: Any) -> None:
        """index.settings from the config (e.g. hnsw.ef_search): per database and this session."""
        for name, value in (self.ctx.config.index.get("settings") or {}).items():
            if not re.fullmatch(r"[a-z_]+\.[a-z_]+", name) or not re.fullmatch(
                r"[A-Za-z0-9_.]+", str(value)
            ):
                raise ValueError(f"unsafe setting {name}={value!r}")
            client.script(f"ALTER DATABASE irbench SET {name} = {value}; SET {name} = {value};")

    def load_documents(self, client: Any, docs: list[dict[str, Any]], vectors: np.ndarray) -> None:
        id_field = self.ctx.scenario.id_field
        names = [n for n, _ in self.columns()]
        rows = []
        for i, doc in enumerate(docs):
            row: list[Any] = [str(doc[id_field])]
            row += [_text(doc.get(f)) for f in self.searchable]
            row += [doc.get(f) for f in self.numeric]
            row += [_terms(doc.get(f)) for f in self.term_fields]
            row += ["[" + ",".join(repr(float(x)) for x in vectors[i]) + "]"]
            rows.append(row)
            if len(rows) == COPY_BATCH:
                client.copy(self.table, names, rows)
                rows = []
        if rows:
            client.copy(self.table, names, rows)

    def wait_until_indexed(self, client: Any, n_docs: int, timeout_s: float) -> None:
        t = self.table
        stmts = [
            self.text_index_ddl(),
            f"CREATE INDEX ON {t} USING hnsw (embedding vector_cosine_ops)",
        ]
        stmts += [f"CREATE INDEX ON {t} ({f})" for f in self.numeric]
        stmts += [f"CREATE INDEX ON {t} USING gin ({f}__terms)" for f in self.term_fields]
        client.script(";\n".join(stmts) + f";\nANALYZE {t};")
        count = client.execute(f"SELECT count(*) AS n FROM {t}")[0]["n"]
        if count != n_docs:
            raise RuntimeError(f"{t} has {count} rows, expected {n_docs}")

    # --- hooks ----------------------------------------------------------------
    def setup_ddl(self) -> str:
        return "CREATE EXTENSION IF NOT EXISTS vector;"

    def extra_columns(self) -> list[tuple[str, str]]:
        return []

    @abstractmethod
    def text_index_ddl(self) -> str: ...

    @abstractmethod
    def keyword_match(self, q: Sql, text: str) -> tuple[str, str]:
        """(WHERE predicate, score expression where higher is better)."""

    # --- queries --------------------------------------------------------------
    def _filter(self, q: Sql, f: Filter | None) -> str:
        if f is None:
            return "TRUE"
        field = ident(f.field)
        if f.op == "eq":
            return f"{field}__terms @> ARRAY[{q.p(str(f.value))}]"
        op = {"gte": ">=", "lte": "<="}[f.op]
        return f"{field} {op} {q.p(float(f.value), 'float8')}"

    def _ranked(self, q: Sql, task: QueryTask, vector: list[float] | None, k_limit: int) -> str:
        """SELECT *, score over the retrieval base, best first, at most k_limit rows.

        Params are added in statement order: query text, then vector, then filter, then limits.
        """
        t = self.table
        if task.base == "keyword":
            pred, score = self.keyword_match(q, task.text)
            where = self._filter(q, task.filter)
            lim = q.p(k_limit, "int")
            return f"SELECT *, {score} AS score FROM {t} WHERE {pred} AND {where} ORDER BY score DESC, id LIMIT {lim}"
        if task.base == "semantic":
            vec = q.p(require_vector(task, vector), "vector")
            where = self._filter(q, task.filter)
            lim = q.p(k_limit, "int")
            return f"SELECT *, -(embedding <=> {vec}) AS score FROM {t} WHERE {where} ORDER BY embedding <=> {vec}, id LIMIT {lim}"
        # hybrid: RRF over the keyword and vector top_k (spec §7 amendment)
        pred, score = self.keyword_match(q, task.text)
        vec = q.p(require_vector(task, vector), "vector")
        where = self._filter(q, task.filter)
        top_k = q.p(self.ctx.scenario.top_k, "int")
        lim = q.p(k_limit, "int")
        return (
            f"SELECT d.*, coalesce(1.0 / ({RRF_K} + kw.r), 0) + coalesce(1.0 / ({RRF_K} + vec.r), 0) AS score "
            f"FROM (SELECT id, row_number() OVER (ORDER BY {score} DESC, id) AS r FROM {t} "
            f"WHERE {pred} AND {where} ORDER BY {score} DESC, id LIMIT {top_k}) kw "
            f"FULL OUTER JOIN (SELECT id, row_number() OVER (ORDER BY embedding <=> {vec}, id) AS r FROM {t} "
            f"WHERE {where} ORDER BY embedding <=> {vec}, id LIMIT {top_k}) vec ON kw.id = vec.id "
            f"JOIN {t} d ON d.id = coalesce(kw.id, vec.id) ORDER BY score DESC, d.id LIMIT {lim}"
        )

    def render(
        self, task: QueryTask, vector: list[float] | None, limit: int, facets: bool = True
    ) -> SqlRequest:
        q = Sql()
        top_k = self.ctx.scenario.top_k
        if task.sort is not None:
            field, direction = ident(task.sort.field), task.sort.direction.upper()
            # keyword base: every match; semantic base: the top_k nearest (same rules as other engines)
            pool = (self.ctx.n_docs or 1_000_000_000) if task.base == "keyword" else top_k
            inner = self._ranked(q, task, vector, pool)
            sql = f"SELECT id FROM ({inner}) m ORDER BY {field} {direction} NULLS LAST, id LIMIT {q.p(limit, 'int')}"
        elif facets and task.facets:
            names = [ident(f) for f in task.facets]
            inner = self._ranked(q, task, vector, self.ctx.n_docs or 1_000_000_000)
            lim = q.p(limit, "int")
            facet_sql = ", ".join(
                f"'{n}', (SELECT coalesce(json_object_agg(v, c), '{{}}') FROM "
                f"(SELECT unnest({n}__terms) AS v, count(*) AS c FROM m GROUP BY 1) x)"
                for n in names
            )
            sql = (
                f"WITH m AS ({inner}), hits AS (SELECT id, row_number() OVER (ORDER BY score DESC, id) AS o "
                f"FROM m ORDER BY score DESC, id LIMIT {lim}) "
                f"SELECT (SELECT coalesce(json_agg(id ORDER BY o), '[]') FROM hits) AS ids, "
                f"json_build_object({facet_sql}) AS facets"
            )
        else:
            sql = (
                f"SELECT id FROM ({self._ranked(q, task, vector, limit)}) m ORDER BY score DESC, id"
            )
        params = q.params
        extra = self.search_params(task)
        if extra:
            raise ValueError(
                f"{self.name}: search.{task.base} settings are not supported: {sorted(extra)}"
            )
        return SqlRequest(sql=sql, params=params)

    def parse(self, task: QueryTask, payload: Any) -> SearchResult:
        rows = payload or []
        if rows and "ids" in rows[0]:
            return SearchResult(ids=[str(i) for i in rows[0]["ids"]], facets=rows[0]["facets"])
        return SearchResult(ids=[str(r["id"]) for r in rows])


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) or None
    return str(value) or None


def _terms(value: Any) -> list[str] | None:
    if value is None:
        return None
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]
