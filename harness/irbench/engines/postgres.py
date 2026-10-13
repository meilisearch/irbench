"""Shared Postgres adapter (spec 2026-10-12 §4-§5). Subclasses implement `keyword_match` and `text_index_ddl` and may override `setup_ddl` and `extra_columns`."""

import re
from abc import abstractmethod
from dataclasses import dataclass
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
        """index.settings from the config (dotted extension GUCs like hnsw.ef_search, or core GUCs like
        plan_cache_mode): per database and this session."""
        settings = self.ctx.config.index.get("settings") or {}
        stmts = ["ALTER DATABASE {db} RESET ALL"]
        session = []
        for name, value in settings.items():
            if not re.fullmatch(r"[a-z_]+(\.[a-z_]+)?", name) or not re.fullmatch(
                r"[A-Za-z0-9_.]+", str(value)
            ):
                raise ValueError(f"unsafe setting {name}={value!r}")
            stmts.append(f"ALTER DATABASE {{db}} SET {name} = {value}")
            session.append(f"SET {name} = {value}")
        body = "; ".join(
            f"EXECUTE format('{st}', current_database())".replace("{db}", "%I") for st in stmts
        )
        client.script(f"DO $$ BEGIN {body}; END $$;" + "".join(f"\n{st};" for st in session))

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

    def keyword_tiebreak(self) -> str:
        """Suffix after `score DESC` wherever the keyword base is ranked by score. ", id" makes
        the order total; an engine whose BM25 scan can only push a bare score order down to
        the index (Top-K) returns "" and accepts arbitrary order among equal scores."""
        return ", id"

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

    def _parts(self, q: Sql, task: QueryTask, vector: list[float] | None) -> "_Parts":
        """Placeholders for the task's values, allocated once in order: text, vector, filter."""
        pred = score = vec = None
        if task.base in ("keyword", "hybrid"):
            pred, score = self.keyword_match(q, task.text)
        if task.base in ("semantic", "hybrid"):
            vec = q.p(require_vector(task, vector), "vector")
        return _Parts(pred, score, vec, self._filter(q, task.filter))

    def _ranked(
        self,
        q: Sql,
        task: QueryTask,
        p: "_Parts",
        k_limit: int | None,
        depth: int,
        extras: list[str] = (),
        scored: bool = True,
    ) -> str:
        """SELECT id [, score] [, extras] over the retrieval base, best first, at most k_limit rows.

        k_limit None means every match: keyword base only, unordered. `depth` is the per-side
        pool of the hybrid RRF. Only the columns callers
        need are projected, so no wide column (embedding, text, tsv) travels through a sort.
        Params are added in statement order: query text, vector, filter, then limits.
        """
        t = self.table
        ext = "".join(f", {e}" for e in extras)
        if task.base == "keyword":
            sc = f", {p.score} AS score" if scored else ""
            head = f"SELECT id{sc}{ext} FROM {t} WHERE {p.pred} AND {p.where}"
            if k_limit is None:
                return head
            order = "score" if scored else p.score
            tb = self.keyword_tiebreak()
            return f"{head} ORDER BY {order} DESC{tb} LIMIT {q.p(k_limit, 'int')}"
        if k_limit is None:
            raise ValueError("an unbounded pool is only defined for the keyword base")
        if task.base == "semantic":
            sc = f", -(embedding <=> {p.vec}) AS score" if scored else ""
            return (
                f"SELECT id{sc}{ext} FROM {t} WHERE {p.where} "
                f"ORDER BY embedding <=> {p.vec}, id LIMIT {q.p(k_limit, 'int')}"
            )
        # hybrid: RRF over the keyword and vector top `depth` (spec §7 amendment)
        side = q.p(depth, "int")
        tb = self.keyword_tiebreak()
        fused = (
            f"SELECT coalesce(kw.id, vec.id) AS id, coalesce(1.0 / ({RRF_K} + kw.r), 0) "
            f"+ coalesce(1.0 / ({RRF_K} + vec.r), 0) AS score "
            f"FROM (SELECT id, row_number() OVER (ORDER BY {p.score} DESC{tb}) AS r FROM {t} "
            f"WHERE {p.pred} AND {p.where} ORDER BY {p.score} DESC{tb} LIMIT {side}) kw "
            f"FULL OUTER JOIN (SELECT id, row_number() OVER (ORDER BY embedding <=> {p.vec}, id) AS r "
            f"FROM {t} WHERE {p.where} ORDER BY embedding <=> {p.vec}, id LIMIT {side}) vec "
            f"ON kw.id = vec.id"
        )
        lim = q.p(k_limit, "int")
        if not extras:
            return f"SELECT id, score FROM ({fused}) f ORDER BY score DESC, id LIMIT {lim}"
        cols = "".join(f", d.{e}" for e in extras)
        return (
            f"SELECT f.id, f.score{cols} FROM ({fused}) f JOIN {t} d ON d.id = f.id "
            f"ORDER BY f.score DESC, f.id LIMIT {lim}"
        )

    def render(
        self,
        task: QueryTask,
        vector: list[float] | None,
        limit: int,
        facets: bool = True,
        *,
        depth: int | None = None,
    ) -> SqlRequest:
        q = Sql()
        depth = self.pool_depth(depth)
        t = self.table
        p = self._parts(q, task, vector)
        if task.sort is not None:
            field, direction = ident(task.sort.field), task.sort.direction.upper()
            # keyword base: every match; other bases: the depth (same rules as other engines)
            pool = None if task.base == "keyword" else depth
            inner = self._ranked(q, task, p, pool, depth, [field], scored=False)
            sql = (
                f"SELECT id FROM ({inner}) m ORDER BY {field} {direction} NULLS LAST, id "
                f"LIMIT {q.p(limit, 'int')}"
            )
        elif facets and task.facets:
            terms = [f"{ident(f)}__terms" for f in task.facets]
            every = self.ctx.n_docs or 1_000_000_000
            if task.base == "keyword":
                # counts over every match, hits ranked from the same match set
                m = self._ranked(q, task, p, None, depth, terms)
                hits_src, order = "m", f"score DESC{self.keyword_tiebreak()}"
            elif task.base == "semantic":
                # counts over every filtered document with no ordering or limit (an HNSW scan
                # would be capped by hnsw.max_scan_tuples); hits from a separate nearest query
                m = f"SELECT {', '.join(terms)} FROM {t} WHERE {p.where}"
                hits_src, order = f"{t} WHERE {p.where}", f"embedding <=> {p.vec}, id"
            else:
                m = self._ranked(q, task, p, every, depth, terms)
                hits_src, order = "m", "score DESC, id"
            lim = q.p(limit, "int")
            # at most max_facet_values buckets per facet, most frequent first (as ES/Meilisearch)
            cap = q.p(self.ctx.scenario.max_facet_values, "int")
            facet_sql = ", ".join(
                f"'{n}', (SELECT coalesce(json_object_agg(v, c), '{{}}') FROM "
                f"(SELECT unnest({n}__terms) AS v, count(*) AS c FROM m GROUP BY 1 "
                f"ORDER BY c DESC, v LIMIT {cap}) x)"
                for n in (ident(f) for f in task.facets)
            )
            sql = (
                f"WITH m AS ({m}), hits AS (SELECT id, row_number() OVER (ORDER BY {order}) AS o "
                f"FROM {hits_src} ORDER BY {order} LIMIT {lim}) "
                f"SELECT (SELECT coalesce(json_agg(id ORDER BY o), '[]') FROM hits) AS ids, "
                f"json_build_object({facet_sql}) AS facets"
            )
        else:
            tb = self.keyword_tiebreak() if task.base == "keyword" else ", id"
            sql = f"SELECT id FROM ({self._ranked(q, task, p, limit, depth)}) m ORDER BY score DESC{tb}"
        extra = self.search_params(task)
        if extra:
            raise ValueError(
                f"{self.name}: search.{task.base} settings are not supported: {sorted(extra)}"
            )
        return SqlRequest(sql=sql, params=q.params)

    def parse(self, task: QueryTask, payload: Any) -> SearchResult:
        rows = payload or []
        if rows and "ids" in rows[0]:
            return SearchResult(ids=[str(i) for i in rows[0]["ids"]], facets=rows[0]["facets"])
        return SearchResult(ids=[str(r["id"]) for r in rows])


@dataclass(frozen=True)
class _Parts:
    pred: str | None
    score: str | None
    vec: str | None
    where: str


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
