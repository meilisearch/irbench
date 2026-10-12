# IRBench Postgres Engines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark Postgres as two engines — `postgres` (Postgres 17 + pgvector, built-in full-text search) and `postgres-paradedb` (ParadeDB: pg_search BM25 + pgvector) — through the full IRBench pipeline (relevance and performance), next to Meilisearch, Elasticsearch and Qdrant.

**Architecture:** Rendered requests gain a second protocol, `postgres` (`{sql, params}`), executed by a psycopg client in the harness and by a `tokio-postgres` target in the Rust loadgen (one connection per concurrent client). Every parameter is bound as text and cast in the SQL (`$1::text`, `$2::text::vector`), so both clients bind values identically without custom Postgres types. One shared `PostgresAdapter` base (schema, `COPY` loading, filters, sorting, facets, hybrid RRF) has two subclasses that differ only in keyword search: `tsvector`/`ts_rank` vs pg_search BM25.

**Tech Stack:** Python 3.12 harness (existing) + `psycopg[binary] >= 3.2` (`RawCursor` for `$n` placeholders); Rust loadgen (existing) + `tokio-postgres 0.7`; Docker images `pgvector/pgvector:0.8.7-pg17` and `paradedb/paradedb:0.26.1-pg17`.

**Spec:** `docs/superpowers/specs/2026-10-12-irbench-postgres-design.md` (parent: `docs/superpowers/specs/2026-10-09-irbench-design.md`).

## Spike findings (verified 2026-10-12 on the pinned images; the plan relies on them)

- `paradedb/paradedb:0.26.1-pg17` ships `pg_search 0.26.1` and `vector 0.8.6`, already installed. `pgvector/pgvector:0.8.7-pg17` ships `vector 0.8.7`.
- pg_search: `CREATE INDEX ... USING bm25 (id, col, ...)` (the `key_field` option is deprecated and a no-op in 0.26 — do not pass it). Per-field query `col ||| 'text'::pdb.boost(5)`, combined with `OR`; score with `pdb.score(id)`; `EXPLAIN` shows `Custom Scan (ParadeDB Base Scan) ... Index: <bm25 index>`. The default tokenizer does not stem ("beds" does not match "bed").
- pg_search 0.26.1 has no synonym support (no `synonym` function in the `pdb`/`paradedb` schemas) → `postgres-paradedb` declares `synonyms: <reason>`.
- Vanilla: a synonym dictionary (`TEMPLATE = synonym`) works when its file is in `$(pg_config --sharedir)/tsearch_data/` (`/usr/share/postgresql/17/tsearch_data`, root-owned → mount the file read-only via compose). `setweight` A > D ranking works with `ts_rank`.
- `ALTER DATABASE <db> SET hnsw.ef_search = 100` and `... SET hnsw.iterative_scan = relaxed_order` apply to every new connection (verified with `SHOW`). The pgvector default `hnsw.ef_search` is 40.
- Text-bound params with casts work in a prepared statement: `PREPARE q(text,text,int) AS ... $1::text ... $2::text::vector ... LIMIT $3::int`.
- RRF hybrid (`FULL OUTER JOIN` of a keyword CTE and a vector CTE, `1/(60 + rank)`), filtered + sorted over all keyword matches, and facet counts over the result set all run as single statements on both engines.

## Global Constraints

- Images pinned: `pgvector/pgvector:0.8.7-pg17` (engine `postgres`, host port 15432) and `paradedb/paradedb:0.26.1-pg17` (engine `postgres-paradedb`, host port 15433). Database `irbench`, user `postgres`, password `irbench`.
- Request protocol (spec §3): `{"protocol": "postgres", "sql": "...", "params": [...]}`. SQL uses `$1..$n`; **every placeholder is written `$n::text`, optionally followed by a cast** (`$2::text::vector`, `$3::text::int`). Params are JSON values bound as text by one rule in both clients: null → NULL; bool → `true`/`false`; int → decimal; float → shortest repr; string → itself; list of numbers → pgvector literal `[a,b,...]`.
- Relevance: SQL error or timeout → the task scores 0 and is logged in `errors.jsonl` (same as HTTP); per-request timeout = scenario `request_timeout_s` (10 s) via `statement_timeout`.
- Perf: one Postgres connection per concurrent client; statements prepared once per connection per distinct SQL text; errors and step validity as for HTTP.
- Field priority: vanilla `setweight` classes in scenario order `name` A, `product_class` B, `category` C, `features` D, `description` D; ParadeDB per-field boost `n − i` (`name` 5 … `description` 1).
- Vanilla keyword: `websearch_to_tsquery('<config>', $1::text)` (AND semantics, Postgres as shipped). ParadeDB keyword: disjunction `|||`.
- pgvector: HNSW `vector_cosine_ops` at defaults (`m = 16`, `ef_construction = 64`); `hnsw.ef_search = 100` and `hnsw.iterative_scan = relaxed_order` set per database, each with a `justification:` in the config.
- Sorted mode: keyword base sorts every match; semantic base sorts the top_k nearest; `NULLS LAST`, ties by `id`.
- Faceted mode: counts over the query's own match set (keyword: every match; semantic: every document passing the filter), hits = the top `limit` of that set, returned by one statement as one row `{ids, facets}`.
- Hybrid: RRF, k = 60, each side top_k, in one statement (spec §7 amendment).
- Python ruff-clean (`uv run ruff check .`, `uv run ruff format --check harness engines`); Rust `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` clean; `uv run irbench lint` 0 findings.
- Commits must NOT contain `Co-Authored-By` lines (user rule).

## Review Focus

1. **A query text containing SQL syntax or quotes** (`sofa'; DROP TABLE docs; --`, `"grey" | !velvet`). It is a bound parameter, never interpolated, and `websearch_to_tsquery` / pg_search parse it without a SQL error. Tests: Task 3 `test_query_text_is_a_parameter_not_sql`; Task 3 contract `test_hostile_query_text_is_harmless`.
2. **A Postgres request file given to an HTTP target, or the reverse.** The loadgen refuses to start with a clear message instead of counting every request as an error. Test: Task 5 `rejects_protocol_mismatch`.
3. **A statement that exceeds the 10 s timeout during relevance.** It becomes an `EngineError` (task scored 0, logged), and the connection stays usable for the next task. Test: Task 2 `test_timeout_is_an_engine_error_and_connection_survives` (docker).
4. **A document with null numeric fields or empty text fields.** It loads, never matches a numeric range filter, sorts last, and an empty field contributes nothing to keyword scoring. Tests: Task 3 contract fixture includes such docs; `test_sorted_keyword_orders_and_puts_missing_values_last` already covers ordering.
5. **The synonym file drifts from `synonyms.txt`.** A test fails and prints the expected file content. Test: Task 3 `test_thesaurus_file_matches_scenario_synonyms`.

---

## File Structure

```
pyproject.toml                                  # + psycopg[binary]
harness/irbench/engines/base.py                 # SqlRequest, load_request, protocol classvar, engine_base_url
harness/irbench/engines/pg.py                   # PostgresClient, bind_param
harness/irbench/engines/postgres.py             # PostgresAdapter base (schema, COPY, filters, sort, facets, hybrid)
harness/irbench/runtime/docker.py               # wait_ready (HTTP or Postgres)
harness/irbench/stages/context.py               # default_client dispatch by URL scheme; base_url via engine_base_url
harness/irbench/stages/index.py                 # wait_ready
harness/irbench/stages/relevance.py             # execute() dispatch; load_request
engines/postgres/{adapter.py,README.md,docker/compose.yaml,docker/tsearch/irbench_ecommerce_product_search.syn,configs/ecommerce/product-search.yaml}
engines/postgres-paradedb/{adapter.py,README.md,docker/compose.yaml,configs/ecommerce/product-search.yaml}
loadgen/Cargo.toml                              # + tokio-postgres
loadgen/src/requests.rs                         # Kind::{Http, Postgres}, bind_param
loadgen/src/runner.rs                           # Target::{Http, Postgres}, worker index
loadgen/src/benchmark.rs                        # Target::connect().await, protocol check
loadgen/tests/postgres.rs                       # live test (ignored unless IRBENCH_PG_URL)
harness/tests/test_sql_request.py, test_pg_client.py, test_adapter_postgres.py, test_adapter_paradedb.py
harness/tests/test_contract.py                  # + postgres engines; open client by protocol
docs: methodology/parity.mdx, reference/results.mdx, contributing/new-engine.mdx, index.mdx; parent spec §5 §7
.github/workflows/ci.yml                        # loadgen job: Postgres service + ignored tests
```

---

### Task 1: Spec amendments from the spike

**Files:** Modify `docs/superpowers/specs/2026-10-12-irbench-postgres-design.md`, `docs/superpowers/specs/2026-10-09-irbench-design.md`

**Interfaces:** Produces the authoritative text later tasks cite. No code.

- [ ] **Step 1: Update the Postgres spec** with the spike findings:
  - §1 table: pin `pgvector/pgvector:0.8.7-pg17` and `paradedb/paradedb:0.26.1-pg17`.
  - §3: params are bound as text; every placeholder is `$n::text` with an optional further cast (replace "`$2::vector`" by "`$2::text::vector`"); add the bind rule from Global Constraints.
  - §4.2: pgvector settings are applied with `ALTER DATABASE irbench SET ...` (and `SET` on the indexing session); vanilla synonyms come from a dictionary file committed in `engines/postgres/docker/tsearch/` and mounted read-only into `/usr/share/postgresql/17/tsearch_data/`.
  - §5: ParadeDB synonyms row → "Not supported by pg_search 0.26.1; declared with `synonyms: <reason>`"; remove "if pg_search supports". BM25 index without `key_field`. Default tokenizer does not stem (same as Elasticsearch's standard analyzer).
  - §5 faceted row: counts over the query's own match set (keyword: every match; semantic: every document passing the filter).
- [ ] **Step 2: Amend the parent spec:** §5 step 5 → the request format of the Postgres spec §3 (and `path` instead of `url` for HTTP, rendered under `rendered/<mode>/<base>/<qid>.json` in the run dir); §7 hybrid rule → the amended text in the Postgres spec §7.
- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-10-12-irbench-postgres-design.md docs/superpowers/specs/2026-10-09-irbench-design.md
git commit -m "docs: Postgres spec updated from the spike; parent spec §5 request format and §7 SQL fusion"
```

---

### Task 2: The `postgres` request protocol in the harness

**Files:**
- Modify: `pyproject.toml`, `harness/irbench/engines/base.py`, `harness/irbench/runtime/docker.py`, `harness/irbench/stages/context.py`, `harness/irbench/stages/index.py`, `harness/irbench/stages/relevance.py`, `harness/tests/test_contract.py` (client opening only)
- Create: `harness/irbench/engines/pg.py`
- Test: `harness/tests/test_sql_request.py`, `harness/tests/test_pg_client.py`

**Interfaces:**
- Produces:
  - `irbench.engines.base.SqlRequest(sql: str, params: list[Any])` with `.protocol == "postgres"`, `.to_json()`, `SqlRequest.from_json(dict)`.
  - `irbench.engines.base.load_request(data: dict) -> Request | SqlRequest` (dispatch on `protocol`; unknown → `ValueError`).
  - `EngineAdapter.protocol: ClassVar[str] = "http"`; Postgres adapters set `"postgres"`.
  - `irbench.engines.base.PG_DSN = "postgresql://postgres:irbench@{host}:{port}/irbench"` and `engine_base_url(adapter_cls, host) -> str` (HTTP: `http://host:port`; Postgres: the DSN).
  - `irbench.engines.pg.bind_param(value) -> str | None`; `PostgresClient(dsn)` with `.execute(sql, params=(), timeout_s=None) -> list[dict]`, `.script(sql) -> None` (no params, multiple statements allowed), `.copy(table, columns, rows) -> None`, `.ping() -> bool`, `.close()`, context manager. Errors → `EngineError`.
  - `irbench.runtime.docker.wait_ready(client, health_path, timeout_s)` (Postgres client → poll `ping`; otherwise `wait_healthy`).
  - `irbench.stages.relevance.execute(client, request, timeout) -> Any` (HTTP → `call`; SQL → `client.execute`); `run_request` uses it.
  - `default_client(base_url)` returns `PostgresClient(base_url)` for `postgresql://` URLs.

- [ ] **Step 1: Add psycopg**

Run: `uv add "psycopg[binary]>=3.2"` then `uv run python -c "import psycopg; print(psycopg.__version__, hasattr(psycopg, 'RawCursor'))"`
Expected: a 3.2+ version and `True`.

- [ ] **Step 2: Write the failing unit tests**

`harness/tests/test_sql_request.py`:

```python
import pytest

from irbench.engines.base import Request, SqlRequest, load_request
from irbench.engines.pg import bind_param


def test_sql_request_round_trip():
    r = SqlRequest(sql="SELECT id FROM t WHERE x = $1::text LIMIT $2::text::int", params=["a", 10])
    data = r.to_json()
    assert data == {"protocol": "postgres", "sql": r.sql, "params": ["a", 10]}
    assert load_request(data) == r


def test_http_files_still_load():
    data = {"protocol": "http", "method": "POST", "path": "/s", "headers": {}, "body": {"q": 1}}
    assert isinstance(load_request(data), Request)


def test_unknown_protocol_is_an_error():
    with pytest.raises(ValueError, match="protocol"):
        load_request({"protocol": "grpc"})


@pytest.mark.parametrize("value,expected", [
    (None, None), (True, "true"), (False, "false"), (10, "10"), (0.5, "0.5"), (1e-7, "1e-07"),
    ("sofa", "sofa"), ([0.25, -1, 2.0], "[0.25,-1.0,2.0]"),
])
def test_bind_param(value, expected):
    assert bind_param(value) == expected


@pytest.mark.parametrize("value", [{"a": 1}, ["a", "b"], [True, 1.0]])
def test_bind_param_rejects_other_types(value):
    with pytest.raises(TypeError):
        bind_param(value)
```

`harness/tests/test_pg_client.py` (Docker; starts a throwaway pgvector container):

```python
import subprocess
import time

import pytest

from irbench.engines.base import EngineError
from irbench.engines.pg import PostgresClient

pytestmark = pytest.mark.docker

DSN = "postgresql://postgres:irbench@localhost:15499/irbench"


@pytest.fixture(scope="module")
def pg():
    name = "irbench-test-pg-client"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run(
        ["docker", "run", "-d", "--name", name, "-e", "POSTGRES_PASSWORD=irbench",
         "-e", "POSTGRES_DB=irbench", "-p", "15499:5432", "pgvector/pgvector:0.8.7-pg17"],
        check=True, capture_output=True,
    )
    client = PostgresClient(DSN)
    deadline = time.monotonic() + 60
    while not client.ping():
        assert time.monotonic() < deadline, "postgres did not start"
        time.sleep(1)
    yield client
    client.close()
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def test_execute_binds_text_params(pg):
    pg.script("CREATE EXTENSION IF NOT EXISTS vector; DROP TABLE IF EXISTS t; "
              "CREATE TABLE t (id text, v vector(2), n double precision)")
    pg.copy("t", ["id", "v", "n"], [["a", "[1,0]", 1.5], ["b", "[0,1]", None]])
    rows = pg.execute(
        "SELECT id FROM t WHERE n IS NULL OR n >= $1::text::float8 "
        "ORDER BY v <=> $2::text::vector LIMIT $3::text::int",
        [1.0, [0.0, 1.0], 10],
    )
    assert [r["id"] for r in rows] == ["b", "a"]


def test_sql_error_is_an_engine_error(pg):
    with pytest.raises(EngineError):
        pg.execute("SELECT * FROM no_such_table")
    assert pg.execute("SELECT 1 AS x") == [{"x": 1}]


def test_timeout_is_an_engine_error_and_connection_survives(pg):
    with pytest.raises(EngineError, match="timeout|cancel"):
        pg.execute("SELECT pg_sleep(2)", timeout_s=0.2)
    assert pg.execute("SELECT 1 AS x") == [{"x": 1}]
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest harness/tests/test_sql_request.py -v`
Expected: `ImportError: cannot import name 'SqlRequest'`.

- [ ] **Step 4: Implement**

In `harness/irbench/engines/base.py`, after `Request`, add:

```python
@dataclass(frozen=True)
class SqlRequest:
    """One SQL statement, exactly as written to a rendered-request file (protocol `postgres`).

    Placeholders are `$n::text` (optionally cast further, e.g. `$2::text::vector`); every
    param is bound as text by `irbench.engines.pg.bind_param`, the same rule as the loadgen.
    """

    sql: str
    params: list[Any] = field(default_factory=list)
    protocol: Literal["postgres"] = "postgres"

    def to_json(self) -> dict[str, Any]:
        return {"protocol": self.protocol, "sql": self.sql, "params": list(self.params)}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SqlRequest":
        return cls(sql=data["sql"], params=list(data.get("params") or []))


def load_request(data: dict[str, Any]) -> "Request | SqlRequest":
    protocol = data.get("protocol", "http")
    if protocol == "http":
        return Request.from_json(data)
    if protocol == "postgres":
        return SqlRequest.from_json(data)
    raise ValueError(f"unknown request protocol {protocol!r}")


PG_DSN = "postgresql://postgres:irbench@{host}:{port}/irbench"


def engine_base_url(adapter_cls: "type[EngineAdapter]", host: str) -> str:
    if adapter_cls.protocol == "postgres":
        return PG_DSN.format(host=host, port=adapter_cls.host_port)
    return f"http://{host}:{adapter_cls.host_port}"
```

Add `protocol: ClassVar[str] = "http"` to `EngineAdapter` (next to `name`), and change `render`'s return annotation to `"Request | SqlRequest"`.

Create `harness/irbench/engines/pg.py`:

```python
"""Postgres client for the `postgres` request protocol (spec 2026-10-12 §3, §4.1)."""

from collections.abc import Iterable, Sequence
from typing import Any

import psycopg
from psycopg.rows import dict_row

from irbench.engines.base import EngineError


def bind_param(value: Any) -> str | None:
    """Every param travels as text (or NULL); the SQL casts it. Mirrors loadgen bind_param."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(
        isinstance(x, int | float) and not isinstance(x, bool) for x in value
    ):
        return "[" + ",".join(repr(float(x)) for x in value) + "]"
    raise TypeError(f"unsupported SQL parameter {value!r}")


class PostgresClient:
    """One autocommit connection; the Postgres counterpart of httpx.Client for adapters."""

    def __init__(self, dsn: str):
        self.dsn = dsn
        self._conn: psycopg.Connection | None = None
        self._timeout_ms: int | None = None

    def _connection(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(
                self.dsn, autocommit=True, row_factory=dict_row,
                cursor_factory=psycopg.RawCursor, connect_timeout=10,
            )
            self._timeout_ms = None
        return self._conn

    def _set_timeout(self, conn: psycopg.Connection, timeout_s: float | None) -> None:
        ms = 0 if timeout_s is None else max(1, int(timeout_s * 1000))
        if ms != self._timeout_ms:
            conn.execute(f"SET statement_timeout = {ms}")
            self._timeout_ms = ms

    def execute(
        self, sql: str, params: Sequence[Any] = (), timeout_s: float | None = None
    ) -> list[dict[str, Any]]:
        try:
            conn = self._connection()
            self._set_timeout(conn, timeout_s)
            with conn.cursor() as cur:
                cur.execute(sql, [bind_param(p) for p in params])
                return list(cur.fetchall()) if cur.description else []
        except psycopg.Error as e:
            raise EngineError(f"postgres: {str(e).strip()[:500]}") from e

    def script(self, sql: str) -> None:
        """Several statements, no params (DDL). No timeout: index builds can take minutes."""
        try:
            conn = self._connection()
            self._set_timeout(conn, None)
            with psycopg.ClientCursor(conn) as cur:
                cur.execute(sql)
        except psycopg.Error as e:
            raise EngineError(f"postgres: {str(e).strip()[:500]}") from e

    def copy(self, table: str, columns: list[str], rows: Iterable[Sequence[Any]]) -> None:
        cols = ", ".join(columns)
        try:
            conn = self._connection()
            self._set_timeout(conn, None)
            with conn.cursor() as cur, cur.copy(f"COPY {table} ({cols}) FROM STDIN") as cp:
                for row in rows:
                    cp.write_row(row)
        except psycopg.Error as e:
            raise EngineError(f"postgres: {str(e).strip()[:500]}") from e

    def ping(self) -> bool:
        try:
            self.execute("SELECT 1")
            return True
        except EngineError:
            self.close()
            return False

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "PostgresClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
```

`ClientCursor` is used for `script` because multiple statements need client-side binding. `copy` writes `None` as NULL and strings verbatim; a vector column accepts its `[...]` text. If `psycopg.RawCursor` is missing in the installed version, stop and report BLOCKED (Step 1 checks it).

In `harness/irbench/runtime/docker.py`, add:

```python
def wait_ready(client: Any, health_path: str, timeout_s: float, interval_s: float = 1.0) -> None:
    """HTTP engines: GET health_path until 200. Postgres engines: SELECT 1 until it answers."""
    from irbench.engines.pg import PostgresClient

    if isinstance(client, PostgresClient):
        poll(client.ping, timeout_s, f"{client.dsn.rsplit('@', 1)[-1]} to accept queries", interval_s)
    else:
        wait_healthy(client, health_path, timeout_s, interval_s)
```

(add `from typing import Any`). In `harness/irbench/stages/index.py`, replace `wait_healthy(...)` with `wait_ready(client, adapter.health_path, HEALTH_TIMEOUT_S)`.

In `harness/irbench/stages/context.py`:

```python
ClientFactory = Callable[[str], Any]


def default_client(base_url: str) -> Any:
    if base_url.startswith("postgresql://"):
        return PostgresClient(base_url)
    return httpx.Client(base_url=base_url, timeout=60.0)
```

and `base_url` → `return engine_base_url(self.adapter_cls, self.spec.host)` (imports: `PostgresClient` from `irbench.engines.pg`, `engine_base_url` from `irbench.engines.base`, `Any`).

In `harness/irbench/stages/relevance.py`:

```python
def execute(client: Any, request: Request | SqlRequest, timeout: float) -> Any:
    if isinstance(request, SqlRequest):
        return client.execute(request.sql, request.params, timeout_s=timeout)
    return call(client, request, timeout)


def run_request(client, adapter, task, request, timeout) -> SearchResult:
    return adapter.parse(task, execute(client, request, timeout))


def _load(path: Path) -> Request | SqlRequest:
    return load_request(json.loads(path.read_text(encoding="utf-8")))
```

`REQUEST_ERRORS` already contains `EngineError`; leave it.

In `harness/tests/test_contract.py`, replace the client construction and health wait in the `engine` fixture with:

```python
    client = default_client(engine_base_url(cls, "localhost"))
    try:
        wait_ready(client, cls.health_path, 180)
```

(imports: `default_client` from `irbench.stages.context`, `engine_base_url` from `irbench.engines.base`, `wait_ready` from `irbench.runtime.docker`; keep `client.close()` in the teardown — both client types have `close`). Change the `Engine.client` annotation to `Any`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest harness/tests/test_sql_request.py -v`, then `uv run pytest -m docker harness/tests/test_pg_client.py -v` (OrbStack running), then `uv run pytest`.
Expected: all pass; existing suites unchanged.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix . && uv run ruff format harness engines
git add pyproject.toml uv.lock harness
git commit -m "feat: postgres request protocol and psycopg client in the harness"
```

---

### Task 3: Shared Postgres adapter base and the vanilla `postgres` engine

**Files:**
- Create: `harness/irbench/engines/postgres.py`, `engines/postgres/adapter.py`, `engines/postgres/README.md`, `engines/postgres/docker/compose.yaml`, `engines/postgres/docker/tsearch/irbench_ecommerce_product_search.syn`, `engines/postgres/configs/ecommerce/product-search.yaml`
- Test: `harness/tests/test_adapter_postgres.py`; modify `harness/tests/test_contract.py` (`ENGINES`, hostile-text test)

**Interfaces:**
- Consumes: Task 2 (`SqlRequest`, `PostgresClient`, `protocol` classvar).
- Produces: `irbench.engines.postgres.PostgresAdapter` with hooks `text_search_ddl() -> str` (extra column/config DDL run before `CREATE TABLE`... see code), `text_columns_sql() -> list[str]`, `text_index_ddl() -> str`, `keyword_match(q: Sql, text: str) -> tuple[str, str]` (predicate, score expression, higher = better); `Sql` param builder; `table` name = `ctx.index`. Task 4's ParadeDB adapter subclasses it.

- [ ] **Step 1: Write the failing unit tests**

`harness/tests/test_adapter_postgres.py`:

```python
import json
from pathlib import Path

import pytest

from adapter_helpers import SCENARIO, make_task
from conftest import REPO_ROOT
from irbench.engines.base import IndexContext, SqlRequest, load_adapter_class
from irbench.engines.config import load_engine_config
from irbench.models import Filter, Sort
from irbench.scenario import load_scenario
from irbench.terms import load_term_groups

ENGINE = "postgres"


@pytest.fixture
def adapter():
    scenario = load_scenario(REPO_ROOT, SCENARIO)
    cls = load_adapter_class(REPO_ROOT, ENGINE)
    config = load_engine_config(REPO_ROOT, ENGINE, SCENARIO).track("default")
    return cls(IndexContext(scenario=scenario, config=config, index="irbench_t", dim=3,
                            synonyms=load_term_groups(scenario.synonyms_path), n_docs=1000))


def placeholders_are_text(r: SqlRequest) -> bool:
    import re
    return all(m.group(1) for m in re.finditer(r"\$\d+(::text)?", r.sql))


def test_keyword_request(adapter):
    r = adapter.render(make_task("keyword", text="grey sofa"), None, 100)
    assert isinstance(r, SqlRequest) and placeholders_are_text(r)
    assert "websearch_to_tsquery('irbench_ecommerce_product_search', $1::text)" in r.sql
    assert "ts_rank(" in r.sql and "LIMIT $2::text::int" in r.sql
    assert r.params == ["grey sofa", 100]


def test_query_text_is_a_parameter_not_sql(adapter):
    text = "sofa'; DROP TABLE docs; --"
    r = adapter.render(make_task("keyword", text=text), None, 100)
    assert text not in r.sql and r.params[0] == text


def test_semantic_request(adapter):
    r = adapter.render(make_task("semantic"), [0.1, 0.2, 0.3], 100)
    assert "embedding <=> $1::text::vector" in r.sql
    assert r.params == [[0.1, 0.2, 0.3], 100]


def test_hybrid_is_rrf_in_one_statement(adapter):
    r = adapter.render(make_task("hybrid", text="sofa"), [0.1, 0.2, 0.3], 100)
    assert "FULL OUTER JOIN" in r.sql and "60 +" in r.sql
    assert r.params[:2] == ["sofa", [0.1, 0.2, 0.3]]


def test_filters(adapter):
    eq = adapter.render(make_task("filtered", text="sofa",
                                  filter=Filter(field="product_class", op="eq", value="Sofas")), None, 100)
    assert "product_class__terms @> ARRAY[$" in eq.sql and "Sofas" in eq.params
    rng = adapter.render(make_task("filtered", text="sofa",
                                   filter=Filter(field="average_rating", op="gte", value=4.0)), None, 100)
    assert "average_rating >= $" in rng.sql and "::text::float8" in rng.sql and 4.0 in rng.params


def test_sorted_keyword_sorts_every_match(adapter):
    r = adapter.render(make_task("sorted", text="sofa",
                                 sort=Sort(field="average_rating", direction="desc")), None, 100)
    assert "ORDER BY average_rating DESC NULLS LAST, id" in r.sql
    assert "embedding" not in r.sql


def test_sorted_semantic_sorts_top_k(adapter):
    r = adapter.render(make_task("sorted", base="semantic",
                                 sort=Sort(field="review_count", direction="asc")), [0.1, 0.2, 0.3], 100)
    assert "ORDER BY review_count ASC NULLS LAST, id" in r.sql and "embedding <=>" in r.sql


def test_faceted_returns_one_row_with_ids_and_facets(adapter):
    r = adapter.render(make_task("faceted", text="sofa", facets=["product_class"]), None, 100)
    assert "json_agg" in r.sql and "unnest(product_class__terms)" in r.sql
    found = adapter.parse(make_task("faceted", text="sofa", facets=["product_class"]),
                          [{"ids": ["2", "1"], "facets": {"product_class": {"Sofas": 2}}}])
    assert found.ids == ["2", "1"] and found.facets == {"product_class": {"Sofas": 2}}


def test_facet_check_fetch_is_a_plain_query(adapter):
    task = make_task("faceted", text="sofa", facets=["product_class"])
    r = adapter.render(task, None, 1000, facets=False)
    assert "json_agg" not in r.sql and r.params[-1] == 1000
    assert adapter.parse(task, [{"id": "1"}, {"id": "2"}]).ids == ["1", "2"]


def test_synonym_file_matches_scenario_synonyms(adapter):
    path = REPO_ROOT / "engines/postgres/docker/tsearch/irbench_ecommerce_product_search.syn"
    expected = adapter.synonym_file()
    assert path.read_text(encoding="utf-8") == expected, (
        f"{path} is out of date; replace its content with:\n{expected}"
    )


def test_synonym_file_skips_multiword_groups(adapter):
    text = adapter.synonym_file()
    assert "gray grey" in text.splitlines()
    assert not any(" seater" in line for line in text.splitlines())
```

Check `adapter_helpers.make_task`'s signature before running; if it differs (e.g. filter/sort keyword names), adapt the calls to it, not the helper.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest harness/tests/test_adapter_postgres.py -v`
Expected: collection error — no `engines/postgres/adapter.py`.

- [ ] **Step 3: Implement the shared base**

`harness/irbench/engines/postgres.py`:

```python
"""Shared Postgres adapter (spec 2026-10-12 §4-§5). Subclasses provide keyword search only."""

import re
from abc import abstractmethod
from typing import Any

import numpy as np

from irbench.engines.base import EngineAdapter, SqlRequest, require_vector
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
            if not re.fullmatch(r"[a-z_]+\.[a-z_]+", name) or not re.fullmatch(r"[A-Za-z0-9_.]+", str(value)):
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
        stmts = [self.text_index_ddl(), f"CREATE INDEX ON {t} USING hnsw (embedding vector_cosine_ops)"]
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

    def render(self, task: QueryTask, vector: list[float] | None, limit: int, facets: bool = True) -> SqlRequest:
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
            sql = f"SELECT id FROM ({self._ranked(q, task, vector, limit)}) m ORDER BY score DESC, id"
        params = q.params
        extra = self.search_params(task)
        if extra:
            raise ValueError(f"{self.name}: search.{task.base} settings are not supported: {sorted(extra)}")
        return SqlRequest(sql=sql, params=params)

    def parse(self, task: QueryTask, payload: Any) -> "SearchResult":
        from irbench.engines.base import SearchResult

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
```

Notes for the implementer:
- `_ranked` is called once per render and adds params in a fixed order (query text, vector, filter, limits); the caller adds its own trailing params (the sorted mode's final limit) after it returns. Where one value is used twice (the query vector in `ORDER BY` and `score`, the filter in both hybrid CTEs), the same placeholder string is reused — never call `q.p` twice for one value.
- The `keyword` base `ORDER BY score` uses the score alias, which Postgres allows in `ORDER BY`.
- Unit tests assert substrings and param order (`test_keyword_request`: text `$1`, limit `$2`; semantic: vector, limit; hybrid: text, vector, ...). If a test disagrees with the code, fix the code so the documented order holds; don't loosen the test.
- The faceted `m` CTE keeps every match (keyword) or every filtered document (semantic, ordered by distance). For the semantic base that is a scan of the table; that is the spec's definition ("every document passing the filter"), and the perf numbers will show its cost.

- [ ] **Step 4: Implement the vanilla engine**

`engines/postgres/adapter.py`:

```python
"""Postgres 17 + pgvector, built-in full-text search. Parity notes: engines/postgres/README.md."""

import re

from irbench.engines.postgres import PostgresAdapter, Sql, ident

WEIGHTS = "ABCD"


class Adapter(PostgresAdapter):
    name = "postgres"
    host_port = 15432

    @property
    def ts_config(self) -> str:
        return ident("irbench_" + re.sub(r"[^a-z0-9]+", "_", self.ctx.scenario.id.lower()))

    def synonym_file(self) -> str:
        """Single-word groups only: each term maps to the group's first term."""
        lines = []
        for group in self.ctx.synonyms:
            words = [w.lower() for w in group]
            if any(" " in w for w in words):
                continue
            lines += [f"{w} {words[0]}" for w in words[1:]]
        return "".join(line + "\n" for line in lines)

    def setup_ddl(self) -> str:
        cfg = self.ts_config
        syn = "" if self.ctx.config.synonyms else (
            f"CREATE TEXT SEARCH DICTIONARY {cfg}_syn (TEMPLATE = synonym, SYNONYMS = {cfg});\n"
            f"ALTER TEXT SEARCH CONFIGURATION {cfg} ALTER MAPPING FOR asciiword, word "
            f"WITH {cfg}_syn, english_stem;"
        )
        return (
            "CREATE EXTENSION IF NOT EXISTS vector;\n"
            f"DROP TABLE IF EXISTS {self.table};\n"
            f"DROP TEXT SEARCH CONFIGURATION IF EXISTS {cfg};\n"
            f"DROP TEXT SEARCH DICTIONARY IF EXISTS {cfg}_syn;\n"
            f"CREATE TEXT SEARCH CONFIGURATION {cfg} (COPY = english);\n{syn}"
        )

    def extra_columns(self) -> list[tuple[str, str]]:
        cfg = self.ts_config
        parts = [
            f"setweight(to_tsvector('{cfg}', coalesce({f}, '')), '{WEIGHTS[min(i, 3)]}')"
            for i, f in enumerate(self.searchable)
        ]
        return [("tsv", f"tsvector GENERATED ALWAYS AS ({' || '.join(parts)}) STORED")]

    def text_index_ddl(self) -> str:
        return f"CREATE INDEX ON {self.table} USING gin (tsv)"

    def keyword_match(self, q: Sql, text: str) -> tuple[str, str]:
        query = f"websearch_to_tsquery('{self.ts_config}', {q.p(text)})"
        return f"tsv @@ {query}", f"ts_rank(tsv, {query})"
```

Because `keyword_match` is called once per statement but its placeholder string appears twice (predicate and score), it adds one param — consistent with "one value, one placeholder". For hybrid, `_ranked` calls `keyword_match` once and uses `pred`/`score` together.

`engines/postgres/docker/compose.yaml`:

```yaml
# Pinned engine image. The resolved digest is recorded in every run manifest.
services:
  engine:
    image: pgvector/pgvector:0.8.7-pg17
    container_name: irbench-postgres
    ports:
      - "15432:5432"
    environment:
      POSTGRES_PASSWORD: irbench   # local benchmark only
      POSTGRES_DB: irbench
    volumes:
      - data:/var/lib/postgresql/data
      # synonyms.txt as a Postgres synonym dictionary (generated; see README)
      - ./tsearch/irbench_ecommerce_product_search.syn:/usr/share/postgresql/17/tsearch_data/irbench_ecommerce_product_search.syn:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d irbench"]
      interval: 2s
      timeout: 3s
      retries: 60

volumes:
  data:
```

`engines/postgres/docker/tsearch/irbench_ecommerce_product_search.syn`: write the exact output of `adapter.synonym_file()` (run the failing `test_synonym_file_matches_scenario_synonyms` once; it prints the expected content).

`engines/postgres/configs/ecommerce/product-search.yaml`:

```yaml
# Postgres 17 + pgvector on ecommerce/product-search.
#
# default: Postgres as shipped plus the minimum to load the scenario: a weighted
#   tsvector over the searchable fields (setweight A-D in scenario order), the
#   english text search configuration with synonyms.txt as a synonym dictionary,
#   pgvector HNSW at its defaults, B-tree/GIN indexes on filter and sort fields.
# tuned: changes within the parity rules (docs/methodology/parity); add a `tuned:`
#   block here, deep-merged over `default`.
#
# Keys: index.settings (database-level GUCs), unsupported (key -> reason),
# justification (key -> why a non-default setting is set).
default:
  index:
    settings:
      hnsw.ef_search: 100
      hnsw.iterative_scan: relaxed_order
  search: {}
  unsupported: {}
  justification:
    settings.hnsw.ef_search: >-
      pgvector's HNSW scan returns at most ef_search rows (default 40), so the
      default could never return the scenario's top_k = 100. 100 is the
      smallest value that lets a vector query return top_k results.
    settings.hnsw.iterative_scan: >-
      Without iterative scans (off by default in pgvector 0.8), a filtered
      vector query keeps only the ef_search nearest rows before filtering and
      can return fewer than top_k results. relaxed_order is pgvector's
      documented setting for filtered queries.
```

The lint flattens `index.settings` into paths like `index.settings.hnsw.ef_search` and accepts a justification key that names the path without its leading `index.` (here `settings.hnsw.ef_search`), so the keys above satisfy `irbench lint`.

`engines/postgres/README.md` — same structure as `engines/qdrant/README.md`: image and port; "What the default track sends" table (searchable fields → weighted tsvector with the A/B/C/D/D mapping; synonyms → synonym dictionary, list the dropped multi-word groups `three seater, 3 seater` and `two drawer, 2 drawer`; filter fields → `<field>__terms text[]` with GIN, numeric → B-tree; vectors → HNSW defaults; the two pgvector settings and why); "Modes" (keyword AND semantics via `websearch_to_tsquery`, no typo/prefix tolerance; semantic; hybrid RRF k=60 with the statement shape; filtered; sorted rules; faceted over the match set); "Known limits" (4 weight classes for 5 fields; AND matching; faceted on the semantic base scans every filtered document); "Changing this configuration".

- [ ] **Step 5: Run the unit tests**

Run: `uv run pytest harness/tests/test_adapter_postgres.py -v`, `uv run irbench lint`, `uv run pytest`
Expected: all pass, lint 0 findings.

- [ ] **Step 6: Contract tests**

In `harness/tests/test_contract.py`: add `"postgres"` to `ENGINES`, and add:

```python
def test_hostile_query_text_is_harmless(engine):
    engine.search(make_task("keyword", text="sofa'; DROP TABLE x; -- \"grey\" | !velvet"))
    assert engine.search(make_task("keyword", text="sofa")).ids  # still answers afterwards
```

(for HTTP engines it is a plain search too; it must not raise for any engine — if an HTTP engine rejects that text with a 4xx, report it rather than special-casing the test.) Then make sure the contract fixture's documents include at least one doc with null numeric fields and one with an empty text field (check `contract_fixture.py`; add them only if missing, without changing the existing known answers).

Run: `uv run pytest -m docker -k postgres -v`
Expected: every claimed mode passes; skips only with declared reasons. Then `uv run pytest -m docker -v` (all engines) — no regressions.

- [ ] **Step 7: Commit**

```bash
uv run ruff check --fix . && uv run ruff format harness engines
git add harness engines/postgres
git commit -m "feat: Postgres engine (built-in full-text search + pgvector)"
```

---

### Task 4: The `postgres-paradedb` engine

**Files:**
- Create: `engines/postgres-paradedb/adapter.py`, `README.md`, `docker/compose.yaml`, `configs/ecommerce/product-search.yaml`
- Test: `harness/tests/test_adapter_paradedb.py`; modify `harness/tests/test_contract.py` (`ENGINES`)

**Interfaces:** Consumes `PostgresAdapter`, `Sql`, `ident` (Task 3).

- [ ] **Step 1: Verify the boost cast on a bound parameter**

Start the pinned image (`docker run -d --rm --name pdb-check -e POSTGRES_PASSWORD=irbench -e POSTGRES_DB=irbench -p 25433:5432 paradedb/paradedb:0.26.1-pg17`), create a 3-row table with a bm25 index, and run with psql:

```sql
PREPARE q(text) AS SELECT id, pdb.score(id) FROM t WHERE name ||| $1::text::pdb.boost(5) OR description ||| $1::text::pdb.boost(1) ORDER BY pdb.score(id) DESC;
EXECUTE q('bed');
```

Expected: rows ranked with the name match first. If the cast chain on a parameter is rejected, try `($1::text)::pdb.boost(5)`; use whichever form works in `keyword_match` and note it in the README. If neither works, stop and report NEEDS_CONTEXT with the exact errors. Remove the container afterwards.

- [ ] **Step 2: Write the failing unit tests**

`harness/tests/test_adapter_paradedb.py`:

```python
import pytest

from adapter_helpers import SCENARIO, make_task
from conftest import REPO_ROOT
from irbench.engines.base import IndexContext, SqlRequest, load_adapter_class
from irbench.engines.config import load_engine_config
from irbench.scenario import load_scenario

ENGINE = "postgres-paradedb"


@pytest.fixture
def adapter():
    scenario = load_scenario(REPO_ROOT, SCENARIO)
    cls = load_adapter_class(REPO_ROOT, ENGINE)
    config = load_engine_config(REPO_ROOT, ENGINE, SCENARIO).track("default")
    return cls(IndexContext(scenario=scenario, config=config, index="irbench_t", dim=3,
                            synonyms=[], n_docs=1000))


def test_keyword_is_bm25_with_field_boosts_in_scenario_order(adapter):
    r = adapter.render(make_task("keyword", text="grey sofa"), None, 100)
    assert isinstance(r, SqlRequest)
    for weight, field in zip([5, 4, 3, 2, 1], ["name", "product_class", "category", "features", "description"]):
        assert f"{field} ||| $1::text::pdb.boost({weight})" in r.sql
    assert "pdb.score(id)" in r.sql and r.params[0] == "grey sofa"


def test_bm25_index_ddl(adapter):
    ddl = adapter.text_index_ddl()
    assert "USING bm25 (id, name, product_class, category, features, description)" in ddl
    assert "key_field" not in ddl


def test_synonyms_declared_unsupported(adapter):
    assert adapter.ctx.config.synonyms
```

(If Step 1 required `($1::text)::pdb.boost(n)`, assert that form instead.)

- [ ] **Step 3: Run them to verify they fail**, then **implement**:

`engines/postgres-paradedb/adapter.py`:

```python
"""ParadeDB: pg_search (BM25) + pgvector. Parity notes: engines/postgres-paradedb/README.md."""

from irbench.engines.postgres import PostgresAdapter, Sql


class Adapter(PostgresAdapter):
    name = "postgres-paradedb"
    host_port = 15433

    def setup_ddl(self) -> str:
        return "CREATE EXTENSION IF NOT EXISTS pg_search;\nCREATE EXTENSION IF NOT EXISTS vector;"

    def text_index_ddl(self) -> str:
        return f"CREATE INDEX ON {self.table} USING bm25 (id, {', '.join(self.searchable)})"

    def keyword_match(self, q: Sql, text: str) -> tuple[str, str]:
        ph = q.p(text)
        n = len(self.searchable)
        pred = " OR ".join(
            f"{f} ||| {ph}::pdb.boost({n - i})" for i, f in enumerate(self.searchable)
        )
        return f"({pred})", "pdb.score(id)"
```

`engines/postgres-paradedb/docker/compose.yaml`: as the vanilla one with `image: paradedb/paradedb:0.26.1-pg17`, `container_name: irbench-postgres-paradedb`, port `"15433:5432"`, no tsearch mount.

`engines/postgres-paradedb/configs/ecommerce/product-search.yaml`: header comment (default = pg_search as shipped: BM25 over the searchable fields with `n − i` boosts, disjunction, default tokenizer; pgvector HNSW defaults); `index.settings` with the same two pgvector settings and justifications as the vanilla config; `synonyms:` reason "pg_search 0.26.1 has no synonym feature; expanding queries in the harness would break the parity rules."; `unsupported: {}`.

`engines/postgres-paradedb/README.md`: same structure as the vanilla README: BM25 per-field boosts (same precedent as Elasticsearch and Qdrant, scores summed by `OR` of boosted clauses), disjunction, no stemming by default (like Elasticsearch's standard analyzer), fuzzy matching opt-in (off), synonyms not loaded, pgvector settings, hybrid RRF statement, known limits.

`load_adapter_class` imports `engines/<engine>/adapter.py` by path. Check it handles a folder name with a dash (`postgres-paradedb`); if the generated module name is invalid, make the loader derive a safe module name (e.g. replace `-` with `_`) and add a unit test for it in `harness/tests/test_engine_base.py`.

- [ ] **Step 4: Run** `uv run pytest harness/tests/test_adapter_paradedb.py -v`, add `"postgres-paradedb"` to `ENGINES` in `test_contract.py`, run `uv run pytest -m docker -k paradedb -v`, then `uv run pytest -m docker -v`, `uv run irbench lint`, `uv run pytest`.
Expected: all pass; ParadeDB skips only the synonym test with its declared reason.

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix . && uv run ruff format harness engines
git add harness engines/postgres-paradedb
git commit -m "feat: ParadeDB engine (pg_search BM25 + pgvector)"
```

---

### Task 5: Postgres in the loadgen

**Files:**
- Modify: `loadgen/Cargo.toml`, `loadgen/src/requests.rs`, `loadgen/src/runner.rs`, `loadgen/src/benchmark.rs`, `loadgen/README.md`
- Create: `loadgen/tests/postgres.rs`

**Interfaces:**
- Produces: `requests::Kind::{Http { method, path, headers, body }, Postgres { sql, params: Vec<Option<String>> }}` on `PreparedRequest.kind`; `requests::bind_param(&serde_json::Value) -> Result<Option<String>>`; `runner::Target::connect(url, timeout, max_connections).await -> Result<Target>` (`http://` → HTTP, `postgresql://` → Postgres); `Target::send(&self, worker: usize, req) -> bool`; `Target::protocol() -> &'static str`.

- [ ] **Step 1: Dependency**

Add to `[dependencies]`: `tokio-postgres = "0.7"` and `futures-util = "0.3"` if needed by the connection task; run `cargo build` in `loadgen/`.

- [ ] **Step 2: Failing tests** in `requests.rs`'s test module:

```rust
#[test]
fn loads_postgres_files_and_binds_params_as_text() {
    let dir = tmp("pg");
    write(&dir, "keyword/keyword/0.json",
        r#"{"protocol":"postgres","sql":"SELECT id FROM t WHERE x = $1::text LIMIT $3::text::int","params":["sofa",[0.5,-1],100]}"#);
    let reqs = load_requests(&dir).unwrap();
    match &reqs[0].kind {
        Kind::Postgres { sql, params } => {
            assert!(sql.starts_with("SELECT id"));
            assert_eq!(params, &vec![Some("sofa".into()), Some("[0.5,-1.0]".into()), Some("100".into())]);
        }
        _ => panic!("expected a postgres request"),
    }
}

#[test]
fn bind_param_rules() {
    use serde_json::json;
    assert_eq!(bind_param(&json!(null)).unwrap(), None);
    assert_eq!(bind_param(&json!(true)).unwrap().as_deref(), Some("true"));
    assert_eq!(bind_param(&json!(10)).unwrap().as_deref(), Some("10"));
    assert_eq!(bind_param(&json!(0.5)).unwrap().as_deref(), Some("0.5"));
    assert!(bind_param(&json!({"a": 1})).is_err());
}
```

Replace the existing `rejects_other_protocols_and_empty_dirs` assertion for `"postgres"` with an unknown protocol (`"grpc"`). In `tests/benchmark.rs` (or the closest existing benchmark test), add `rejects_protocol_mismatch`: postgres request files with an `http://` base URL → `run_benchmark` returns an error whose message names both protocols.

- [ ] **Step 3: Implement**
  - `requests.rs`: `RenderedFile` gets optional `sql: Option<String>` and `params: Option<Vec<Value>>`, with `method`/`path` optional; dispatch on `protocol` (`http` → today's code producing `Kind::Http`; `postgres` → `Kind::Postgres { sql, params: params.iter().map(bind_param).collect()? }`; other → error). `bind_param`: null → None; bool → "true"/"false"; i64/u64 → decimal; f64 → `format!("{}", f)` (Rust's shortest round-trip form; `0.5` → `0.5`); string → itself; array of numbers → `"[" + join(format!("{:?}", f as f64)) + "]"` so `-1` becomes `-1.0` like Python's `repr(float(-1))`; anything else → error naming the value.
  - `runner.rs`: `pub enum Target { Http { client, base_url }, Postgres { conns: Vec<tokio::sync::Mutex<PgConn>> } }` with `struct PgConn { client: tokio_postgres::Client, stmts: HashMap<String, tokio_postgres::Statement> }`. `Target::connect(url, timeout, max_connections)`: `postgresql://` → parse with `tokio_postgres::Config::from_str`, set `.options(&format!("-c statement_timeout={}", timeout.as_millis()))`, open `max_connections` connections with `NoTls`, spawn each connection future with `tokio::spawn`; otherwise build today's reqwest client. `send(&self, worker, req)`: HTTP as today (ignore `worker`); Postgres: lock `conns[worker % conns.len()]`, prepare the SQL once per connection (cache in `stmts`), `client.query(&stmt, &params_refs)` where `params_refs: Vec<&(dyn ToSql + Sync)>` built from the `Option<String>` values; `Ok` → true, `Err` → false. A request whose kind doesn't match the target → false (the benchmark checks this up front). `protocol()` returns `"http"` or `"postgres"`.
  - `run()`: spawn workers with their index (`for w in 0..concurrency.max(1)`) and call `target.send(w, &requests[i % total])`.
  - `benchmark.rs`: `let target = Arc::new(Target::connect(&cfg.base_url, cfg.timeout, max_conn).await?);` and, before the warm-up, bail if any request's kind protocol differs from `target.protocol()`: `"rendered requests use protocol {req} but --base-url is a {target} target"`.
- [ ] **Step 4: Live test** `loadgen/tests/postgres.rs`, ignored unless `IRBENCH_PG_URL` is set:

```rust
//! Live Postgres test: IRBENCH_PG_URL=postgresql://postgres:irbench@localhost:15432/irbench cargo test -- --ignored
use std::time::Duration;

#[tokio::test]
#[ignore]
async fn postgres_target_runs_prepared_statements() {
    let Ok(url) = std::env::var("IRBENCH_PG_URL") else { return };
    let target = irbench_loadgen::runner::Target::connect(&url, Duration::from_secs(5), 2).await.unwrap();
    let ok = irbench_loadgen::requests::PreparedRequest::postgres_for_test(
        "x/x", "SELECT $1::text::int + 1 AS n", vec![Some("41".into())]);
    let bad = irbench_loadgen::requests::PreparedRequest::postgres_for_test(
        "x/x", "SELECT * FROM no_such_table", vec![]);
    assert!(target.send(0, &ok).await);
    assert!(target.send(1, &ok).await);
    assert!(!target.send(0, &bad).await);
    assert!(target.send(0, &ok).await, "connection still usable after an error");
}
```

Add `PreparedRequest::postgres_for_test(group, sql, params)` as a small `pub fn` constructor (doc-commented as a test helper). Adjust the crate paths to the actual library name in `Cargo.toml`.

- [ ] **Step 5: Run** `cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test` in `loadgen/`, then start `engines/postgres` (`docker compose -f engines/postgres/docker/compose.yaml -p irbench-postgres up -d --wait`) and run `IRBENCH_PG_URL=postgresql://postgres:irbench@localhost:15432/irbench cargo test -- --ignored`; tear it down (`... down -v`).
Expected: all pass.

- [ ] **Step 6: README and commit.** In `loadgen/README.md`, document the Postgres target (one connection per concurrent client, statements prepared per connection, `statement_timeout`, protocol check).

```bash
git add loadgen
git commit -m "feat: Postgres wire protocol in the loadgen"
```

---

### Task 6: Docs and CI

**Files:** `docs/methodology/parity.mdx`, `docs/reference/results.mdx`, `docs/contributing/new-engine.mdx`, `docs/index.mdx`, `README.md`, `.github/workflows/ci.yml`

- [ ] **Step 1: Docs.**
  - `parity.mdx`: replace the hybrid rule with the amended §7 text; add the SQL-engine note.
  - `results.mdx` (rendered requests): document the `postgres` protocol shape and the text-binding rule.
  - `new-engine.mdx`: SQL engines subclass `irbench.engines.postgres.PostgresAdapter` or implement `render` returning `SqlRequest`; `protocol = "postgres"`.
  - `index.mdx` / `README.md`: engine list now includes Postgres (built-in FTS + pgvector) and ParadeDB.
  - Run `cd docs && npx --yes mint@4.2.1006 broken-links`.
- [ ] **Step 2: CI.** In `.github/workflows/ci.yml`, the `loadgen` job gets a Postgres service and runs the ignored tests:

```yaml
  loadgen:
    runs-on: ubuntu-24.04
    services:
      postgres:
        image: pgvector/pgvector:0.8.7-pg17
        env:
          POSTGRES_PASSWORD: irbench
          POSTGRES_DB: irbench
        ports: ["15432:5432"]
        options: >-
          --health-cmd "pg_isready -U postgres -d irbench" --health-interval 2s --health-retries 30
```

and, after `cargo test`, `- run: cargo test -- --ignored` with `env: IRBENCH_PG_URL: postgresql://postgres:irbench@localhost:15432/irbench`. The `contract` job already runs every engine in `ENGINES`. Validate with `uvx --from actionlint-py actionlint .github/workflows/ci.yml`.
- [ ] **Step 3: Commit**

```bash
git add docs README.md .github/workflows/ci.yml
git commit -m "docs, ci: Postgres engines and the postgres request protocol"
```

---

### Task 7: Real runs

**Files:** none expected. A bug found here is fixed test-first in the task that owns the code.

- [ ] **Step 1: Index use on real data.** Run `uv run irbench run --scenario ecommerce/product-search --engine postgres --model bge-small-en-v1.5 --stages prepare,embed,up,index,render`, then with `psql` (`docker exec -it irbench-postgres psql -U postgres -d irbench`) run `EXPLAIN` on one rendered file per mode (substitute params): keyword must use the GIN index (`Bitmap Index Scan ... tsv`), semantic the HNSW index (`Index Scan using ..._embedding_idx`), filtered keyword the GIN indexes. Repeat for `postgres-paradedb` (keyword: `ParadeDB Base Scan`). If a mode does a sequential scan on 43k rows where the other engines use an index, report it with the plan output.
- [ ] **Step 2: Full runs** (background; each may take 20–40 minutes):

```bash
for e in postgres postgres-paradedb; do
  uv run irbench run --scenario ecommerce/product-search --engine $e --model bge-small-en-v1.5 2>&1 | tail -30
done
```

Expected: every stage `succeeded`; N/A only where declared; no constraint violations.
- [ ] **Step 3: Report.** `uv run irbench report --results .irbench/runs --out .irbench/site` and open the scenario page: five engines, colors stable, Postgres rows filled.
- [ ] **Step 4:** No commit. Put the nDCG@10 table, perf table and indexing cost for both Postgres engines in the final message, labelled `local`.

---

## Self-Review

**Spec coverage**

| Spec section | Where |
|---|---|
| §1 two engines, pinned images | Tasks 3, 4 (compose files), Task 1 (pins in spec) |
| §2 decisions (RRF both, wire-protocol perf, AND for vanilla, ef_search/iterative_scan, ParadeDB disjunction) | Tasks 3, 4, 5 |
| §3 request protocol | Task 2 (`SqlRequest`, `bind_param`), Task 5 (Rust `bind_param`) |
| §4.1 harness execution, statement_timeout, errors | Task 2 |
| §4.2 indexing (`COPY`, indexes, ANALYZE, settings) | Task 3 base |
| §4.3 loadgen | Task 5 |
| §5 mode mapping | Tasks 3, 4 |
| §6 error handling | Task 2 tests, Task 5 live test, Review Focus 2–3 |
| §7 parity amendment | Task 1 (spec), Task 6 (parity.mdx) |
| §8 testing (spike, unit, contract, loadgen, e2e) | spike done; Tasks 2–5 tests; Task 7 |
| §9 deliverables | Tasks 2–6 |

**Decided here:** all params bound as text with SQL casts (avoids custom types in two clients); pgvector settings via `ALTER DATABASE` (inherited by every connection, inspectable in the config); the synonym dictionary is a committed, generated file guarded by a test; one shared `PostgresAdapter` base (the two engines differ only in keyword search).

**Placeholder scan.** Code steps carry code. One step depends on a fact verified at execution time and states the exact fallback: the pg_search boost cast on a bound parameter (Task 4 Step 1). The engine loader's handling of a dashed folder name is checked in Task 4 with a stated fix.

**Type consistency.** `SqlRequest`, `load_request`, `engine_base_url`, `PG_DSN`, `PostgresClient.execute/script/copy/ping`, `wait_ready`, `execute` (relevance), `PostgresAdapter` hooks (`setup_ddl`, `extra_columns`, `text_index_ddl`, `keyword_match`), `Sql.p`, `ident`, Rust `Kind`, `bind_param`, `Target::connect/send/protocol` keep the same names across tasks.
