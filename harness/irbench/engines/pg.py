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
                self.dsn,
                autocommit=True,
                row_factory=dict_row,
                cursor_factory=psycopg.RawCursor,
                connect_timeout=10,
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
