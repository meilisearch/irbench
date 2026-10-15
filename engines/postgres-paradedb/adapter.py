"""ParadeDB: pg_search (BM25) + pgvector. Parity notes: engines/postgres-paradedb/README.md."""

from irbench.engines.postgres import PostgresAdapter, Sql


class Adapter(PostgresAdapter):
    name = "postgres-paradedb"
    host_port = 15433

    def setup_ddl(self) -> str:
        return "CREATE EXTENSION IF NOT EXISTS pg_search;\nCREATE EXTENSION IF NOT EXISTS vector;"

    def text_index_ddl(self) -> str:
        # No key_field: deprecated no-op in pg_search 0.26 (the primary key is used).
        return f"CREATE INDEX ON {self.table} USING bm25 (id, {', '.join(self.searchable)})"

    def keyword_tiebreak(self) -> str:
        # pg_search pushes a Top-K order down to the BM25 index only when the whole ORDER BY
        # is the score: a trailing `, id` leaves a full scan plus a Postgres sort of every match.
        return ""

    def keyword_match(self, q: Sql, text: str) -> tuple[str, str]:
        ph = q.p(text)
        n = len(self.searchable)
        pred = " OR ".join(
            f"{f} ||| {ph}::pdb.boost({n - i})" for i, f in enumerate(self.searchable)
        )
        return f"({pred})", "pdb.score(id)"
