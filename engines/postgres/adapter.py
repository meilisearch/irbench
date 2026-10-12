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
        syn = (
            ""
            if self.ctx.config.synonyms
            else (
                f"CREATE TEXT SEARCH DICTIONARY {cfg}_syn (TEMPLATE = synonym, SYNONYMS = {cfg});\n"
                f"ALTER TEXT SEARCH CONFIGURATION {cfg} ALTER MAPPING FOR asciiword, word "
                f"WITH {cfg}_syn, english_stem;"
            )
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
