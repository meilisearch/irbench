"""Postgres 17 + pgvector, built-in full-text search. Parity notes: engines/postgres/README.md."""

import re
from typing import Any

from irbench.engines.postgres import PostgresAdapter, Sql, ident

WEIGHTS = "ABCD"
# Token types english maps to english_stem (hyphenated words included).
WORD_TOKENS = "asciiword, word, asciihword, hword, hword_asciipart, hword_part"


class Adapter(PostgresAdapter):
    name = "postgres"
    host_port = 15432
    # rank: ts_rank (default) or ts_rank_cd; normalization: the rank functions' integer flags
    # (default 0); match: all (websearch_to_tsquery, every word) or any (its words OR-ed).
    search_keys = PostgresAdapter.search_keys | {"rank", "normalization", "match"}

    @property
    def ts_config(self) -> str:
        return ident("irbench_" + re.sub(r"[^a-z0-9]+", "_", self.ctx.scenario.id.lower()))

    def thesaurus_groups(self) -> list[list[str]]:
        """Groups the thesaurus can unify on both sides.

        Groups with a digit token (`3 seater`) are dropped: websearch_to_tsquery does not run
        multi-word thesaurus phrases on the query side, so `3 seater` would be indexed as
        `three seater` yet searched as `3 & seater` and match nothing.
        """
        return [
            [w.lower() for w in group]
            for group in self.ctx.synonyms
            if not any(re.search(r"\d", w) for w in group)
        ]

    def thesaurus_file(self) -> str:
        """`member : first member` for every member of every kept group.

        A thesaurus dictionary runs both sides through its sub-dictionary (english_stem), so
        `kids` and `children` meet on the same lexeme; a plain synonym dictionary does not.
        """
        return "".join(f"{w} : {words[0]}\n" for words in self.thesaurus_groups() for w in words)

    def setup_ddl(self) -> str:
        cfg = self.ts_config
        ths = (
            ""
            if self.ctx.config.synonyms or not self.thesaurus_groups()
            else (
                f"CREATE TEXT SEARCH DICTIONARY {cfg}_ths "
                f"(TEMPLATE = thesaurus, DictFile = {cfg}, Dictionary = english_stem);\n"
                f"ALTER TEXT SEARCH CONFIGURATION {cfg} ALTER MAPPING FOR {WORD_TOKENS} "
                f"WITH {cfg}_ths, english_stem;"
            )
        )
        return (
            "CREATE EXTENSION IF NOT EXISTS vector;\n"
            f"DROP TABLE IF EXISTS {self.table};\n"
            f"DROP TEXT SEARCH CONFIGURATION IF EXISTS {cfg};\n"
            f"DROP TEXT SEARCH DICTIONARY IF EXISTS {cfg}_ths;\n"
            f"CREATE TEXT SEARCH CONFIGURATION {cfg} (COPY = english);\n{ths}"
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

    def keyword_match(self, q: Sql, text: str, options: dict[str, Any]) -> tuple[str, str]:
        query = f"websearch_to_tsquery('{self.ts_config}', {q.p(text)})"
        match = options.get("match", "all")
        if match == "any":
            # The same parsed query with its AND operators turned into OR (phrases kept).
            query = f"replace({query}::text, ' & ', ' | ')::tsquery"
        elif match != "all":
            raise ValueError(f"postgres: unknown match {match!r}")
        rank = options.get("rank", "ts_rank")
        if rank not in ("ts_rank", "ts_rank_cd"):
            raise ValueError(f"postgres: unknown rank function {rank!r}")
        norm = f", {int(options['normalization'])}" if "normalization" in options else ""
        return f"tsv @@ {query}", f"{rank}(tsv, {query}{norm})"
