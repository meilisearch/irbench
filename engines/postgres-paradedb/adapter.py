"""ParadeDB: pg_search (BM25) + pgvector. Parity notes: engines/postgres-paradedb/README.md."""

import re
from typing import Any

from irbench.engines.language import split_language
from irbench.engines.postgres import PostgresAdapter, Sql

# `pdb.<name>` or `pdb.<name>(<args>)`: arguments are bare words, numbers and single-quoted
# strings: no `;`, `-` or inner `)`, so the value can neither end the statement nor start
# a comment.
TOKENIZER_RE = re.compile(r"pdb\.[a-z_]+(\([\w\s,='.]*\))?")


class Adapter(PostgresAdapter):
    name = "postgres-paradedb"
    host_port = 15433
    language_keys = ("index.tokenizer",)
    # match: any (default, `|||`, a document matches if any word does) or all (`&&&`);
    # fuzzy: the edit distance of pg_search's fuzzy matching (off by default).
    search_keys = PostgresAdapter.search_keys | {"match", "fuzzy"}

    def setup_ddl(self) -> str:
        return "CREATE EXTENSION IF NOT EXISTS pg_search;\nCREATE EXTENSION IF NOT EXISTS vector;"

    @property
    def tokenizer(self) -> str | None:
        """index.tokenizer: a pg_search tokenizer expression such as `pdb.lindera(japanese)` or
        `pdb.simple('stemmer=spanish')`, cast onto every searchable column."""
        expr = self.ctx.config.index.get("tokenizer")
        if expr is None:
            return None
        if not isinstance(expr, str) or not TOKENIZER_RE.fullmatch(expr):
            raise ValueError(f"postgres-paradedb: index.tokenizer {expr!r} is not pdb.<name>(...)")
        return expr

    def text_index_ddl(self) -> str:
        # No key_field: deprecated no-op in pg_search 0.26 (the primary key is used).
        _, lang = split_language(self.ctx.config.index)
        filters = (["'stemmer=english'"] if lang.stemming else []) + (
            ["'stopwords_language=english'"] if lang.stop_words else []
        )
        tokenizer = self.tokenizer
        if tokenizer and filters:
            raise ValueError(
                "postgres-paradedb: index.tokenizer replaces stop_words/stemming; set one or the other"
            )
        # Token filters go on the default tokenizer (unicode_words), cast per field.
        if filters:
            tokenizer = f"pdb.unicode_words({', '.join(filters)})"
        cols = [f"({f}::{tokenizer})" if tokenizer else f for f in self.searchable]
        return f"CREATE INDEX ON {self.table} USING bm25 (id, {', '.join(cols)})"

    def keyword_tiebreak(self) -> str:
        # pg_search pushes a Top-K order down to the BM25 index only when the whole ORDER BY
        # is the score: a trailing `, id` leaves a full scan plus a Postgres sort of every match.
        return ""

    def keyword_match(self, q: Sql, text: str, options: dict[str, Any]) -> tuple[str, str]:
        ph = q.p(text)
        n = len(self.searchable)
        op = {"any": "|||", "all": "&&&"}[options.get("match", "any")]
        fuzzy = int(options.get("fuzzy", 0))
        cast = f"::pdb.fuzzy({fuzzy})" if fuzzy else ""
        pred = " OR ".join(
            f"{f} {op} {ph}{cast}::pdb.boost({n - i})" for i, f in enumerate(self.searchable)
        )
        return f"({pred})", "pdb.score(id)"
