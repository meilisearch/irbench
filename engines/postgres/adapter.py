"""Postgres 17 + pgvector, built-in full-text search (or PGroonga, `index.text_search: pgroonga`).

Parity notes: engines/postgres/README.md."""

import re
from typing import Any

from irbench.engines.postgres import PostgresAdapter, Sql, ident

WEIGHTS = "ABCD"
# Built-in snowball dictionaries (`<language>_stem`); any other configuration copies
# `simple`, and its thesaurus falls back to the `simple` dictionary.
STEM_LANGUAGES = frozenset(
    "danish dutch english finnish french german hungarian italian norwegian portuguese "
    "romanian russian spanish swedish turkish".split()
)
TEXT_SEARCHES = frozenset({"tsvector", "pgroonga"})
# index.pgroonga_tokenizer: PGroonga tokenizers for Han text (TokenBigram is its default).
PGROONGA_TOKENIZERS = frozenset({"TokenBigram", "TokenNgram"})
# index.text_search_config: zhparser (the zhparser extension, only in the image of
# docker/compose.zh.yaml) builds the configuration on zhparser's parser instead of copying a
# built-in one. Its content token types (noun, verb, adjective, idiom, exclamation, temporary
# word) map to `simple`, as zhparser's README configures it.
ZHPARSER = "zhparser"
ZHPARSER_TOKENS = "n,v,a,i,e,l"
# Token types english maps to english_stem (hyphenated words included).
WORD_TOKENS = "asciiword, word, asciihword, hword, hword_asciipart, hword_part"


class Adapter(PostgresAdapter):
    name = "postgres"
    host_port = 15432
    language_keys = ("index.text_search_config",)
    # rank: ts_rank (default) or ts_rank_cd; normalization: the rank functions' integer flags
    # (default 0); match: all (websearch_to_tsquery, every word) or any (its words OR-ed).
    search_keys = PostgresAdapter.search_keys | {"rank", "normalization", "match"}

    @property
    def base_config(self) -> str:
        """The built-in configuration the scenario's is copied from (index.text_search_config)."""
        return ident(str(self.ctx.config.index.get("text_search_config", "english")))

    @property
    def stem_dictionary(self) -> str:
        base = self.base_config
        return f"{base}_stem" if base in STEM_LANGUAGES else "simple"

    @property
    def text_search(self) -> str:
        """index.text_search: `tsvector` (built-in full-text search, the default) or `pgroonga`
        (the PGroonga extension, only in the image of docker/compose.pgroonga.yaml)."""
        value = self.ctx.config.index.get("text_search", "tsvector")
        if value not in TEXT_SEARCHES:
            raise ValueError(
                f"postgres: index.text_search {value!r} must be one of {sorted(TEXT_SEARCHES)}"
            )
        return str(value)

    @property
    def pgroonga_index(self) -> str:
        """Name of the PGroonga index: `pgroonga_condition(index_name => ...)` must repeat it,
        so it is built here (and kept under the 63-character identifier limit) for both."""
        return f"{self.table[:59]}_pgr"

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

        A thesaurus dictionary runs both sides through its sub-dictionary (the stemmer), so
        `kids` and `children` meet on the same lexeme; a plain synonym dictionary does not.
        """
        return "".join(f"{w} : {words[0]}\n" for words in self.thesaurus_groups() for w in words)

    def setup_ddl(self) -> str:
        if self.text_search == "pgroonga":
            return (
                "CREATE EXTENSION IF NOT EXISTS pgroonga;\nCREATE EXTENSION IF NOT EXISTS vector;"
            )
        if self.base_config == ZHPARSER:
            return self._zhparser_ddl()
        cfg = self.ts_config
        ths = (
            ""
            if self.ctx.config.synonyms or not self.thesaurus_groups()
            else (
                f"CREATE TEXT SEARCH DICTIONARY {cfg}_ths "
                f"(TEMPLATE = thesaurus, DictFile = {cfg}, Dictionary = {self.stem_dictionary});\n"
                f"ALTER TEXT SEARCH CONFIGURATION {cfg} ALTER MAPPING FOR {WORD_TOKENS} "
                f"WITH {cfg}_ths, {self.stem_dictionary};"
            )
        )
        return (
            "CREATE EXTENSION IF NOT EXISTS vector;\n"
            f"DROP TABLE IF EXISTS {self.table};\n"
            f"DROP TEXT SEARCH CONFIGURATION IF EXISTS {cfg};\n"
            f"DROP TEXT SEARCH DICTIONARY IF EXISTS {cfg}_ths;\n"
            f"CREATE TEXT SEARCH CONFIGURATION {cfg} (COPY = {self.base_config});\n{ths}"
        )

    def _zhparser_ddl(self) -> str:
        if self.thesaurus_groups() and not self.ctx.config.synonyms:
            raise ValueError(
                "postgres: the zhparser configuration takes no thesaurus; declare a `synonyms:` "
                "reason in the config"
            )
        cfg = self.ts_config
        return (
            "CREATE EXTENSION IF NOT EXISTS vector;\n"
            "CREATE EXTENSION IF NOT EXISTS zhparser;\n"
            f"DROP TABLE IF EXISTS {self.table};\n"
            f"DROP TEXT SEARCH CONFIGURATION IF EXISTS {cfg};\n"
            f"CREATE TEXT SEARCH CONFIGURATION {cfg} (PARSER = zhparser);\n"
            f"ALTER TEXT SEARCH CONFIGURATION {cfg} ADD MAPPING FOR {ZHPARSER_TOKENS} WITH simple;"
        )

    @property
    def pgroonga_tokenizer(self) -> str | None:
        value = self.ctx.config.index.get("pgroonga_tokenizer")
        if value is not None and value not in PGROONGA_TOKENIZERS:
            raise ValueError(
                f"postgres: index.pgroonga_tokenizer {value!r} must be one of "
                f"{sorted(PGROONGA_TOKENIZERS)}"
            )
        return value

    def extra_columns(self) -> list[tuple[str, str]]:
        if self.text_search == "pgroonga":
            return []
        cfg = self.ts_config
        parts = [
            f"setweight(to_tsvector('{cfg}', coalesce({f}, '')), '{WEIGHTS[min(i, 3)]}')"
            for i, f in enumerate(self.searchable)
        ]
        return [("tsv", f"tsvector GENERATED ALWAYS AS ({' || '.join(parts)}) STORED")]

    def text_index_ddl(self) -> str:
        if self.text_search == "pgroonga":
            # One index over all searchable columns: PGroonga weights them per column at
            # query time (see keyword_match).
            tokenizer = self.pgroonga_tokenizer
            with_ = f" WITH (tokenizer='{tokenizer}')" if tokenizer else ""
            return (
                f"CREATE INDEX {self.pgroonga_index} ON {self.table} USING pgroonga "
                f"((ARRAY[{', '.join(self.searchable)}])){with_}"
            )
        return f"CREATE INDEX ON {self.table} USING gin (tsv)"

    def _pgroonga_match(self, q: Sql, text: str, options: dict[str, Any]) -> tuple[str, str]:
        """`&@~` (PGroonga query syntax: every word, like websearch_to_tsquery) with the
        scenario's field boosts as column weights, scored by pgroonga_score. The score needs
        an index scan: a sequential scan scores every row 0."""
        unsupported = sorted(set(options) - {"rrf"})
        if unsupported:
            raise ValueError(f"postgres: pgroonga does not support search settings {unsupported}")
        n = len(self.searchable)
        weights = ", ".join(str(n - i) for i in range(n))
        cond = (
            f"pgroonga_condition({q.p(text)}, weights => ARRAY[{weights}], "
            f"index_name => '{self.pgroonga_index}')"
        )
        return (
            f"ARRAY[{', '.join(self.searchable)}] &@~ {cond}",
            "pgroonga_score(tableoid, ctid)",
        )

    def keyword_match(self, q: Sql, text: str, options: dict[str, Any]) -> tuple[str, str]:
        if self.text_search == "pgroonga":
            return self._pgroonga_match(q, text, options)
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
