# IRBench — Postgres Engines Design

Date: 2026-10-12
Status: Draft, awaiting review
Parent spec: `2026-10-09-irbench-design.md` (this document amends §5 and §7 where stated)

## 1. Goal

Benchmark Postgres as two engines, with the extensions people actually use for search:

| Engine folder | Image | Search stack |
|---|---|---|
| `engines/postgres` | `pgvector/pgvector:0.8.7-pg17` (digest recorded) | built-in full-text search (`tsvector`) + pgvector |
| `engines/postgres-paradedb` | `paradedb/paradedb:0.26.1-pg17` (digest recorded; ships pg_search 0.26.1 and pgvector 0.8.6) | pg_search (BM25) + pgvector |

Both run the existing scenario (`ecommerce/product-search` on WANDS) through the full pipeline (relevance and performance) and appear in the report next to Meilisearch, Elasticsearch and Qdrant. Comparing the two Postgres engines shows what the extension adds.

Non-goals: other engines; Postgres-specific scenarios; the published-vector reuse fix (tracked separately).

## 2. Decisions (agreed with the user)

| Topic | Decision |
|---|---|
| Postgres variants | Both: vanilla Postgres 17 + pgvector, and ParadeDB (pg_search + pgvector), as two engines. |
| Hybrid | Reciprocal Rank Fusion written as one SQL statement that runs inside the database, for both engines. This is a deliberate amendment to parent §7 (see §7 below). |
| Performance | The Rust loadgen speaks the Postgres wire protocol natively (`tokio-postgres`). No HTTP proxy. |
| Vanilla keyword matching | Postgres as shipped: `websearch_to_tsquery('english', …)`, which requires every term (AND). Documented; a tuned track may change it. |
| pgvector `ef_search` | `hnsw.ef_search = 100` (the default 40 caps results below the scenario's `top_k = 100`) and `hnsw.iterative_scan = relaxed_order` (filtered vector queries can fill top_k). Both are set with a `justification:` as the minimum to load the scenario. |
| ParadeDB matching | Disjunction (`|||`), matching Elasticsearch's `multi_match` and BM25 convention. |

## 3. Request protocol (amends parent §5 step 5)

Rendered request files gain a second protocol. The parent spec's `{protocol: http|postgres, method, url, headers, body | sql, params}` becomes, matching the implementation:

```json
{"protocol": "http", "method": "POST", "path": "/indexes/i/search", "headers": {}, "body": {...}}
{"protocol": "postgres", "sql": "SELECT id FROM docs WHERE ... LIMIT $3", "params": ["sofa", [0.01, ...], 100]}
```

- `sql` uses positional placeholders `$1..$n`; `params` is a JSON array. Typed values: strings, numbers, booleans, null, and a vector as a JSON array of floats.
- Params are bound as text. Every placeholder is written `$n::text`, optionally followed by a further cast (`$2::text::vector`, `$3::text::int`).
- Bind rule, identical in the Python and Rust clients: null → NULL; bool → `true`/`false`; int → decimal; float → shortest repr; string → itself; list of numbers → pgvector literal `[a,b,...]`.
- The file stays the single source of truth: relevance and perf execute exactly this SQL with these params.
- Results are rows. The adapter's `parse` reads them: column `id` (text) in rank order; for faceted queries, facet rows are returned by the same statement (see §5).

## 4. Execution

### 4.1 Harness (relevance and index stages)

- `Request` becomes a tagged union: `HttpRequest` (today's fields) and `PostgresRequest(sql, params)`. `Request.from_json` dispatches on `protocol`; existing HTTP files load unchanged.
- A `PostgresClient` (psycopg 3) replaces `httpx.Client` for Postgres engines in the client factory: one connection, `statement_timeout` = scenario `request_timeout_s` (10 s), autocommit. Request errors (SQL error, timeout) are `EngineError`s, so relevance scores the task 0 and logs it to `errors.jsonl`, exactly as for HTTP. A failed connection fails the stage.
- Engine adapters keep the same interface (`create_index`, `load_documents`, `wait_until_indexed`, `render`, `parse`); Postgres adapters receive the Postgres client.

### 4.2 Indexing

- Schema: one table `docs` with the scenario's fields (text fields `text`, numeric fields `double precision`, multi-valued fields `text[]`), the original id as `id text primary key`, and `embedding vector(dim)`.
- Load with `COPY ... FROM STDIN` in batches; then `CREATE INDEX` for text search (GIN or BM25), HNSW on `embedding` (`vector_cosine_ops`, defaults `m = 16`, `ef_construction = 64`), and B-tree indexes on filterable/sortable numeric columns. Then `ANALYZE`.
- pgvector settings (`hnsw.ef_search`, `hnsw.iterative_scan`) are applied with `ALTER DATABASE irbench SET ...`, which every new connection picks up (including the loadgen's), and with `SET` on the indexing session.
- Vanilla synonyms come from a dictionary file committed in `engines/postgres/docker/tsearch/` and mounted read-only into `/usr/share/postgresql/17/tsearch_data/` (that directory is root-owned, so the file cannot be written from a SQL session).
- The BM25 index (ParadeDB) is `CREATE INDEX ... USING bm25 (id, <fields>)` without `key_field`, which is deprecated and a no-op in pg_search 0.26.
- "Indexed" = every `CREATE INDEX` and `ANALYZE` has returned. Indexing time covers load + index builds. Memory and disk come from the same container metrics as the other engines (`memory.peak`, data directory size).

### 4.3 Loadgen (perf stage)

- `requests.rs` accepts `protocol: "postgres"` files; `Target` gains a Postgres variant (connection string from the run config, passed by the harness like `--base-url` today).
- One `tokio-postgres` connection per concurrent client, opened before the measured phase. Statements are prepared once per connection per distinct SQL text, then executed with each file's params.
- A request completes when all rows are received. Errors (SQL error, timeout via `statement_timeout`, broken connection) count as errors; the step-validity and max-QPS rules are unchanged.
- The HTTP path is unchanged.

## 5. Mode mapping

| Mode | `postgres` | `postgres-paradedb` |
|---|---|---|
| Keyword | Generated `tsvector` column with GIN index; `websearch_to_tsquery('english', $1::text)`; `ts_rank` with default weights | BM25 index (pg_search) over the searchable fields, no `key_field`; per-field `col ||| $n::text::pdb.boost(k)` combined with OR; scored with `pdb.score(id)`. The default tokenizer does not stem (same as Elasticsearch's standard analyzer) |
| Field priority | `setweight` A–D in scenario order; 5 fields onto 4 classes: `name` A, `product_class` B, `category` C, `features` and `description` D (documented) | Per-field boost `n − i` (`name` 5 … `description` 1), the same precedent as the Elasticsearch and Qdrant adapters |
| Typo / prefix | None (not in Postgres FTS by default); documented | Fuzzy matching is opt-in in ParadeDB, off in the default track; a tuned track may enable it with a justification |
| Synonyms | Built-in `thesaurus` dictionary (sub-dictionary `english_stem`, so both sides are stemmed and unify) in the text search configuration, built from `synonyms.txt`. Groups with digit tokens (`3 seater`, `2 drawer`) are dropped because `websearch_to_tsquery` does not expand multi-word thesaurus phrases in queries; they are listed in the README. | Not supported by pg_search 0.26.1; declared with `synonyms: <reason>` |
| Semantic | `ORDER BY embedding <=> $1::text::vector LIMIT top_k` with HNSW | same |
| Hybrid | RRF (k = 60) over a keyword CTE and a vector CTE (each top_k), one statement | same pattern, as published in ParadeDB's hybrid search docs |
| Filtered | `WHERE` predicates (`=` on text, `>=`/`<=` on numeric) | `WHERE` predicates, or pg_search's own filter syntax if ParadeDB documents it as the way to keep the BM25 index in play |
| Sorted | `ORDER BY <field> <dir> NULLS LAST`: over every keyword match on the keyword base, over the top_k nearest on the semantic base (same rules as the other engines) | same |
| Faceted | Facet counts over the query's own match set (keyword: every match; semantic: every document passing the filter), in the same statement as the hits (`GROUP BY` over that set, returned alongside the top `limit` hits) | same |

All parameters not listed here are each engine's defaults. Every non-default setting in `configs/ecommerce/product-search.yaml` carries a `justification:` (enforced by `irbench lint`).

## 6. Error handling

- Relevance: SQL error or timeout → task scored 0, logged in `errors.jsonl`; the run fails above the scenario's `max_error_rate`, as today.
- Perf: errors counted per step; a step above 0.1 % errors is invalid.
- Index: an index build exceeding `index_timeout_s` fails the run.
- Lint: `image-pin` applies to both compose files; config rules unchanged.

## 7. Parity amendment (amends parent §7)

Parent §7 says: "Hybrid uses the engine's native fusion only. No harness-side fusion. Otherwise N/A."

Amended: "Hybrid uses the engine's native fusion. For SQL engines, a fusion query written in SQL and executed entirely inside the database (e.g. Reciprocal Rank Fusion over two CTEs) counts as native; the SQL is part of the rendered request and documented in the engine README. Harness-side fusion is never allowed."

`docs/methodology/parity.mdx` is updated to match.

## 8. Testing

- Spike first, on the pinned images: pg_search boost syntax, default tokenizer, synonym support, pgvector `iterative_scan`, the RRF and facet statements, and `EXPLAIN` showing each mode uses its index (GIN, BM25, HNSW) rather than a sequential scan.
- Unit: rendered SQL and params per mode for both adapters; `Request` round-trip for both protocols; loadgen parsing of Postgres files.
- Contract (Docker): both engines join `harness/tests/test_contract.py`'s known-answer tests for every mode they claim; declared N/A and synonym skips come from their configs.
- Loadgen: the Postgres path is tested against a real Postgres container (gated so plain `cargo test` stays offline; run in CI's contract job).
- End to end: both engines run the full pipeline locally; the report shows them next to the other engines.

## 9. Deliverables

- `engines/postgres/` and `engines/postgres-paradedb/`: adapter, config, README (parity notes, N/A and synonym declarations, known limits), `docker/compose.yaml`.
- Harness: `PostgresRequest`, Postgres client, relevance/index stage support, perf-stage wiring (connection string to the loadgen).
- Loadgen: Postgres target and tests.
- Docs: parity amendment, request-format reference, engine list; parent spec §5 and §7 updated.
- CI: contract and loadgen jobs cover the Postgres engines.
