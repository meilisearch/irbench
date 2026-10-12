# ParadeDB (Postgres + pg_search + pgvector)

Image: `paradedb/paradedb:0.26.1-pg17` (Postgres 17, pg_search 0.26.1 BM25, pgvector 0.8, host port 15433, database `irbench`). Adapter: `adapter.py`, built on the shared `irbench.engines.postgres.PostgresAdapter`; only keyword search and the extension setup differ from the `postgres` engine.

## What the default track sends

| Scenario requirement | ParadeDB setting |
|---|---|
| Searchable fields | `text` columns plus one `bm25` index: `CREATE INDEX ... USING bm25 (id, name, product_class, category, features, description)`. No `key_field` (a deprecated no-op in pg_search 0.26; the primary key `id` is used) |
| Field priority | Per-field boost `n - i` in scenario order: `name` 5, `product_class` 4, `category` 3, `features` 2, `description` 1. Same precedent as Elasticsearch and Qdrant: one clause per field, the scores of the matching clauses are summed by `OR` |
| Analysis | pg_search's default tokenizer: no stemming (like Elasticsearch's standard analyzer); fuzzy matching is opt-in in pg_search and left off |
| Synonyms | Not loaded: pg_search 0.26.1 has no synonym feature, and expanding queries in the harness would break the parity rules. Declared as `synonyms:` in the config and shown on the run page |
| Filterable / facet text fields | `<field>__terms text[]` with a GIN index |
| Numeric filterable / sortable fields | `double precision` columns with a B-tree index |
| Shared vectors | `embedding vector(dim)`, HNSW `vector_cosine_ops` at pgvector's defaults (`m = 16`, `ef_construction = 64`) |
| Document loading | `COPY ... FROM STDIN`, 2000 rows per batch; indexes are built after the load, then `ANALYZE` |
| `hnsw.ef_search = 100`, `hnsw.iterative_scan = relaxed_order` | Same pgvector settings and justifications as the `postgres` engine, set per database |
| `plan_cache_mode = force_custom_plan` | Set per database. The boost cast on a bound parameter (`$1::text::pdb.boost(n)`) only works in a custom plan; once Postgres switches a prepared statement to a generic plan (psycopg after 5 executions, the loadgen always prepares) pg_search fails with "The right-hand side of the `\|\|\|` operator must be a text or text array value" |

## Modes

Every mode is one SQL statement. Placeholders are `$n::text` plus a cast; the query text is always a parameter.

- keyword: `name ||| $1::text::pdb.boost(5) OR product_class ||| $1::text::pdb.boost(4) OR ...`, ranked by `pdb.score(id) DESC` alone, with no `id` tie-break (see Known limits). `|||` is a match-disjunction: a document matches if any query term appears in the field (OR semantics, unlike the `postgres` engine's AND). The EXPLAIN plan shows `Custom Scan (ParadeDB Base Scan)`.
- semantic, filtered, sorted, faceted: as in the `postgres` engine (pgvector and the shared SQL).
- hybrid: reciprocal rank fusion, k = 60, in one statement: the BM25 top `top_k` and the vector top `top_k` (each with the filter) are numbered with `row_number()`, combined by `FULL OUTER JOIN`, scored `1/(60 + r)` per side.

### Boost cast on a bound parameter

`$1::text::pdb.boost(5)` is accepted when the parameter is bound and the plan is custom (verified on 0.26.1-pg17); `($1::text)::pdb.boost(5)` is equivalent. The adapter uses the first form. It fails on a generic plan, hence `plan_cache_mode = force_custom_plan` above.

## Known limits

- Score ties are broken arbitrarily. The shared SQL orders keyword results by `score DESC, id`, but pg_search only pushes a Top-K order down to the BM25 index when the whole `ORDER BY` is the score (a partial prefix is not enough), so on this engine plain, filtered and hybrid-keyword-side queries order by score alone and use `TopKScanExecState` (verified with EXPLAIN on 0.26.1-pg17). Documents with equal scores can appear in any order. Pushdown still cannot happen for sorted keyword (`ORDER BY <field>`, every match is sorted by Postgres) and for faceted keyword (the counts need every match, so the scan is `NormalScanExecState` and the hits are sorted from the match set).
- No synonyms (see above), so synonym-dependent queries score lower here than on engines that load `synonyms.txt`.
- No stemming with the default tokenizer.
- Disjunctive keyword matching with per-field boosts is not identical to BM25F: scores are per-field BM25 sums.
- Faceted on the semantic base scans every filtered document (the spec's definition of the match set).
- Hybrid fusion over `top_k` per side can return fewer than `limit` documents if the sides overlap heavily.

## Changing this configuration

Edit `configs/ecommerce/product-search.yaml` (`index.settings` are database-level GUCs). Run `uv run pytest -m docker -k paradedb` before opening a PR.
