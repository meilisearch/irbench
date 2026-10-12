# Qdrant

Image: `qdrant/qdrant:v1.19.2` (host port 16333). Adapter: `adapter.py`.

## What the default track sends

| Scenario requirement | Qdrant setting |
|---|---|
| Shared vectors | named dense vector `dense` (`size`, `distance: Cosine`); HNSW at Qdrant's defaults |
| Searchable fields | one sparse vector per field, named after it, `modifier: idf`; each point sends the field text as `{text, model: "qdrant/bm25"}` and the server tokenizes, stems and weights it |
| Filterable / facet text fields | `keyword` payload indexes |
| Numeric filterable / sortable fields | `float` payload indexes |
| Document IDs | numeric IDs become integer point IDs, others UUIDv5; the original ID stays in the payload and is what results report |

## Modes

- semantic: Query API `query: <vector>, using: dense`.
- keyword: one prefetch per searchable field (`using: <field>`, `limit: top_k`), combined by a `formula` weighted sum. Field priority follows the scenario order with weight `n - i` (`name` 5 ... `description` 1), missing scores default to 0. This mirrors the Elasticsearch adapter's `name^5 ... description^1` boosts, with one difference: Elasticsearch's `multi_match` takes the best field, while Qdrant formulas have no max, so the fields are summed.
- hybrid: the keyword query above and the dense query as two prefetches, fused by Qdrant's native `fusion: rrf`.
- filtered (any base): `filter.must` with `match` / `range`, applied in every prefetch and at the top level.
- sorted (any base): `prefetch` the candidates, then rank them with a `formula` query that is the sort field alone (`neg` of it for ascending), so the relevance score is not mixed in. Missing values get a `defaults` entry of `-1e18` (`1e18` before the negation for ascending) and sort last. Qdrant's `order_by` was not used because it leaves out points that lack the field. Candidates: on the keyword base every match (each per-field BM25 prefetch is limited to the collection's document count, and a sparse query only returns points sharing a term), like Elasticsearch and Meilisearch, which sort all matches; on the semantic base the `top_k` nearest, like the other engines' kNN with `k = top_k`.
- N/A, with the reason in the config: faceted on every base (Qdrant's facet API counts the points matching a filter, not a query's results).
- Synonyms: `synonyms.txt` is not loaded (Qdrant's BM25 tokenizer takes no synonym list; expanding queries in the harness would break parity). Declared with the config's `synonyms:` reason.

## Known limits

- BM25 scoring uses `qdrant/bm25` with Qdrant's built-in options as shipped (term-frequency saturation, length normalization and the average document length are Qdrant's defaults, not computed per field by the adapter). The IDF modifier is enabled on every sparse vector.
- Unsorted keyword and hybrid prefetch `top_k` documents per field: a document outside every field's top `top_k` cannot be returned, which can cost a little recall against engines that rank all matches.
- A tuned track that wants different BM25 options or prefetch limits edits `adapter.py` (spec section 7's contribution flow allows it): the `prefetch` lists are built in code and are not deep-merged from the YAML `search` block, which only merges top-level query-body keys.

## Changing this configuration

Edit `configs/ecommerce/product-search.yaml`. Removing an `unsupported` entry requires implementing that mode in `adapter.py`. Run `uv run pytest -m docker -k qdrant` before opening a PR.
