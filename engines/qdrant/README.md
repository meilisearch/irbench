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

## What the tuned track changes

Found by a sweep scored on nDCG@10 per scenario and base; each value is justified in `configs/<scenario>.yaml`.

- keyword, every scenario: `keyword_pool: 1000`, each per-field BM25 prefetch keeps 1000 candidates instead of the request depth.
- index, the three knowledge-base scenarios: `bm25_avg_len: corpus` (each field's BM25 `avg_len` is its mean word count in the corpus, instead of Qdrant's 256 for every field), plus `bm25_options` `k: 1.5` (web-questions, community-qa) or `b: 0.5` (covid-papers). WANDS keeps Qdrant's BM25 defaults.
- hybrid: distribution-based score fusion (`fusion: {fusion: dbsf}`) on the knowledge-base scenarios; on WANDS, RRF with the vector side weighted 3x (`{rrf: {k: 60, weights: [1, 3]}}`).

Config keys the adapter applies itself: `index.bm25_avg_len`, `index.bm25_options` (sent with every document's BM25 inference), `search.<base>.fusion` (the hybrid query, default `{fusion: rrf}`) and `search.<base>.keyword_pool` (the per-field prefetch limit).

## Modes

- semantic: Query API `query: <vector>, using: dense`.
- keyword: one prefetch per searchable field (`using: <field>`, `limit` = `max(limit, depth)`, where depth is `top_k` = 100 for relevance, the page size for speed), combined by a `formula` weighted sum. Field priority follows the scenario order with weight `n - i` (`name` 5 ... `description` 1), missing scores default to 0. This mirrors the Elasticsearch adapter's `name^5 ... description^1` boosts, with one difference: Elasticsearch's `multi_match` takes the best field, while Qdrant formulas have no max, so the fields are summed.
- hybrid: the keyword query above and the dense query as two prefetches, fused by Qdrant's native `fusion: rrf`.
- filtered (any base): `filter.must` with `match` / `range`, applied in every prefetch and at the top level.
- sorted (any base): `prefetch` the candidates, then rank them with a `formula` query that is the sort field alone (`neg` of it for ascending), so the relevance score is not mixed in. Missing values get a `defaults` entry of `-1e18` (`1e18` before the negation for ascending) and sort last. Qdrant's `order_by` was not used because it leaves out points that lack the field. Candidates: on the keyword base every match (each per-field BM25 prefetch is limited to the collection's document count, and a sparse query only returns points sharing a term), like Elasticsearch and Meilisearch, which sort all matches; on the semantic base the nearest `depth` (`top_k` for relevance, the page size for speed), like the other engines' kNN with `k` = depth.
- N/A, with the reason in the config: faceted on every base (Qdrant's facet API counts the points matching a filter, not a query's results).
- Synonyms: `synonyms.txt` is not loaded (Qdrant's BM25 tokenizer takes no synonym list; expanding queries in the harness would break parity). Declared with the config's `synonyms:` reason.

## Known limits

- BM25 scoring uses `qdrant/bm25` with Qdrant's built-in options as shipped (term-frequency saturation, length normalization and the average document length are Qdrant's defaults, not computed per field by the adapter). The IDF modifier is enabled on every sparse vector.
- Unsorted keyword and hybrid prefetch `depth` documents per field (`top_k` for relevance, the page size for speed): a document outside every field's top `depth` cannot be returned, which can cost a little recall against engines that rank all matches.
- The `prefetch` lists are built in code and are not deep-merged from the YAML `search` block, which only merges top-level query-body keys; BM25 options, the keyword pool and the fusion are set through the keys listed under the tuned track.

## Changing this configuration

Edit `configs/ecommerce/product-search.yaml`. Removing an `unsupported` entry requires implementing that mode in `adapter.py`. Run `uv run pytest -m docker -k qdrant` before opening a PR.
