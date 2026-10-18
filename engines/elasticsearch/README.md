# Elasticsearch

Image: `elasticsearch:9.5.5`, single node, security disabled (host port 19200), self-generated trial license (see Hybrid below). Adapter: `adapter.py`.

## What the default track sends

| Scenario requirement | Elasticsearch setting |
|---|---|
| Searchable fields, in priority order | `text` fields; `multi_match` (`best_fields`) with boosts `name^5, product_class^4, category^3, features^2, description^1` — Elasticsearch has no attribute order, so boosts `n - i` encode the scenario order |
| Filterable / facet text fields | `<field>.raw` keyword sub-field |
| Numeric filterable / sortable fields | `float` |
| `synonyms.txt` | `synonym_graph` filter in the search analyzer `irbench_search` (standard tokenizer + lowercase) |
| Facet value limit (1000) | `terms` aggregation `size` |
| Shared vectors | `dense_vector`, `similarity: cosine`, other `index_options` at this version's default |
| Single node | `number_of_replicas: 0` |

Not enabled by default (allowed in `tuned` with a visible setting): `fuzziness`, prefix matching, stemming analyzers, the `linear` retriever.

## Modes

- keyword: `multi_match`.
- semantic: top-level `knn` with `k` = the request depth (`top_k` = 100 for relevance, the page size for speed) and the default `num_candidates`.
- hybrid: the `rrf` retriever over a `standard` retriever (the `multi_match` above) and a `knn` retriever (`k` = the request depth), `rank_constant: 60`, `rank_window_size` = the request depth. This is the same reciprocal rank fusion (k = 60, `depth` per side) as Qdrant and the two Postgres engines, run natively by Elasticsearch. No harness-side fusion.

### Hybrid: why RRF, and the trial license

Until 2026-10-18 hybrid sent `query` + `knn` in one request, where Elasticsearch adds the BM25 score to the vector score. BM25 scores are unbounded (often 10 to 50) and the cosine score lies in [0, 1], so the sum ranked almost exactly like keyword alone: hybrid nDCG@10 was 0.336 against 0.320 for keyword on covid-papers, 0.218 against 0.210 on community-qa. That measured the combination formula, not Elasticsearch's hybrid search.

The `rrf` retriever is Elasticsearch's documented hybrid search, but it is a paid feature: on the basic license it answers 403 `current license is non-compliant for [Reciprocal Rank Fusion (RRF)]` (so is the `linear` retriever). The compose file sets `xpack.license.self_generated.type: trial`, Elasticsearch's own 30-day trial, so every run (each starts on a fresh volume) can use it. In production this needs an Enterprise subscription or Elastic Cloud.
- filtered: `bool.filter` (keyword) or `knn.filter` (semantic, pre-filtering).
- sorted: `sort` on the numeric field.
- faceted: `terms` aggregations; for semantic, the aggregation covers the `k` nearest documents.

## Config override caveat

`search.<base>` is deep-merged into the request body, so an override that targets `query.multi_match` reaches only unfiltered keyword requests (filtered ones wrap the query in `bool`). Use `adapter.py` for changes that must reach every request shape.

## Changing this configuration

Edit `configs/ecommerce/product-search.yaml` and run `uv run pytest -m docker -k elasticsearch` before opening a PR.
