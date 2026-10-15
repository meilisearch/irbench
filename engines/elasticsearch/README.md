# Elasticsearch

Image: `elasticsearch:9.5.5`, single node, security disabled (host port 19200). Adapter: `adapter.py`.

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

Not enabled by default (allowed in `tuned` with a visible setting): `fuzziness`, prefix matching, stemming analyzers, the `rrf`/`linear` retrievers.

## Modes

- keyword: `multi_match`.
- semantic: top-level `knn` with `k` = the request depth (`top_k` = 100 for relevance, the page size for speed) and the default `num_candidates`.
- hybrid: `query` + `knn` in one request; Elasticsearch adds the two scores (its documented native combination). No harness-side fusion.
- filtered: `bool.filter` (keyword) or `knn.filter` (semantic, pre-filtering).
- sorted: `sort` on the numeric field.
- faceted: `terms` aggregations; for semantic, the aggregation covers the `k` nearest documents.

## Config override caveat

`search.<base>` is deep-merged into the request body, so an override that targets `query.multi_match` reaches only unfiltered keyword requests (filtered ones wrap the query in `bool`). Use `adapter.py` for changes that must reach every request shape.

## Changing this configuration

Edit `configs/ecommerce/product-search.yaml` and run `uv run pytest -m docker -k elasticsearch` before opening a PR.
