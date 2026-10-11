# Meilisearch

Image: `getmeili/meilisearch:v1.54.3` (host port 17700). Adapter: `adapter.py`.

## What the default track sends

| Scenario requirement | Meilisearch setting |
|---|---|
| Searchable fields, in priority order | `searchableAttributes` in scenario order (Meilisearch ranks earlier attributes higher) |
| Filterable / facet fields | `filterableAttributes` |
| Sortable fields | `sortableAttributes` |
| `synonyms.txt` | `synonyms`, every term mapped to the other terms of its group |
| Facet value limit (1000) | `faceting.maxValuesPerFacet` |
| Shared vectors | embedder `default`, `source: userProvided`, documents carry `_vectors.default` |

Everything else is Meilisearch's default: ranking rules, typo tolerance, prefix search, matching strategy, `pagination.maxTotalHits`.

## Modes

- keyword: `q`.
- semantic: `vector` with `hybrid.semanticRatio: 1.0`, no `q`.
- hybrid: `q` + `vector` with Meilisearch's default `semanticRatio` (native fusion).
- filtered / sorted / faceted: `filter`, `sort`, `facets` on top of keyword or semantic.

## Changing this configuration

Edit `configs/ecommerce/product-search.yaml`. Changes to `default` must stay "as shipped plus the schema"; anything else goes in a `tuned:` block. Run the contract tests (`uv run pytest -m docker -k meilisearch`) before opening a PR.
