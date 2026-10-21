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

| Sorted mode orders results by the field | `rankingRules` with `sort` moved first (shipped order otherwise unchanged) |

Everything else is Meilisearch's default: typo tolerance, prefix search, matching strategy, `pagination.maxTotalHits`.

### Why `sort` is first

With the shipped ranking rules, `sort` comes after `words`, `typo`, `proximity` and `attributeRank`, and vector search ranks by similarity before any rule. A `sort` parameter then only breaks ties: on WANDS, 1,272 of 1,540 sorted keyword queries returned results out of order. Moving `sort` first makes it strict. Queries without a `sort` parameter return identical results either way, because the rule is inactive without one (checked on v1.54.3 for keyword, semantic and hybrid queries).

## What the tuned track changes

Found by a sweep scored on nDCG@10 per scenario and base; each value is justified in `configs/<scenario>.yaml`.

- keyword, every scenario: `matchingStrategy: frequency` (drop the most frequent query words first, instead of the last ones).
- hybrid: `semanticRatio` 0.45 with `frequency` on WANDS, 0.55 on web-questions, 0.95 on covid-papers, and the default 0.5 with `frequency` on community-qa. Above 0.5, every ratio ranks about the same on the knowledge-base scenarios (the vector side leads).

## Modes

- keyword: `q`.
- semantic: `vector` with `hybrid.semanticRatio: 1.0`, no `q`.
- hybrid: `q` + `vector` with Meilisearch's default `semanticRatio` (native fusion).
- filtered / sorted / faceted: `filter`, `sort`, `facets` on top of keyword or semantic. On the semantic base, the vector query matches every document, so a strict sort orders the whole index by the field.

## Changing this configuration

Edit `configs/ecommerce/product-search.yaml`. Changes to `default` must stay "as shipped plus the schema"; anything else goes in a `tuned:` block. Run the contract tests (`uv run pytest -m docker -k meilisearch`) before opening a PR.
