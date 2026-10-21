# Typesense

Image: `typesense/typesense:30.2` (latest stable release; 31.0 is still a release candidate), single node, API key `irbench` (host port 18108). Adapter: `adapter.py`.

## What the default track sends

| Scenario requirement | Typesense setting |
|---|---|
| Searchable fields, in priority order | `string*` fields (a string or a list of strings); `query_by` all of them with `query_by_weights` `n - i` (`name 5, product_class 4, category 3, features 2, description 1` on WANDS), the same boosts the other engines get |
| Filterable / facet text fields | `string*`, `facet: true` for facet fields; exact filters `field:=`​`value`​. Fields that are not searchable keep every punctuation symbol (`symbols_to_index`): Typesense otherwise strips them, and `tags:=webserver` would also match `web-server` |
| Numeric filterable / sortable fields | `float` |
| Missing values | every field is `optional` (the datasets leave some values empty); `null` values are not sent |
| `synonyms.txt` | one synonym set (`<index>_synonyms`, multi-way groups) attached to the collection |
| Facet value limit (1000) | `max_facet_values` |
| Shared vectors | `float[]` with `num_dim`; cosine distance and HNSW parameters at this version's defaults |

Typo tolerance (`num_typos: 2`) and prefix search are on by default in Typesense, as in Meilisearch; nothing is set to enable them.

## What the tuned track changes

Search parameters only, found by a sweep scored on nDCG@10 per scenario and base (each value is justified in `configs/<scenario>.yaml`):

- keyword: keep dropping query words, the first ones first (`drop_tokens_mode: left_to_right`), until 250 documents match, with `text_match_type: max_weight` on WANDS and covid-papers and the default `max_score` on community-qa; on web-questions until 10 match, with `prioritize_token_position`. The default (`drop_tokens_threshold: 1`) needs every query word to match, which hurts long or partly-off queries most.
- hybrid, per scenario: the fusion weight `alpha` (through `vector_query_options`), `rerank_hybrid_matches`, and `text_match_type`; on WANDS also word dropping. Word dropping lowers hybrid relevance on the three knowledge-base scenarios, so it stays off there.
- Stemming and stop words stay off: the parity rules keep stemming at each engine's language default, and no other engine removes stop words.

## Modes

All requests go through `POST /multi_search` (one search per request), so a 384-float vector never sits in a URL.

- keyword: `q` + `query_by` + `query_by_weights`.
- semantic: `q: "*"` + `vector_query: vector:([...], k:<depth>)` (`depth` = `top_k` = 100 for relevance, the page size for speed).
- hybrid: `q` + `query_by` + `vector_query` in one search: Typesense's native rank fusion of the keyword and vector rankings, at its default weighting (`alpha` 0.3 for the vector side). No harness-side fusion.
- filtered: `filter_by`, applied to both sides.
- sorted: `sort_by: <field>(missing_values: last):<direction>`. The sort is strict; `missing_values: last` puts documents without the field last, which is what the scenario's order check expects.
- faceted: `facet_by` + `max_facet_values`; keyword facets count every match, semantic facets the `k` nearest documents.

## Limits and caveats

- Typesense returns at most 250 hits per page. The facet verification fetch (up to 1,000 hits) is rendered as one `multi_search` with pages 1 to 4; `parse` concatenates them. Relevance (100) and speed (20, 100) requests are single pages.
- `multi_search` answers HTTP 200 even when a search fails, with the error inside the result. `parse` raises on it, so the relevance stage counts it as an error. The load generator only sees HTTP status codes, so a failing search would not show as an error in the speed numbers; the relevance stage sends the same requests first and would report it.
- A filter value containing a backtick cannot be quoted in `filter_by`; the adapter refuses it (no scenario value has one).
- The image has neither `curl` nor `wget`; the compose healthcheck asks `/health` through bash's `/dev/tcp`.
- `exhaustive_search: true` was tried while tuning and left out: keyword and hybrid searches on WANDS hit Typesense's 30-second limit (HTTP 408).

## Changing this configuration

`index` in a config is merged into the create-collection body, except three keys the adapter applies itself: `searchable_fields` (merged into every searchable field, e.g. `stem: true`), `fields.<name>` (merged into one field) and `stopwords.<set>` (a list of words created as a stopword set, then used with `search.<base>.stopwords: <set>`). `search.<base>` is merged into every search of that base, except `vector_query_options` (e.g. `{alpha: 0.9}`), which is appended inside `vector_query`.

Edit `configs/<scenario>.yaml` and run `uv run pytest -m docker -k typesense` before opening a PR.
