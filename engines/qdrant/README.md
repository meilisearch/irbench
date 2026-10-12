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
- sorted (any base): `prefetch` the base's `top_k` results, then rank them with a `formula` query that is the sort field alone (`neg` of it for ascending), so the relevance score is not mixed in. Missing values get a `defaults` entry of `-1e18` (`1e18` before the negation for ascending) and sort last. Qdrant's `order_by` was not used because it leaves out points that lack the field.
- N/A, with the reason in the config: faceted on every base (Qdrant's facet API counts the points matching a filter, not a query's results).
- Synonyms: `synonyms.txt` is not loaded (Qdrant's BM25 tokenizer takes no synonym list; expanding queries in the harness would break parity). Declared with the config's `synonyms:` reason.

## Changing this configuration

Edit `configs/ecommerce/product-search.yaml`. Removing an `unsupported` entry requires implementing that mode in `adapter.py`. Run `uv run pytest -m docker -k qdrant` before opening a PR.
