# Qdrant

Image: `qdrant/qdrant:v1.19.2` (host port 16333). Adapter: `adapter.py`.

## What the default track sends

| Scenario requirement | Qdrant setting |
|---|---|
| Shared vectors | collection `vectors: {size, distance: Cosine}`; HNSW at Qdrant's defaults |
| Filterable / facet text fields | `keyword` payload indexes |
| Numeric filterable / sortable fields | `float` payload indexes |
| Document IDs | numeric IDs become integer point IDs, others UUIDv5; the original ID stays in the payload and is what results report |

## Modes

- semantic: Query API `query: <vector>`.
- filtered (semantic base): `filter.must` with `match` / `range`.
- sorted (semantic base): `prefetch` the `top_k` nearest points, then `order_by` the sort field. Qdrant's `order_by` leaves out points that lack the field.
- N/A, with the reasons in the config: keyword retrieval and everything built on it (keyword-based filtered/sorted/faceted, hybrid), and faceted on the semantic base.

## Changing this configuration

Edit `configs/ecommerce/product-search.yaml`. Removing an `unsupported` entry requires implementing that mode in `adapter.py`. Run `uv run pytest -m docker -k qdrant` before opening a PR.
