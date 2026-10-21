# IRBench — Speed at Two Page Sizes, and a Meilisearch Tuned Track

Date: 2026-10-14
Status: Draft, awaiting review
Parent spec: `2026-10-09-irbench-design.md` (amends §5 steps 5 and 7, and §11)

## 1. Goal

Today every speed number is measured with `top_k = 100` results per request. That is the right depth for relevance (recall@100) and for deep retrieval (reranking, RAG), but not what a search results page asks for. Some engines' cost grows with the result count: Meilisearch's semantic search takes 1 ms at `limit: 10`, 2 ms at 20 and 11 ms at 100. So speed measured only at 100 describes one use case.

This change:

1. measures speed at two page sizes, **20** (a search results page, the headline) and **100** (deep retrieval), with the same protocol for both;
2. leaves relevance exactly as it is (measured at `top_k = 100`);
3. adds a Meilisearch `tuned` track with `matchingStrategy: frequency`;
4. re-runs every engine on the current loadgen, which also replaces the stale Meilisearch and Elasticsearch speed numbers.

Non-goals:
- other page sizes;
- relevance at 20 (nDCG@10 does not depend on the page size, except through hybrid pool depth, see §3);
- a tuned track for other engines;
- the ranking-rule finding (ENGPROD-2912), which stays out of the tuned track for now.

## 2. Decisions (agreed with the user)

| Topic | Decision |
|---|---|
| Page sizes | `[20, 100]`, measured both. 20 is the headline in the report. |
| Request shape | Each request is what an application would send for a page of that size. Retrieval pools that adapters derive from the depth (Elasticsearch kNN `k`, Qdrant prefetch `limit`, Postgres CTE limits) follow the page size. |
| Engine-level settings | Unchanged per page size: index and database settings (pgvector `hnsw.ef_search = 100`, ParadeDB `plan_cache_mode`) are part of the engine config, not of the request. |
| Relevance | Unchanged: `top_k = 100`, same rendered files, same scores. |
| Meilisearch tuned track | `matchingStrategy: frequency` on the keyword and hybrid bases, with a justification. |

## 3. Scenario and rendering (amends parent §5 step 5)

### 3.1 Scenario

`scenario.yaml` gains one key under `perf`:

```yaml
perf:
  page_sizes: [20, 100]   # first entry is the headline; each must be <= top_k
```

- It defaults to `[top_k]` when absent, so other scenarios keep today's behavior.
- Validation:
  - non-empty;
  - unique;
  - each value is between 1 and `top_k`.

### 3.2 Adapter contract

`EngineAdapter.render(task, vector, limit, facets=True)` becomes:

```python
def render(self, task, vector, limit, *, depth, facets=True) -> Request
```

- **`limit`**: how many hits the request returns.
- **`depth`**: the retrieval pool size, i.e. how many candidates each retrieval side considers. This is everything that used `scenario.top_k` inside an adapter:
  - Elasticsearch: kNN `k`.
  - Qdrant: the prefetch `limit` (`max(limit, top_k)` today) and the semantic sorted pool.
  - Postgres: the vector and keyword CTE `LIMIT`s, the hybrid RRF sides, and the semantic sorted pool.
  - Meilisearch: no pool parameter; `depth` is unused.
- The render stage passes these values:

| Rendered set | `limit` | `depth` |
|---|---|---|
| Relevance and perf at 100 | `top_k` | `top_k` |
| Perf at page size p | p | p |
| Facet verification (`*.check.json`) | `facet_verify_limit` | `top_k` (unchanged meaning: re-fetch the same match set) |

- Adapters stop reading `scenario.top_k` directly. Pools that today take `max(limit, top_k)` become `max(limit, depth)`, so facet checks stay as they are.
- A unit test asserts each adapter renders identical requests at `(top_k, top_k)`, and identical facet checks, before and after the change.
- So `rendered/` is byte-identical to today, and relevance is unchanged.

### 3.3 Rendered directories

- `rendered/`: as today (`limit = depth = top_k`), used by relevance and, when `top_k` is in `page_sizes`, by perf at that size.
- `rendered-p<N>/`: one directory per other page size, with the same layout `<mode>/<base>/<qid>.json`, and no facet check files.
- Unsupported tasks are the same at every size (N/A does not depend on the page size).

## 4. Perf stage (amends parent §5 step 7)

- The loadgen binary is unchanged. The harness runs it once per page size, in `page_sizes` order:
  - same protocol each time (warm-up pass, latency at concurrency 1, throughput ramp on the throughput groups);
  - `--requests` points to that size's directory;
  - output goes to `perf-p<N>.json`.
- Each file has today's `perf.json` format.
- `scores.json` `perf` becomes keyed by page size:

  ```json
  "perf": {"20": {"keyword/keyword": {"p50_ms": ..., "max_qps": ...}, ...}, "100": {...}}
  ```

- **Backward compatibility:** a bundle with a single `perf.json` (and a flat `scores.perf`) is read as page size 100. The report, `score`, and `publish` all accept both shapes. No old bundle needs rewriting.
- Errors: as today, per run. A failure at any page size fails the perf stage. A run is never published with one size missing.
- Cost: the stage takes about twice as long (about 7 more minutes per engine with the current scenario).

## 5. Report (amends parent §11)

- **Scenario page:** speed tables and the max-QPS chart show the headline size (20). A per-section switch shows 100. The section titles say what each size represents:
  - "Page of 20 results (search results page)"
  - "Page of 100 results (deep retrieval: reranking, RAG)"
- **Run page:** latency and throughput tables for each page size, headline first.
- **Performance methodology page:** explains both sizes, why both exist, and that pools follow the page size while engine-level settings do not.
- **Legacy bundles** show only a 100 column, with "not measured" at 20.
- The CLI summary prints one line per group per page size.

## 6. Meilisearch tuned track

In `engines/meilisearch/configs/ecommerce/product-search.yaml`:

```yaml
tuned:
  search:
    keyword: {matchingStrategy: frequency}
    hybrid:  {matchingStrategy: frequency}
  justification:
    matchingStrategy: >-
      With the default `last`, Meilisearch drops words from the end of the query
      until something matches, so the product type (usually the last word) goes
      first and an unknown leading brand or modifier is kept. `frequency` drops
      the most frequent words first. Measured on WANDS (v1.54.3): keyword nDCG@10
      0.481 -> 0.508, zero-result queries 87 -> 14. A documented, supported
      search parameter; no ranking-rule change.
```

- The adapter already deep-merges `search.<base>` into request bodies of that base, and filtered, sorted and faceted tasks on the keyword base inherit it.
- `irbench lint` must pass. If the lint does not yet require a justification for search parameters, the justification is still written; the lint is not changed by this work.

## 7. Runs

After implementation, on the laptop (`local` profile, not publishable):

1. All five engines, default track, full pipeline. Relevance must match the current scores exactly; any difference is a bug in §3.2.
2. Meilisearch tuned track, full pipeline.
3. Rebuild the local report, and refresh the Linear project's detailed-results note with both page sizes.

## 8. Testing

- **Scenario:** `page_sizes` default, validation errors (empty, duplicate, 0, above `top_k`).
- **Adapters:** for each engine:
  - rendering at `(top_k, top_k)` is identical to the pre-change output, using golden fixtures from the current adapters;
  - at `(20, 20)`, `limit` and every pool are 20;
  - facet checks keep `depth = top_k`.
- **Render stage:** writes `rendered/` and `rendered-p20/`; N/A sets are equal; no check files in `rendered-p20/`.
- **Perf stage:**
  - with a fake runner, one loadgen call per page size with the right `--requests` and `--out`, in order;
  - the summary is keyed by page size;
  - a failing size fails the stage.
- **Score, report and publish:**
  - legacy single-`perf.json` bundles still load and render as 100;
  - new bundles render both sizes;
  - the headline is the first page size.
- **Lint:** the Meilisearch config with the tuned block passes.

## 9. Deliverables

- Harness:
  - `scenario.py` (`perf.page_sizes`);
  - adapter base and all five adapters (`depth`);
  - `stages/render.py`, `stages/perf.py`, `stages/score_run.py`, `stages/run.py` (stage outputs), `cli.py`;
  - `publish.py` (copies `perf-p<N>.json` files).
- Report: `report/data.py`, `report/render.py`, templates.
- Config: Meilisearch `tuned` block; `scenarios/ecommerce/product-search/scenario.yaml` `page_sizes: [20, 100]`.
- Docs:
  - `docs/methodology/performance.mdx`;
  - parity page (pools follow the page size);
  - parent spec §5 and §11 amended to point here.
- Runs and report per §7.
