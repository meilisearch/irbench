# IRBench — Design Spec

Date: 2026-10-09
Status: Draft, awaiting review

## 1. Purpose

IRBench is a public, reproducible benchmark that measures **relevance** and **performance** of search engines and embedding models on **realistic, imperfect** search scenarios (e-commerce, media, knowledge base, …).

Engines in scope: Meilisearch, Typesense, Elasticsearch, OpenSearch, Lucene, Qdrant, Pinecone, Postgres, Milvus.

Non-goals: proving any engine is better. The project is maintained by Meilisearch but must be credible to competitors.

### Success criteria

- Anyone can reproduce a published result with one command on the reference hardware profile.
- Every engine's configuration is declarative, isolated, and editable by that engine's community through a PR.
- Every request sent to an engine is inspectable as a file.
- Results show relevance broken down by query imperfection (typo, prefix, synonym), and performance (latency, throughput, indexing cost) separately — never blended into one score.

## 2. Key decisions

| Topic | Decision |
|---|---|
| Audience | Public repo and results; competitors may PR their engine configs. |
| Queries & judgments | Hybrid: real-query public datasets with human qrels as the base, plus a seeded, documented perturbation layer (typo, prefix, synonym). LLM-generated scenarios only as a separately labelled track (not in v0.1). |
| Performance environment | One pinned reference machine, all engines self-hosted in Docker, run one at a time. |
| Pinecone | No self-hosted offering → relevance-only, run against Pinecone's managed service; excluded from performance tables and marked "not comparable". |
| Search modes | keyword, semantic, hybrid, filtered, faceted, sorted. Unsupported = N/A. |
| Embeddings | Open + API models; vectors computed once, cached and published so reproduction needs no API keys. Engines never embed themselves. |
| Architecture | "Requests as data": Python harness + adapters render raw requests; Rust load generator replays the identical requests for performance. |
| Report | Static HTML generated from results JSON. No frontend framework. |
| Docs | Mintlify site in `docs/` (methodology, parity rules, contributing). |

## 3. Core concepts

- **Dataset** — corpus + real queries + qrels. Pinned by version and checksum. Raw data is downloaded, never committed.
- **Scenario** — a use case on top of a dataset (e.g. `ecommerce/product-search`). Declares the field schema (searchable fields in priority order, filterable, sortable, facet fields), the query set, the supported modes, the synonym file, the filter/sort/facet query templates.
- **Query set** — original queries plus perturbed variants. Each variant has `variant_of: <qid>`, `perturbation: original|typo|prefix|synonym`, and inherits the source query's qrels.
- **Engine adapter** — Python module per engine. Responsibilities: create index from scenario schema + engine config, load documents (with vectors), wait for indexing completion, render a (query variant × mode) into a raw request.
- **Engine config** — YAML per engine per scenario: `engines/<engine>/configs/<scenario>.yaml`. The only file most contributors touch.
- **Run** — (scenario × engine × model × mode × track) → TREC run file, metrics JSON, perf JSON, `manifest.json`.

## 4. Repository layout

```
irbench/
  datasets/            # loaders, checksums, licenses
  scenarios/<domain>/<name>/
    scenario.yaml      # schema, modes, templates
    synonyms.txt       # shared synonym list (reviewed, versioned)
    perturbations.yaml # rules + seed
  engines/<engine>/
    adapter.py
    configs/<scenario>.yaml
    docker/            # compose fragment, pinned image digest
    README.md          # parity notes, N/A justifications
  embeddings/          # model registry, embed + publish scripts
  harness/             # Python package + `irbench` CLI
  loadgen/             # Rust crate: replays rendered requests (HTTP, Postgres wire)
  report/              # static HTML generator + templates
  results/<run-id>/    # published outputs
  docs/                # Mintlify (mint.json, methodology, parity, contributing)
  hardware/            # reference hardware profiles
```

Python managed with `uv`. Rust loadgen is a standalone binary invoked by the harness.

## 5. Pipeline

Each stage reads and writes files; any stage is re-runnable and cacheable.

1. **prepare** — download + verify dataset; normalize to `docs.jsonl`, `queries.jsonl`, `qrels.tsv`; apply perturbations (seeded) → `queries.variants.jsonl`.
2. **embed** — for each model, embed docs and all query variants. Cache as Parquet keyed by (dataset version, model id, text sha256). Publish as release assets / Hugging Face dataset; `embed` downloads published vectors when available instead of recomputing.
3. **up** — start a single engine container with pinned image digest and resource limits from the hardware profile; wait for healthcheck.
4. **index** — adapter creates index and loads documents. Record wall-clock indexing time (until engine reports all tasks done), peak RSS, on-disk size.
5. **render** — adapter renders every (variant × mode) into the run directory, one file per query, containing the request format of the Postgres spec §3 (`2026-10-12-irbench-postgres-design.md`): `{protocol: http, method, path, headers, body}` or `{protocol: postgres, sql, params}`. HTTP requests carry a `path` instead of a `url`; the base URL comes from the run config. Path: `rendered/<mode>/<base>/<qid>.json` in the run dir.
6. **relevance** — execute each rendered request once, sequentially; parse result IDs via adapter; write TREC run file (top 100).
7. **perf** — Rust loadgen replays the same rendered files:
   - warm-up: one full pass, discarded;
   - latency: concurrency 1, report p50/p95/p99/mean;
   - throughput: step concurrency (1, 2, 4, 8, 16, 32, 64); max QPS is the highest step with p99 ≤ 100 ms and error rate ≤ 0.1%.
8. **score** — `ranx`: nDCG@10, Recall@10, Recall@100, MRR@10, Success@1; overall and per perturbation type.
9. **down** — stop container, delete volumes.

`manifest.json` per run: harness git SHA, engine image digest, config file hash + diff vs default config, hardware profile id, model id, dataset version + checksum, perturbation seed, timestamps, per-stage status.

CLI:

```
irbench run --scenario ecommerce/product-search --engine typesense --model bge-small-en-v1.5 [--mode hybrid] [--track default|tuned] [--stages prepare,embed,...]
irbench report --results results/ --out site/
```

### Scoring of filtered, sorted and faceted modes

Dataset qrels judge unfiltered queries, so these modes derive their ground truth deterministically:

- **filtered** — each query is paired with a filter from the scenario's filter templates (seeded selection among filters that keep ≥ 1 relevant doc). Effective qrels = original qrels ∩ docs matching the filter (evaluated by the harness on `docs.jsonl`). Also reported: **filter violations** (returned docs not matching the filter; must be 0).
- **sorted** — query + sort template (e.g. price asc). Reported: **order violations** (must be 0) and nDCG@10 of the sorted top-10 against the original qrels (does relevance survive sorting).
- **faceted** — query + facet fields. Reported: facet count accuracy vs counts computed by the harness over the engine's own full result set for that query (tests correctness, not relevance), plus latency.

## 6. Perturbations

Applied per original query, seeded, rules versioned with the scenario.

- **typo** — 1 edit for words of 4–7 chars, 2 edits for ≥ 8 chars; edit types: adjacent-key substitution (QWERTY map), deletion, duplication, transposition. Words < 4 chars untouched. At most one word per query perturbed by default.
- **prefix** — truncate the last word to 40–70% of its length (min 2 chars), simulating search-as-you-type.
- **synonym** — replace one term with an entry from a reviewed per-scenario vocabulary-mismatch list (e.g. sofa↔couch). Distinct from `synonyms.txt`: the perturbation list tests whether engines bridge the gap; `synonyms.txt` is the config engines may load. Queries with no applicable term produce no synonym variant.

Each variant inherits the source query's qrels.

## 7. Fairness / parity rules

Documented in `docs/methodology/parity.mdx`; machine-checkable rules enforced in CI.

- Same searchable fields and priority order (from scenario). Engines without field weighting use their closest native equivalent, noted in config.
- One shared `synonyms.txt` per scenario; engine either loads it or declares `synonyms: unsupported`.
- Same vectors, cosine similarity, same top-k. ANN parameters start at each engine's documented defaults; any change requires a `justification:` field.
- Hybrid uses the engine's native fusion. For SQL engines, a fusion query written in SQL and executed entirely inside the database (e.g. Reciprocal Rank Fusion over two CTEs) counts as native; the SQL is part of the rendered request and documented in the engine README. Harness-side fusion is never allowed. Otherwise N/A.
- Typo tolerance, prefix search, stemming: language defaults; explicit settings required to enable a feature at all (e.g. `fuzziness: AUTO`) are allowed and visible.
- Modes are supported natively or through a documented, commonly used extension (e.g. pgvector, ParadeDB/pg_search for Postgres). No custom workarounds.
- No per-query or qrels-fitted tuning. CI rejects configs containing query IDs or query strings from the query set.

**Tracks:**
- **default** — engine as shipped plus the minimum to load the schema.
- **tuned** — changes within the parity rules, contributed via PR, credited in the report.

**Contribution flow:** PR edits `engines/<engine>/configs/*.yaml` and/or `adapter.py` → CI runs adapter contract tests + smoke subset → maintainers re-run on the reference machine. Only reference-machine results are published.

## 8. Reference hardware (v0.1)

`hardware/aws-c7i-4xlarge.yaml`: AWS c7i.4xlarge (16 vCPU, 32 GB), gp3 SSD. Engine container pinned to 8 vCPU / 16 GB via cpuset and memory limit; loadgen pinned to a disjoint 4-vCPU cpuset. Profile id recorded in every manifest. Profiles can be added; results are only compared within a profile.

## 9. Error handling

- A failed stage marks that run `failed` with reason in the manifest; the batch continues; the report shows the failure explicitly.
- Relevance: request error/timeout (10 s) → empty result (scored 0), listed in the run's error log.
- Perf: errors counted; a concurrency step with error rate > 0.1% is invalid.
- Indexing timeout per scenario (`index_timeout` in scenario.yaml) → run fails.

## 10. Testing

- Unit: perturbation determinism (same seed → same output), qrels inheritance, TREC run parsing, rendered-request schema.
- Metric correctness: our scores vs `trec_eval` on a fixed fixture.
- Adapter contract tests: each adapter indexes a 100-doc fixture in Docker and must pass known-answer queries for every mode it claims. Run in CI on every PR.
- Loadgen: mock server with injected latency distribution; assert measured percentiles within tolerance.

## 11. Report

Static HTML generated from `results/`:
- Per scenario: engines × modes matrix (nDCG@10), perturbation breakdown (original vs typo/prefix/synonym deltas).
- Performance: latency percentiles, QPS curve, indexing time / RAM / disk.
- Model comparison per engine.
- Per run: config diff vs default, manifest, N/A reasons, credits for tuned configs.
- Links to Mintlify methodology docs.

## 12. Scope

### v0.1 (first vertical slice)

- Scenario: `ecommerce/product-search` on **WANDS** (Wayfair; ~43k products, 480 queries, human judgments, MIT license).
- Engines: Meilisearch, Elasticsearch, Qdrant.
- Modes: keyword, semantic, hybrid, filtered, sorted, faceted (N/A where unsupported).
- Models: `bge-small-en-v1.5` (open), `text-embedding-3-small` (API, cached + published).
- Full pipeline, loadgen, report, Mintlify methodology docs, CI contract tests.

### Later

- Engines: Typesense, OpenSearch, Lucene (thin Java HTTP wrapper), Postgres, Milvus, Pinecone (relevance-only).
- Datasets/scenarios: Amazon ESCI (e-commerce), TMDB (media), BEIR subsets (knowledge base).
- More models; multilingual queries; LLM-generated scenario track (labelled separately).
