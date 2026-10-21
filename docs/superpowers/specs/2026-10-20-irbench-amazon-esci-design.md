# IRBench — Amazon product search and as-you-type scenarios (ESCI + AmazonQAC)

Date: 2026-10-20
Status: Approved 2026-10-21
Parent spec: `2026-10-09-irbench-design.md`

## 1. Goal

Three of the four scenarios today use question-style queries (CQADupStack, TREC-COVID, LoTTE). WANDS has real typed product queries, but only 480 of them, and every scenario sends complete queries. No scenario measures what a search box sees while someone types: unfinished words, half-typed queries, and typing mistakes.

This change adds two scenarios built on Amazon data:

| Scenario | Queries | Why |
|---|---|---|
| `ecommerce/amazon-search` | Real Amazon search queries (ESCI), complete | Short keyword queries as typed in a store's search box, at scale (2,000 queries), with human judgments |
| `ecommerce/as-you-type` | The prefixes real users typed on the way to those queries (AmazonQAC) | Search-as-you-type: relevance of the results shown for each partial query, scored against the intent the user finally searched for |

Both scenarios use one product corpus, one set of document vectors, and the same engine settings.

Non-goals:
- query autocomplete (suggesting the query itself); AmazonQAC is used only for its typed prefixes;
- the Japanese and Spanish locales of ESCI (the multilingual step stays a later item);
- personalisation or session context (AmazonQAC's `past_search_terms` are ignored);
- a keystroke-by-keystroke metric ("characters until a good first page"); see §10.

## 2. Decisions

| Topic | Decision | By |
|---|---|---|
| Datasets | Amazon Shopping Queries (ESCI) for products, queries and judgments; AmazonQAC for real typed prefixes | user |
| Scenarios | One complete-query scenario and one as-you-type scenario on the same corpus | user |
| Locale | `us` only | my call |
| Corpus | The full US catalogue, 1,215,854 products, if every engine fits it on the laptop VM (§7.1); otherwise the 165k-product fallback corpus (§3.1) for every engine | user |
| Grades | Exact 2, Substitute 1, Complement 0, Irrelevant 0, so nDCG uses the same 0–2 scale as WANDS; Complement and Irrelevant stay in the qrels as judged non-relevant, so judged@10 counts them | my call |
| Prefix source | One AmazonQAC train shard (`part-00000`, 19.8M typing sessions); both scenarios use the same 2,000 queries (§3.2) | user |
| Prefix text | Sent exactly as typed, trailing space included | my call |

## 3. Datasets

Both datasets are pinned by URL and sha256 in `datasets/<id>/dataset.yaml`. Raw data is downloaded, never committed.

### 3.1 `amazon-esci`

- **Source:** `github.com/amazon-science/esci-data`, pinned to a commit. Files (Git LFS, fetched through `media.githubusercontent.com`):
  - `shopping_queries_dataset_examples.parquet` (51 MB): `query_id, query, product_id, product_locale, esci_label, small_version, large_version, split`;
  - `shopping_queries_dataset_products.parquet`: `product_id, product_title, product_description, product_bullet_point, product_brand, product_color, product_locale`;
  - `shopping_queries_dataset_sources.csv`: `query_id, source` (how Amazon sampled the query).
- **License:** Apache-2.0.
- **Rows used:** `product_locale == "us"` only (97,345 queries and 1.82M judgments in the full US set).
- **Query pool:** the `small_version == 1`, `split == "test"` US queries (8,956), Amazon's "hard" sample, that have at least one Exact or Substitute product and at least one AmazonQAC typing session in the shard (§3.2): 2,075 queries.
- **Corpus:** every US product in the catalogue, 1,215,854 products (all but 3 are judged for some US query; 1.37 GB of title, bullet and description text; 53% have a description, 85% bullet points, 94% a brand, 67% a color). Products judged only for queries outside the pool are distractors, as in a real store.
- **Fallback corpus** (used only if §7.1 fails): every US product judged for a small-version test query, about 165k products. The choice is a loader parameter (`corpus: full | pool`) in `dataset.yaml`, so both builds share one code path.
  - `id`: `product_id` (ASIN);
  - `title`, `brand`, `color`: strings, stripped;
  - `bullet_points`: `product_bullet_point`, kept as one string;
  - `description`: `product_description` with HTML tags removed and entities unescaped (ESCI descriptions contain `<p>`, `<br>`, `<b>`).
  - Missing values are null. A product with no title is dropped with its judgments.
- **Complete queries (`queries.jsonl`):** a seeded sample of 2,000 queries from the pool, about the size of the knowledge-base scenarios (1,746 on community-qa). The seed (`20261020`) and the size are loader constants next to the normaliser, like WANDS's fixed rules: the sample is part of the dataset's definition, not a run setting (my call).
  - id `esci-<query_id>`; text with whitespace normalised, as for WANDS;
  - `group`: the query's `source` from `sources.csv`.
- **Qrels (`qrels.tsv`):** grades per §2. A query with no Exact or Substitute product is dropped and counted in `prepare.manifest.json`.

### 3.2 `amazon-qac` prefixes

- **Source:** `huggingface.co/datasets/amazon/AmazonQAC`, pinned to a revision; train shard `train/part-00000-1d4931ef-16a0-4dc1-8034-ddaa3f7ddec6-c000.snappy.parquet` (2.96 GB, 19,777,475 rows). Only `prefixes` (the sequence typed, in order) and `final_search_term` are read, in batches, so the shard is never loaded whole.
- **License:** CDLA-Permissive-2.0.
- **Matching:** a row matches an ESCI query when its `final_search_term`, lower-cased with whitespace collapsed, equals the ESCI query normalised the same way.
  - The shard matches 32,520 distinct US ESCI queries (2.36M sessions), 2,075 of them in the pool (§3.1).
  - Measured and not used: the AmazonQAC test split (20,000 single prefixes) matches only 1,524 ESCI queries, mostly popular ones.
- **Prefix queries (`queries.prefixes.jsonl`):** for each of the 2,000 complete queries:
  - one of its matching sessions, chosen with the dataset seed;
  - from that session's typed sequence, one prefix per completion group (below), chosen with the seed among the prefixes in that group;
  - plus the complete query itself, group `complete`, so the scenario carries its own full-query reference.
  - Measured on the shard: 7,429 prefix queries (2,000 `complete`, 1,480 `≤25%`, 1,430 `25–50%`, 1,084 `50–75%`, 675 `75–<100%`, 760 `diverges`; a session does not always reach every group).
  - id `<esci query id>-<group>`; text exactly as typed, trailing space included (engines treat a last word followed by a space as complete, without one as a prefix).
- **Prefix qrels (`qrels.prefixes.tsv`):** each prefix gets the qrels of its final ESCI query (grades per §2). The prefix is scored against what the user meant, not what they typed so far.
- **Groups**, by how much of the final query was typed:
  - `diverges` when the lower-cased prefix is not a literal prefix of the lower-cased final query (typing mistakes and corrections, as in `yettidr` → `yeti drinkware`; 16% of the shard's keystrokes);
  - otherwise by typed length over final length: `≤25%`, `25–50%`, `50–75%`, `75–<100%`;
  - `complete`: the final query.

### 3.3 Normalization rules

- §3.4 of the knowledge-base spec applies (empty documents dropped with their judgments; queries left with no positive dropped; counts in the manifest).
- Complete queries are whitespace-normalised. Prefix queries are not.

## 4. Scenarios

Both scenarios use `dataset: amazon-esci`, so they share `docs.jsonl` and the document vectors. Each one keeps its own index, tasks and query vectors, as today.

### 4.1 `ecommerce/amazon-search`

```yaml
id: ecommerce/amazon-search
dataset: amazon-esci
index_fields:
  searchable: [title, brand, color, bullet_points, description]
  filterable: [brand, color]
  sortable: []
  facets: [brand]
modes: [keyword, semantic, hybrid, filtered, faceted]
structured_bases: [keyword, semantic]
embedding_fields: [title, brand, color, bullet_points]
filter_templates:
  - {field: brand, op: eq}
  - {field: color, op: eq}
sort_templates: []
top_k: 100
index_timeout_s: 10800
perturbations:
  seed: 20261020
  kinds: [typo, prefix]
perf:
  page_sizes: [20, 100]
```

- `sorted` is not in the modes: ESCI has no price, rating or date.
- `brand` has 211k distinct values and `color` 175k (free text, as sellers enter it). The facet test still returns at most `max_facet_values` (1000), as on WANDS.
- Embedding text leaves out `description`. It is long and often marketing copy, and bge-small truncates at 512 tokens.
- The synthetic `prefix` perturbation stays here, so its scores can be read next to the real prefixes of §4.2.

### 4.2 `ecommerce/as-you-type`

```yaml
id: ecommerce/as-you-type
dataset: amazon-esci
query_set: prefixes
# index_fields and embedding_fields: written out in full, identical to
# ecommerce/amazon-search (lint checks it, §5.5)
modes: [keyword, semantic, hybrid]
structured_bases: []
filter_templates: []
sort_templates: []
top_k: 100
index_timeout_s: 10800
perturbations:
  seed: 20261020
  kinds: []
perf:
  page_sizes: [10, 20]
```

- No synthetic perturbations: the queries already are the real perturbations.
- Perf at page sizes 10 and 20: an as-you-type panel shows few results. Throughput uses the same groups as today; with these short queries it is the load a search-as-you-type box puts on an engine, one request per keystroke.

## 5. Harness changes

### 5.1 Query sets

- `Scenario.query_set: str | None = None`. When set, `prepare` reads `queries.<set>.jsonl` and `qrels.<set>.tsv` from the dataset directory instead of `queries.jsonl` and `qrels.tsv`.
- A normaliser may write extra query sets; `NormalizedDataset` gains `query_sets: dict[str, tuple[Path, Path]]`, and the manifest records per-set counts.
- `PerturbationSpec.kinds` may be empty: the tasks are the original queries only.

### 5.2 Loader

- `irbench.datasets.amazon_esci`: reads the parquet files with pyarrow (already a dependency), applies §3.1–§3.3, and writes `docs.jsonl`, `queries.jsonl`, `qrels.tsv`, `queries.prefixes.jsonl`, `qrels.prefixes.tsv`.
- `datasets/amazon-esci/dataset.yaml` lists the four files (three ESCI, the AmazonQAC shard) as one dataset, since the prefix set only exists through the join.
- HTML stripping uses the standard library (`html.parser`, `html.unescape`); no new dependency.

### 5.3 Query text kept as typed

- The task builder and every adapter must pass a query's text unchanged, including a trailing space. A test renders a task whose text ends with a space for each adapter and checks the space reaches the request body.
- Postgres `websearch_to_tsquery` and the Qdrant BM25 tokenizer drop it. That is the engine's behaviour, measured as such, and documented in their READMEs.

### 5.4 Report

- The scenario page already shows a per-group table for grouped scenarios.
- For `ecommerce/as-you-type` it also shows a line chart: nDCG@10 per completion group (`≤25%` … `complete`), one line per engine and base, with `diverges` as a separate point. Every group comes from the same 2,000 queries, so the chart reads as how much of its full-query relevance an engine reaches while the query is being typed. Groups are ordered as in §3.2, not alphabetically: `Scenario.group_order: list[str] | None`.

### 5.5 Lint: scenarios sharing a dataset

Scenario files have no inheritance, so `ecommerce/as-you-type` repeats the fields of `ecommerce/amazon-search`. `irbench lint` gains a check (my call): two scenarios on the same `dataset` must have equal `index_fields`, `embedding_fields` and `id_field`, or lint reports both files. Without it the prefix scores could silently run against a different index layout.

## 6. Engine configs

- All 6 engines get one config per scenario (12 files), with the default track per the existing justification rules.
- The as-you-type config is the amazon-search config. Both scenarios search the same index settings, and lint checks the two are identical apart from the header comment, so the prefix scores are not tuned separately.
- **Tuned track:** the per-scenario sweep used for the other scenarios (search settings within the parity rules, scored on nDCG@10), run on `ecommerce/amazon-search` only. The sweep script is a scratch tool outside the repo, as for the earlier tuned tracks; this change does not commit it. Its results are transcribed into the `tuned:` blocks with numbers in each justification, as `docs/methodology/parity.mdx` describes (my call). The winning settings apply unchanged to `ecommerce/as-you-type`. Tuning on the prefixes would reward settings that suit partial queries at the cost of complete ones, and an engine runs one configuration for both.

## 7. Runs

On the laptop (`local` profile):

1. Prepare and embed: 1.2M products with bge-small (about 4 hours locally, run while nothing else is benchmarking), 2,000 complete queries with their perturbations, 7,429 prefix queries.
2. Sweeps on `ecommerce/amazon-search` for each engine, then the tuned track on both scenarios, sequentially, full pipeline.
3. Report rebuilt with the tuned runs only, as for the other scenarios; the prototype report and the Linear notes updated.

### 7.1 Fit test (before step 2)

The full catalogue is used only if it fits. Fits means, for each of the 6 engines on the `local` profile (7.8 GiB VM, 8 CPUs), with the default track:
- the index builds without an out-of-memory kill or container restart, within `index_timeout_s` (3 hours);
- the relevance stage of `ecommerce/amazon-search` stays under the 0.1% error rate (requests within the 10-second timeout).

Peak container memory (`docker stats`) and build time are recorded per engine. Engine settings that size memory (Elasticsearch heap, Postgres `shared_buffers` and `maintenance_work_mem`) may be raised within the VM, each with a justification in the config that cites the corpus size.

If any engine fails, every engine uses the fallback corpus: a scenario compares engines on one corpus, so the corpus is not chosen per engine. The failure and its numbers go in the scenario's docs page, since "does not index 1.2M products on 8 GB" is itself a result.

Risks:
- index memory at 1.2M products: the raw vectors alone are 1.8 GB (1.2M × 384 floats). Typesense keeps documents and vectors in RAM, Elasticsearch already peaked at 5.2 GB on WANDS (43k products), and pgvector's HNSW build slows sharply once the graph outgrows `maintenance_work_mem`. §7.1 settles it before any run;
- unjudged results: ESCI judges about 16 products per query, all from Amazon's own ranking. Engines returning other relevant products are scored as wrong. judged@10 is reported next to every nDCG, as today, and the scenario page says so.

## 8. Testing

- **Loader** (small fixture parquet files):
  - locale filter; `corpus: full` keeps every US product, `corpus: pool` only the judged products of the pool;
  - HTML stripped from descriptions; missing values null; a product with no title dropped with its judgments;
  - grades E 2, S 1, C 0, I 0; a query with no E or S dropped;
  - the 2,000-query sample is stable for a seed;
  - the QAC join is case- and whitespace-insensitive; the pool keeps only queries with a session; one session and one prefix per group are chosen stably for a seed; prefixes keep their trailing space; group assignment for each bucket, `diverges` and `complete`; a prefix's qrels are its final query's.
- **Scenario:** `query_set` reads the named files; an unknown set fails with the file path; empty `kinds` gives original tasks only.
- **Adapters:** each of the 6 passes a trailing space through (§5.3).
- **Report:** the completion chart renders with groups in `group_order`; scenarios without `group_order` keep today's ordering.
- **Lint:** the 12 configs pass; the as-you-type configs equal the amazon-search ones; two scenarios on one dataset with different `index_fields` are reported.

## 9. Deliverables

- `datasets/amazon-esci/dataset.yaml`; loader `harness/irbench/datasets/amazon_esci.py`.
- Harness: `scenario.py` (`query_set`, `group_order`, empty kinds), `stages/prepare.py` (query sets), report data, render and template (completion chart), lint (config equality).
- Scenarios: `scenarios/ecommerce/{amazon-search,as-you-type}/scenario.yaml`.
- Engine configs: 12 files, default and tuned.
- Docs: one Mintlify page per scenario (sources, licenses, the join, the unjudged-results caveat, what a prefix is scored against).

## 10. Later

- Keystroke curves: the shard already stores every keystroke of each session (`s`, `sh`, `sho`, …). Scoring all of them, not one per group, would give a metric such as "characters typed until the first page has an Exact product".
- The TREC Product Search track (2023, 2024) reuses the ESCI products with deeper NIST judgments on a few hundred queries. It would address the unjudged-results risk of §7 for the complete-query scenario.

## 11. Open questions

None. Corpus size (the full catalogue if it fits, §7.1) and prefix source (one train shard, the same 2,000 queries in both scenarios) were decided on 2026-10-21.
