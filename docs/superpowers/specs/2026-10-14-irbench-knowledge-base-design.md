# IRBench — Knowledge-Base Scenarios (CQADupStack, TREC-COVID, LoTTE)

Date: 2026-10-14
Status: Draft, awaiting review
Parent spec: `2026-10-09-irbench-design.md` (implements the "Later → Datasets/scenarios" item for the knowledge-base use case)

## 1. Goal

IRBench measures one use case today: product search on WANDS. Every finding so far (`matchingStrategy`, ranking-rule order, the vector search width) comes from that one catalog. This change adds three knowledge-base scenarios. They test the engines on help-center, document and Q&A search, where queries are questions or short problem statements and documents are long text.

| Scenario | Dataset | Why |
|---|---|---|
| `knowledge-base/community-qa` | CQADupStack (Webmasters, Android, WordPress) | Help-center and product-support search: short typed question titles against forum posts, with tags and forums as structured fields |
| `knowledge-base/covid-papers` | TREC-COVID + CORD-19 metadata | Document search with dense, graded human judgments and real metadata (date, source) |
| `knowledge-base/web-questions` | LoTTE lifestyle, search queries | Real web-search questions (from Google, via GooAQ) against Q&A answers |

Non-goals:
- synonym perturbations (no domain vocabulary is built);
- multilingual datasets (a later step: MIRACL or Amazon ESCI Japanese and Spanish);
- the OpenAI embedding model on these scenarios (bge-small only, as in local runs today);
- engine tuning beyond the existing Meilisearch `tuned` track parameter.

## 2. Decisions (agreed with the user)

| Topic | Decision |
|---|---|
| Datasets | CQADupStack (replaces FiQA), TREC-COVID, LoTTE lifestyle search |
| CQADupStack forums | Webmasters, Android and WordPress, merged into one index, with `forum` as a field |
| TREC-COVID queries | Both forms of each of the 50 topics (question and keyword) as separate queries, reported per form |
| Perturbations | Typo and prefix; no synonym |
| Sequencing | All three scenarios in one plan |
| Meilisearch tuned track | `matchingStrategy: frequency` on each scenario, as on WANDS |

## 3. Datasets

Each dataset is pinned by URL and sha256 in `datasets/<id>/dataset.yaml`, as WANDS is. Raw data is downloaded, never committed.

### 3.1 `cqadupstack`

- **Source:** the BEIR archive `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip` (5.3 GB, all 12 forums).
- **Files used,** per forum (`webmasters`, `android`, `wordpress`): `corpus.jsonl` and `qrels/test.tsv`, and the query text from `queries.jsonl`.
- **Fetching:** only these members are read, by HTTP range requests against the zip (§5.1). Each member is pinned by the sha256 of its extracted bytes.
- **Docs:**
  - `id`: `<forum>-<post id>` (post ids are only unique within a forum);
  - `title`, `text`;
  - `forum`: one of `webmasters`, `android`, `wordpress`;
  - `tags`: list from `metadata.tags`.
- **Queries:** test queries of the three forums, ids `<forum>-<query id>`, text only. BEIR's `queries.jsonl` metadata embeds whole post bodies (177 MB for Webmasters); it is ignored.
- **Qrels:** BEIR test qrels (binary duplicates), ids prefixed as above. About 1,746 queries.
- **License:** CC BY-SA (StackExchange content, redistributed by BEIR).

### 3.2 `trec-covid`

- **Sources:**
  - BEIR archive `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip` (74 MB): `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`;
  - CORD-19 metadata, round of 2020-07-16: `https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-07-16/metadata.csv` (269 MB).
- **Docs:** 171,332 papers.
  - `id`: the CORD-19 `cord_uid` (BEIR's `_id`);
  - `title`, `text` (the abstract);
  - from `metadata.csv`, joined on `cord_uid`:
    - `publish_year` (number);
    - `publish_date` (number, `YYYYMMDD`; a year-only date becomes `YYYY0101`);
    - `source` (list, split from `source_x` on `;`);
    - `journal` (string, kept in the document but not indexed as a field).
  - When a `cord_uid` has several metadata rows, the first one in file order is used. Missing values are null.
- **Queries:** each of the 50 topics gives two queries:
  - `<topic>-question`: BEIR `text` ("what is the origin of COVID-19"), group `question`;
  - `<topic>-keyword`: `metadata.query` ("coronavirus origin"), group `keyword`.
  Both share the topic's qrels.
- **Qrels:** graded 0/1/2. The two grades of −1 become 0.
- **License:** CC BY-SA 4.0 (BEIR); CORD-19 papers carry their own licenses, listed in `metadata.csv`.

### 3.3 `lotte-lifestyle`

- **Sources:** Hugging Face mirrors, pinned to a commit:
  - passages: `colbertv2/lotte_passages`, `lifestyle/test_collection.jsonl` (109 MB, about 119k answers);
  - queries and answers: `colbertv2/lotte`, `lifestyle/test_qas.search.jsonl` (661 queries).
- **Docs:** `id` (the passage `doc_id`), `text`. The `author` field is dropped.
- **Queries:** `qid`, `query` text. The `url` field gives the forum (cooking, travel, sports); it is kept as the query's group, so scores are also reported per forum.
- **Qrels:** every `answer_pids` entry is relevant with grade 1.
- **License:** Apache-2.0 on the mirror's card; the underlying content is StackExchange, CC BY-SA.

### 3.4 Normalization rules (all three)

- Documents with no text at all (empty title and text) are dropped, together with any judgments pointing to them. The count of dropped documents and judgments is recorded in `prepare.manifest.json`.
- Queries left with no positive judgment are dropped and counted the same way.
- Text is kept as published (no lowercasing or stripping beyond whitespace normalization of queries, as for WANDS).

## 4. Scenarios

All three use `top_k: 100`, `perf.page_sizes: [20, 100]`, the same perf protocol as WANDS, `perturbations: {kinds: [typo, prefix]}` with their own seed, and no synonyms.

### 4.1 `knowledge-base/community-qa`

```yaml
id: knowledge-base/community-qa
dataset: cqadupstack
index_fields:
  searchable: [title, text]
  filterable: [forum, tags]
  sortable: []
  facets: [forum]
modes: [keyword, semantic, hybrid, filtered, faceted]
structured_bases: [keyword, semantic]
embedding_fields: [title, text]
filter_templates:
  - {field: forum, op: eq}
  - {field: tags, op: eq}
sort_templates: []
```

`sorted` is not in the modes: BEIR's copy has no date or score field. `tags` has several thousand values, so it is a filter but not a facet.

### 4.2 `knowledge-base/covid-papers`

```yaml
id: knowledge-base/covid-papers
dataset: trec-covid
index_fields:
  searchable: [title, text]
  filterable: [source, publish_year]
  sortable: [publish_date]
  facets: [source]
  numeric: [publish_year, publish_date]
modes: [keyword, semantic, hybrid, filtered, sorted, faceted]
structured_bases: [keyword, semantic]
embedding_fields: [title, text]
filter_templates:
  - {field: source, op: eq}
  - {field: publish_year, op: gte, values: [2020]}
sort_templates:
  - {field: publish_date, direction: desc}
index_timeout_s: 3600
```

### 4.3 `knowledge-base/web-questions`

```yaml
id: knowledge-base/web-questions
dataset: lotte-lifestyle
index_fields:
  searchable: [text]
  filterable: []
  sortable: []
  facets: []
modes: [keyword, semantic, hybrid]
structured_bases: []
embedding_fields: [text]
filter_templates: []
sort_templates: []
```

## 5. Harness changes

### 5.1 Fetching files inside a zip

`FileSpec` gains an optional `member` (a path inside the archive at `url`). When it is set, `fetch_file` reads only that member:

- it reads the zip's central directory and the member's bytes with HTTP range requests;
- it decompresses the member and writes it to `raw/<dataset>/<member path>`;
- it checks sha256 against the extracted bytes, as for a plain file;
- if the server does not answer a range request with `206 Partial Content`, it fails with an error that names the URL. There is no silent full download of a 5.3 GB archive.

A plain zip (`trec-covid.zip`, 74 MB) is pinned the same way, member by member, so a loader always receives extracted files.

### 5.2 Loaders

`irbench.datasets` gains `beir.py` (reading BEIR's `corpus.jsonl`, `queries.jsonl` and qrels TSV), with three normalizers registered next to `wands`:

- `normalize_cqadupstack` (§3.1), `normalize_trec_covid` (§3.2, including the `metadata.csv` join), `normalize_lotte` (§3.3);
- each one returns `NormalizedDataset` and applies §3.4.

### 5.3 Optional synonyms

- `Scenario.synonyms` becomes optional. When absent, `synonyms_path` is `None` and the run context gives adapters an empty list of term groups (today it always reads a file).
- Each adapter must accept an empty list: no synonym setting, filter or table is created (an empty Elasticsearch synonym filter, for instance, may be rejected). A test covers each adapter with no synonyms.
- `PerturbationSpec.vocabulary` becomes optional and is required only when `synonym` is in `kinds` (validation error otherwise).
- The report's perturbation table shows only the kinds a scenario uses.

### 5.4 Query groups

- A query in `queries.jsonl` may carry `group` (a string). The tasks carry it through: `QueryTask.group: str | None = None`.
- `score` adds relevance rows per group, next to the per-perturbation rows: `{mode, base, perturbation: "all", group, n, metrics}`. Rows without a group keep today's shape (`group` absent or null), so WANDS scores are unchanged.
- The scenario page shows a per-group table for the retrieval bases when the scenario has groups (TREC-COVID: question vs keyword; LoTTE: cooking, travel, sports).

### 5.5 What does not change

- The task builder already handles list fields (`forum`, `tags`, `source`) for `eq` filters and facets, and numeric fields for `gte` and sorting.
- The report already groups runs by scenario; each new scenario gets its own page and appears on the index.
- Adapters read field lists from the scenario; no adapter code is expected to change. Engine configs (§6) carry the per-scenario settings.

## 6. Engine configs

Each of the 5 engines gets one config per scenario: `engines/<engine>/configs/knowledge-base/<scenario>.yaml`.

- **Default track:** the engine's shipped settings plus what the scenario's fields require (searchable order, filterable, sortable, facets), with the same justification rules as WANDS. `irbench lint` must pass.
- **Meilisearch `tuned` track:** `matchingStrategy: frequency` on keyword and hybrid, as on WANDS, with a justification that says it carries the WANDS finding over to question-style queries (the tuned score is what this run measures).
- Engine-level settings that WANDS configs justify by corpus size (pgvector `hnsw.ef_search`, HNSW build parameters) keep the WANDS values unless the larger corpus (171k papers) makes an index build fail or time out; any change is justified in the config.

## 7. Runs

After implementation, on the laptop (`local` profile, not publishable):

1. All five engines, default track, on the three scenarios, full pipeline at both page sizes.
2. Meilisearch tuned track on the three scenarios.
3. WANDS regression: re-score the existing WANDS runs with the new `score` code. WANDS relevance must be identical: the grouping change must not alter rows without groups.
4. Rebuild the local report, then add one Linear document per scenario in the "Information Retrieval Benches" project with the detailed tables. Update the project description to list the new scenarios.

Expected cost:
- embedding about 380k new documents with bge-small, about an hour locally (TREC-COVID dominates);
- indexing the 171k-paper corpus is the memory risk (8 GB VM, Elasticsearch peaked at 5.2 GB on WANDS). If an engine fails to index, the run records the failure; the scenario is not shrunk to fit.

## 8. Testing

- **Zip member fetch:** a local HTTP server fixture serving a small zip with range support. Cases:
  - member extracted and checksum-verified;
  - wrong checksum fails;
  - a server without range support fails with a clear error;
  - an already-present file with the right checksum is not re-fetched.
- **Loaders:** small fixture files for each dataset:
  - CQADupStack: forum id prefixing, tags, query text taken without the bulky metadata;
  - TREC-COVID: metadata join, the first row wins for a repeated `cord_uid`, dates to `publish_year` and `publish_date` (year-only dates), `source` split, −1 grades to 0, the two query forms with groups;
  - LoTTE: `answer_pids` to qrels, group from the URL host;
  - all: an empty document is dropped with its judgments, a query left with no positive is dropped, counts land in the manifest.
- **Scenario:** a scenario without synonyms loads; `synonym` in kinds without `vocabulary` fails validation.
- **Score and report:** per-group rows appear only for grouped scenarios; WANDS-shaped scores are unchanged (golden comparison); the group table renders.
- **Adapters:** each of the 5 adapters renders its index settings with no synonyms (no empty synonym object).
- **Lint:** the 15 new configs pass; the Meilisearch tuned blocks carry a justification.

## 9. Deliverables

- `datasets/{cqadupstack,trec-covid,lotte-lifestyle}/dataset.yaml`.
- Harness:
  - `datasets/fetch.py` and `datasets/registry.py` (zip members);
  - `datasets/beir.py` and the three normalizers;
  - `scenario.py` (optional synonyms and vocabulary);
  - `models.py`, `tasks.py`, `score/relevance.py` (query groups);
  - report data, render and templates (group table, perturbation kinds per scenario).
- Scenarios: `scenarios/knowledge-base/{community-qa,covid-papers,web-questions}/scenario.yaml`.
- Engine configs: 15 files.
- Docs:
  - one page per new dataset and scenario in the Mintlify docs (source, license, fields, what each mode means here, the small-n caveat for TREC-COVID);
  - parent spec "Later" list updated.
- Runs, report and Linear notes per §7.
