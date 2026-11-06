# IRBench — Entity search scenario (DBpedia-Entity v2)

Date: 2026-10-31
Status: Approved 2026-11-01
Parent specs: `2026-10-09-irbench-design.md`, `2026-10-14-irbench-knowledge-base-design.md` (reuses its BEIR loader path and query groups)

## 1. Goal

Three of IRBench's four scenarios send long natural-language questions against long documents (CQADupStack, TREC-COVID, LoTTE). WANDS has short typed queries, but against a product catalogue. No scenario measures the search a user runs to find one named thing (a person, a place, a film, a band) or a short list of them, in a collection where each record is a name and a short description. That is the search box of an encyclopedia, a media or music catalogue, a directory, a CRM or a knowledge graph, and a common use case for site search.

This change adds one scenario built on DBpedia-Entity v2, the standard test collection for entity search:

| Scenario | Queries | What it adds |
|---|---|---|
| `entity-search/dbpedia` | 467 real queries from four earlier benchmark campaigns, from bare names (`glenn frey`, `mario bros`) to type lists (`Airports in Germany`) and questions | Records with a short name field and a short description; queries that are often just a name; graded human judgments, deep (105 judged entities per query on average) |

Non-goals:
- the entity-linking side of DBpedia (query annotations, types, related entities); only the text of each entity is indexed;
- filtered, sorted and faceted modes (§4.2);
- reproducing published BEIR or MTEB scores: the corpus is a sample (§3.3), so absolute numbers differ;
- any other language (DBpedia-Entity v2 is English only).

Anti-goals:
- A query subset chosen because it favours one engine. All 467 queries are kept, the four original groups are reported separately (§3.4), and the headline number is over all of them.
- A sample that changes which engine wins. §3.3 explains why it should not, §3.5 measures it for lexical ranking, and §7 names the bias that could not be measured.

## 2. Decisions

| Topic | Decision | By |
|---|---|---|
| Dataset | DBpedia-Entity v2, the `mteb/dbpedia` copy on Hugging Face (BEIR format) | user |
| Corpus | All judged entities plus uniform random distractors, seed `20261031`, 1,000,000 entities in total; 500,000 if the fit test fails (§3.3, §7.1) | my call |
| Queries | All 467 (BEIR's 400 test and 67 dev queries), QALD2's questions included, with the four original query groups (§3.4); the headline score is over all 467 | user (2026-11-01) |
| Searchable fields | `title` (the entity name), then `text` (the abstract) | my call |
| Modes | keyword, semantic, hybrid; filtered, sorted and faceted not run (§4.2) | my call |
| Grades | As published: 0, 1 (relevant), 2 (highly relevant); grade 0 judgments stay in the qrels as judged non-relevant | my call |
| Perturbations | `typo` and `prefix` | my call |
| Document id | A hash of the DBpedia URI; the URI is kept as a stored field (§3.4) | my call |

## 3. Data

### 3.1 Source

- **Files,** from `huggingface.co/datasets/mteb/dbpedia`, pinned to revision `7d8780d8ef0d0a55d13fa797fe39224ec806592c`:

  | File | Size | sha256 |
  |---|---|---|
  | `corpus.jsonl` | 1,792,050,525 bytes | `218fd9274382857e1424ae3f3ad664424636766df2673ec0caf75830c60fb1ad` |
  | `queries.jsonl` | 32,978 bytes | `7573758032694ac960d59b3c59746cf7ff49a01dfc32f12a47af87181640482f` |
  | `qrels/test.tsv` | 2,110,129 bytes | `7939c61fa8f0f99c43bd04e2c118bd3d97ec7ee4d67d05c63d14fde103bedc96` |
  | `qrels/dev.tsv` | 284,736 bytes | `316a5c4bab1356cd49dca79175e15f0bbd16cb028063371fe08a69a136505a73` |

  These are plain files on Hugging Face (as for LoTTE), so no zip-member fetch is needed.
- **License:** MIT on the Hugging Face card and on the upstream repository (`github.com/iai-group/DBpedia-Entity`). The entity text is from DBpedia 2015-10, itself drawn from Wikipedia (CC BY-SA 3.0). The docs page states both.
- **Reference:** Hasibi et al., *DBpedia-Entity v2: A Test Collection for Entity Search*, SIGIR 2017.

### 3.2 What the data looks like (measured 2026-10-31 on the pinned files)

**Corpus.** 4,635,922 entities (DBpedia 2015-10), each with `_id` (the DBpedia URI), `title` (the entity name) and `text` (the abstract). None has an empty name or abstract. Names are short: median 2 words, 90th percentile 5. Abstracts are short too, cut to about the first paragraph: median 47 words, 90th percentile 76, 99th percentile 88.

```json
{"_id": "<dbpedia:Animalia_(book)>", "title": "Animalia (book)", "text": "Animalia is an illustrated children's book by Graeme Base. It was originally published in 1986, ..."}
```

**Queries.** 467, in seven sources that the DBpedia-Entity v2 paper merges into four groups:

| Group (paper) | Sources (`_id` prefix) | Queries | Median words | Judged per query (mean) | Relevant per query (median) | Examples |
|---|---|---|---|---|---|---|
| SemSearch ES | `SemSearch_ES` | 113 | 2 | 66 | 9 | `glenn frey`, `bourbonnais il`, `mr rourke fantasy island`, `emery` |
| INEX-LD | `INEX_LD` | 99 | 4 | 104 | 17 | `vietnam war movie`, `bicycle benefits health`, `guitar chord minor` |
| ListSearch | `INEX_XER`, `SemSearch_LS`, `TREC_Entity` | 115 | 6 | 96 | 28 | `Airports in Germany`, `Neil Gaiman novels`, `provinces and territories of Canada` |
| QALD2 | `QALD2_tr`, `QALD2_te` | 140 | 8 | 146 | 12 | `Who designed the Brooklyn Bridge?`, `Which instruments did John Lennon play?` |

(Per-group judged and relevant counts are over both splits.) Query length over all 467: median 5 words; 68 queries have two words, 5 have one.

**Judgments.** 49,188 judgments (43,515 test, 5,673 dev), grades 0: 32,497, 1: 9,809, 2: 6,882. Every query has at least one relevant entity; 62 have no grade-2 entity. Judged per query: mean 105, median 88, range 21–1,499. 45,595 distinct entities are judged, all of them present in the corpus. Two thirds of the judgments are grade 0: the original pooling kept the near misses, which matters for §3.3.

What this shows for the case made in §1: SemSearch ES is the entity-name use case in its plainest form (median 2 words, mostly a name). INEX-LD and ListSearch are short keyword queries. QALD2 is question-style, like the existing knowledge-base scenarios; it is kept so the collection is used as published (anti-goal, §1), and its group row shows it apart.

### 3.3 Corpus sample

The full corpus does not fit: 4.64M entities would need 6.7 GB for the bge-small vectors alone (4.64M × 384 × 4 bytes), on a 7.8 GiB VM, and about 12 hours of local embedding.

**Rule.** The corpus is:
1. every entity judged (any grade, either split) for any of the 467 queries: 45,595 entities;
2. plus distractors drawn uniformly at random, without replacement, from the other entities, with `random.Random(20261031)`, until the corpus holds 1,000,000 entities.

The draw, exactly: number the non-judged entities 0 … n−1 in corpus file order; keep those whose number is in `random.Random(20261031).sample(range(n), corpus_size − n_judged)`. It depends only on the pinned file, the seed and the size. Documents are written in corpus file order, not draw order.

- The seed is a loader constant (`SEED`), like ESCI's query-sample seed.
- The size is a loader parameter in `dataset.yaml` (`params: {corpus_size: 1000000}`), like ESCI's `corpus: full | pool`, so the fallback of §7 changes one line and both sizes share one code path.
- The sample is part of the dataset's definition, not a run setting: `prepare.manifest.json` records the size, the seed and the number of judged and distractor entities.

**Why 1,000,000.**
- It is 22 times the judged pool, a fifth of the full corpus. In §3.5, BM25 at 1M scores nDCG@10 0.307, between the full corpus (0.274) and the judged pool alone (0.334), and judged@10 rises from 0.61 to 0.77. Between 2M and 250,000, each halving of the size adds 0.007 to 0.013.
- 500,000 is almost as faithful (0.317, judged@10 0.83), which is why it is the fallback. Below 250,000 the curve flattens towards the pool-only score: the sample stops behaving like a large collection.
- Cost: 1.5 GB of raw vectors and about 0.4 GB of text, below ESCI's 1.2M products on both counts; 2–3 hours of local embedding.

**Why scores on the sample still mean something.**
- *Every judged entity stays.* The judgments came from pooling the top results of many 2017 systems. Grade-0 entities, 66% of the judgments, are the near misses those systems ranked high: the entities that compete hardest with the right answers. They are all in the sample, so the hard part of each query is intact.
- *Distractors are uniform.* They are not picked by BM25 or by an embedding model, so no engine or retrieval mode gets a sample built around its own ranking. A query's competition from unjudged entities shrinks by the same factor for every engine.
- *Engines are compared with each other, on one corpus.* Scores are higher than on the full corpus, since there are fewer unjudged entities to rank above the right answers. What IRBench reports is the comparison between engines on the same data. §3.5 shows that, for lexical ranking, which of two rankings wins survives the sampling, while the gap between them widens.
- *Unjudged results shrink.* On the full corpus, an engine's top 10 often holds entities no one judged, scored as wrong. The sample drops most of them, so judged@10 goes up and the scores rest on human judgments more than on the corpus size.

What the sample cannot show: how an engine copes with millions of near-duplicate names (four people called `John Smith`). The docs page says so.

### 3.4 Loader

`irbench.datasets.dbpedia_entity`, registered as `dbpedia-entity`, reads the files with the BEIR helpers (`beir.read_jsonl_rows`, `beir.read_qrels_tsv`) and writes through `beir.write_normalized`, so §3.4 of the knowledge-base spec applies (empty documents dropped with their judgments, counts in `prepare.manifest.json`).

- **Docs:**
  - `id`: `dbp-` followed by the first 16 hex characters of the sha1 of the DBpedia URI (`<dbpedia:Animalia_(book)>`). URIs contain `<`, `:`, `>`, parentheses, commas and non-ASCII letters, which Meilisearch rejects in a primary key. The loader fails if two URIs collide;
  - `title`: the entity name, as published (`Animalia (book)`, disambiguation in parentheses included);
  - `text`: the abstract, as published;
  - `uri`: the DBpedia URI, kept in the document but not indexed as a field (as `journal` on TREC-COVID), so a result can be traced back.
- **Queries:** all 467 from `queries.jsonl`; `qid` the original `_id` (`SemSearch_ES-12`), text with whitespace normalised; `group` from the `_id` prefix per §3.2 (`SemSearch ES`, `INEX-LD`, `ListSearch`, `QALD2`).
- **Qrels:** `test.tsv` and `dev.tsv` merged (no query is in both), corpus ids mapped to the hashed `id`. Every judged entity is in the pinned corpus (§3.2), so no judgment is dropped; the loader fails if one is missing, rather than letting `build_tasks` silently skip a query left with no relevant entity.
- **Corpus:** sampled per §3.3. The loader reads `corpus.jsonl` twice, streaming: once to count the non-judged entities (n), once to write the kept documents. It never holds the 1.8 GB file in memory.

`datasets/dbpedia-entity/dataset.yaml` lists the four files with the URLs and hashes of §3.1.

### 3.5 Does the sample change who wins? (measured 2026-10-31)

Before writing this spec, BM25 (k1 1.2, b 0.75, lower-cased words, a short stop-word list, no stemming) was run over the full corpus and over samples built by the rule of §3.3 at several sizes, on all 467 queries, with two field choices standing in for two engines: name and abstract together (`name+abstract`), and name only (`name`). The script is a scratch tool outside the repo, and it used a different random draw than §3.3's (same rule, numpy's generator), so the table is evidence about the rule, not about the exact shipped set. The docs page shows it with that label.

nDCG@10 over the 467 queries, and judged@10:

| Corpus | Entities | `name+abstract` | `name` | Gap | judged@10 (`name+abstract`) |
|---|---|---|---|---|---|
| Full | 4,635,922 | 0.274 | 0.259 | +0.015 | 0.61 |
| Sample | 2,000,000 | 0.294 | 0.272 | +0.022 | 0.70 |
| **Sample** | **1,000,000** | **0.307** | **0.280** | **+0.027** | **0.77** |
| Sample | 500,000 | 0.317 | 0.290 | +0.027 | 0.83 |
| Sample | 250,000 | 0.324 | 0.293 | +0.031 | 0.87 |
| Sample | 100,000 | 0.329 | 0.299 | +0.030 | 0.90 |
| Judged pool only | 45,595 | 0.334 | 0.303 | +0.031 | 0.92 |

Per group, full corpus → 1M sample:

| Group | `name+abstract` | `name` |
|---|---|---|
| SemSearch ES | 0.496 → 0.519 | 0.567 → 0.590 |
| INEX-LD | 0.239 → 0.275 | 0.209 → 0.235 |
| ListSearch | 0.185 → 0.221 | 0.130 → 0.146 |
| QALD2 | 0.194 → 0.229 | 0.152 → 0.170 |

What it shows:
- **The order holds.** At every size, overall and in every group, the same field choice wins: `name+abstract` overall and on INEX-LD, ListSearch and QALD2; `name` on SemSearch ES (bare names, where a passing mention in an abstract is noise).
- **Per query, the sample tracks the full corpus.** Spearman correlation of per-query nDCG@10, full versus 1M: 0.97 (`name+abstract`), 0.98 (`name`). Of the 399 queries where the two field choices score differently on either corpus, 352 (88%) have the same winner on both corpora.
- **Scores rise by about a tenth** (0.274 → 0.307), as expected with fewer unjudged entities to rank above the answers.
- **Gaps widen, and can become significant.** The gap between the two field choices goes from +0.015 to +0.027 at 1M. On the full corpus it is not significant (paired Wilcoxon p = 0.17); at 1M it is (p = 0.01). The sample removes unjudged distractors that hurt both choices, so differences in how each ranks the judged entities stand out more. A difference between engines on this scenario is therefore a real difference in ranking the same candidates, but may look larger, and more significant, than on the full 4.64M. The docs page says so next to the significance markers.

The experiment is lexical only. Embedding the full 4.64M corpus to repeat it for semantic search is the same 12-hour cost that rules out the full corpus. §7 lists this as a risk.

## 4. Scenario

### 4.1 `entity-search/dbpedia`

```yaml
id: entity-search/dbpedia
dataset: dbpedia-entity
id_field: id
index_fields:
  searchable: [title, text]
  filterable: []
  sortable: []
  facets: []
  numeric: []
modes: [keyword, semantic, hybrid]
structured_bases: []
embedding_fields: [title, text]
filter_templates: []
sort_templates: []
top_k: 100
index_timeout_s: 7200
perturbations:
  seed: 20261031
  kinds: [typo, prefix]
perf:
  page_sizes: [20, 100]
```

The other keys (`max_facet_values`, `request_timeout_s`, `max_error_rate`, the perf protocol) are copied from `knowledge-base/web-questions`, which also has three modes.

- **Field order.** `title` first: on this collection the name is the strongest signal, and an entity search box that ranked a passing mention in an abstract above the entity itself would be wrong. §3.5 supports both halves: on SemSearch ES (bare names) the name alone beats name and abstract together, and on the other three groups adding the abstract helps, so it stays searchable, second.
- **Embedding text.** `title` and `text` joined with ". ", as on the knowledge-base scenarios. Abstracts are short (99th percentile 88 words, §3.2), so bge-small's 512-token limit truncates almost none of them.
- **Perturbations.** `typo` fits this use case closely: people misspell names (`glen frey`). `prefix` stands for a name not typed to the end. No `synonym`: no vocabulary is built.
- **`index_timeout_s`.** 2 hours, between TREC-COVID (1 hour for 171k long abstracts) and ESCI (3 hours for 1.2M products); the builder adjusts it from measured times, with the measurement in a comment.
- **Name.** A new category, `entity-search/`, since this is neither product nor knowledge-base search. The docs page is `docs/scenarios/entity-search.mdx`.

### 4.2 Filtered, sorted and faceted modes: not run

The BEIR copy has only a name and an abstract per entity: no type, date, category or number. Inventing a field (abstract length, the first letter of the name) would test nothing a user does.

Entity types would make these modes meaningful: filtering on `Person` or `Film` and faceting by type is a real entity-search feature. They exist in DBpedia 2015-10 (`instance_types_en`), joinable on the URI. That join is §10, not this change: it adds a second source, an ontology to flatten, and filter values that need choosing per query.

So the scenario lists only the three retrieval modes, as `knowledge-base/web-questions` does. The modes are absent from the scenario, not marked N/A per engine, since no engine could run them.

## 5. Harness changes

None expected beyond the loader. Query groups, BEIR readers, scenarios without filters or facets, and HF-pinned files all exist (knowledge-base spec §5).

- Query groups show in the report's default order (alphabetical): INEX-LD, ListSearch, QALD2, SemSearch ES. `Scenario.group_order` (ESCI spec §5.4) has not landed; this change does not add it (my call). If it lands first, the scenario may set `[SemSearch ES, INEX-LD, ListSearch, QALD2]`, shortest queries first.
- The fetch cache holds a 1.8 GB raw file. Nothing new: ESCI's AmazonQAC shard is 2.96 GB.

## 6. Engine configs

- All 6 engines get `engines/<engine>/configs/entity-search/dbpedia.yaml`, default and tuned, with the justification rules of the parity doc. Field priority maps as on the knowledge-base scenarios (two fields: Postgres `setweight` A and B, ParadeDB boosts, and so on).
- Settings carried over from the knowledge-base configs keep their values and justifications (for example pgvector `hnsw.ef_search: 100`, justified by `top_k`).
- No memory setting is changed today (Elasticsearch runs with no heap override; Postgres `maintenance_work_mem` is not set). If the fit test (§7) shows a 1M-entity build needs one (Elasticsearch heap, Postgres `maintenance_work_mem` for the HNSW build), it is a new departure from the shipped engine, with a justification that cites the corpus size and the measured failure, as ESCI §7.1 allows.
- **Tuned track:** the per-scenario sweep used for the other scenarios, scored on nDCG@10 over the 467 queries, per the parity doc. The sweep script stays a scratch tool outside the repo; its results go into the `tuned:` blocks with numbers in each justification.

## 7. Runs

On the laptop (`local` profile):

1. Prepare: stream the corpus, sample, write 1M documents. Embed 1M entities with bge-small (about 2–3 hours locally, by ESCI's estimate of 4 hours for 1.2M longer products) and 467 queries with their perturbations.
2. Fit test (§7.1).
3. Default track for the 6 engines, then the sweeps, then the tuned track, sequentially.
4. Report rebuilt; scenario docs page written.

### 7.1 Fit test (before step 3)

This scenario runs its own fit test, with ESCI §7.1's criteria; it does not wait for ESCI's, which has not run. For each of the 6 engines on the `local` profile, default track:
- the index builds without an out-of-memory kill or container restart, within `index_timeout_s`;
- the relevance stage stays under the 0.1% error rate.

Peak container memory (`docker stats`) and build time are recorded per engine. If any one engine fails, every engine uses the fallback size, 500,000 (`corpus_size` in `dataset.yaml`), after re-preparing and re-embedding (about an hour). A scenario compares engines on one corpus, so the size is never chosen per engine. The failure and its numbers go on the docs page.

Risks:
- **Index memory at 1M entities.** The vectors are 1.5 GB raw, the text about 0.4 GB: less than ESCI's 1.2M products on both counts, but no engine has indexed a million documents in IRBench yet. §7.1 settles it before any scored run.
- **The sample may favour semantic search more than keyword search.** The pool was built from 2017 systems, mostly lexical, so the entities a dense model ranks high but no one judged are more often dropped by sampling than those BM25 ranks high. This would raise semantic scores relative to keyword scores on the same engine. It does not change the comparison between engines within a mode, since every engine searches the same vectors and the same text. The docs page warns against reading the keyword-versus-semantic gap on this scenario as it would be on the full corpus.
- **Unjudged results.** judged@10 is shown next to every nDCG, as on the other scenarios.

## 8. Testing

Loader, on small fixture files (a 30-entity corpus, a few queries in each group):
- every judged entity is kept, whatever its grade; the total equals the configured size; the sample is identical for a seed and changes with it; documents are in file order;
- ids: the hash is stable, a collision fails, the `uri` field holds the original URI, qrels use the hashed ids;
- groups from `_id` prefixes, including the merged ListSearch and QALD2 groups; an unknown prefix fails;
- test and dev qrels merged; a judged entity missing from the corpus fails the loader; an empty document is dropped with its judgments (shared rule).

Scenario and configs: the scenario validates; the 6 configs pass `irbench lint`.

## 9. Deliverables

- `datasets/dbpedia-entity/dataset.yaml`; loader `harness/irbench/datasets/dbpedia_entity.py`, registered in `harness/irbench/datasets/__init__.py`; tests.
- `scenarios/entity-search/dbpedia/scenario.yaml`.
- Engine configs: 6 files, default and tuned.
- Docs: `docs/scenarios/entity-search.mdx` (source, licenses, the sampling rule and why it holds, the §3.5 table, the query groups, the semantic-versus-keyword caveat, the unjudged-results caveat), added to `docs/docs.json`.

## 10. Later

- Entity types from DBpedia 2015-10 `instance_types_en`, joined on the URI, as a filterable and facet field. Would enable the filtered and faceted modes (§4.2).
- A larger corpus on the reference hardware (AWS profile, 16 GB engine container), up to the full 4.64M, to measure the effect of corpus size directly.
- Repeat §3.5 for semantic search once the full corpus is embedded somewhere it is cheap to do.

## 11. Open questions

None from the user. Decided on 2026-11-01: QALD2's 140 question-style queries stay in the scenario and count in the headline score, so the collection is used as published; the group table shows SemSearch ES and QALD2 apart.

Calls made in this spec, to check while reading:
- 1,000,000 entities, 500,000 if any engine fails the fit test; uniform distractors, seed `20261031` (§3.3, §7.1);
- the sample makes gaps between engines look larger and more significant than on the full corpus; the docs page says so (§3.5);
- `title` before `text`; both embedded (§4.1);
- keyword, semantic and hybrid only; entity types for filters and facets left for later (§4.2);
- test and dev merged, in the paper's four groups (§3.4);
- hashed ids with the URI kept as a stored field (§3.4);
- `typo` and `prefix` perturbations; perf page sizes 20 and 100, as on the other scenarios;
- a new `entity-search/` category; this scenario runs its own fit test rather than waiting for ESCI's (§4.1, §7.1).
