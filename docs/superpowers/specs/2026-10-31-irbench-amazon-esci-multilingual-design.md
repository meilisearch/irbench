# IRBench — Japanese and Spanish product search (ESCI jp and es)

Date: 2026-10-31
Status: Approved 2026-10-31
Parent specs: `2026-10-09-irbench-design.md`, `2026-10-20-irbench-amazon-esci-design.md` (builds on its loader, grades and scenario layout)

## 1. Goal

Every scenario today is in English, and every adapter is set up for English: Postgres indexes with its `english` configuration, Typesense with locale `en`, Elasticsearch with its standard tokenizer. The benchmark says nothing about a store that sells in another language, which is a common case for site and product search.

This change adds two scenarios built on the Japanese and Spanish parts of Amazon ESCI:

| Scenario | Queries | What it adds |
|---|---|---|
| `ecommerce/amazon-search-ja` | Real Amazon.co.jp search queries | A language written without spaces between words: 51% of the queries contain a run of 6 or more Japanese characters with no space. An engine has to split that text into words before it can match anything. |
| `ecommerce/amazon-search-es` | Real Amazon.es search queries | A language with accents and inflection, close to English in how it is split into words. Shows what language settings (stemming, stop words, accent folding) do on a second Latin-script language. |

The two scenarios also measure how much setup each engine needs to handle a language. The default track runs every engine with only its built-in language setting; the tuned track also allows the vendor's official language plugin (§2, §6). The results page shows both tracks, so it shows both the setup cost and how close each engine gets once set up.

### Non-goals

- As-you-type: AmazonQAC, the source of real typed prefixes, has US sessions only.
- Cross-language search (a Japanese query finding an English product). Each scenario searches its own locale's catalogue.
- Any other language. A third language would follow the same design.
- Changing the English scenarios. §5.2 adds a `language` field defaulting to `en`, and the English runs must render byte-identical requests.

### Anti-goals

- A scenario that Meilisearch wins only because the others are handicapped. Every engine gets its documented language setting in the default track, and the official plugin in the tuned track. A reader who knows Elasticsearch should not be able to say "nobody runs Japanese without kuromoji" about the tuned results.

## 2. Decisions

| Topic | Decision | By |
|---|---|---|
| Locales | Japanese (`jp`) and Spanish (`es`), one scenario each | user |
| Language plugins | Default track: each engine as shipped, with only its built-in language setting. Tuned track: the vendor's official language plugin is also allowed (Elasticsearch `analysis-kuromoji`, Postgres PGroonga) | user |
| Embedding model | `intfloat/multilingual-e5-small` (384 dimensions, MIT), model id `multilingual-e5-small`, for both scenarios. bge-small-en reads English only. Same vector size, so index memory stays comparable with the English runs | my call |
| Grades | As the US scenario: Exact 2, Substitute 1, Complement 0, Irrelevant 0 | my call (inherited) |
| Query sample | 2,000 queries per locale from Amazon's hard test sample, same seed as the US scenario | my call |
| Corpus | The full catalogue of each locale (339,059 jp products, 259,973 es) | my call |
| Perturbations | Spanish: `typo`, `prefix` and a new `accents` kind (§3.3). Japanese: `prefix` only | my call |
| Scenario language | A new `language` field on the scenario (`en`, `ja`, `es`), read by every adapter | my call |

## 3. Data

### 3.1 Source

The same ESCI files and commit as the US scenario (`datasets/amazon-esci/dataset.yaml`): `shopping_queries_dataset_examples.parquet` and `shopping_queries_dataset_products.parquet`. AmazonQAC and `sources.csv` are not used: the query `group` used on the US scenario comes from `sources.csv`, which covers all locales, so it is kept (my call; drop it if `sources.csv` turns out to be US-only).

License: Apache-2.0.

### 3.2 Counts (measured 2026-10-31 on the pinned examples file)

| | jp | es | us (for reference) |
|---|---|---|---|
| Queries, all splits | 18,127 | 15,180 | 97,345 |
| Judgments | 446,053 | 356,410 | 1,818,825 |
| Products judged | 339,059 | 259,973 | 1,215,851 |
| Hard test queries (`small_version == 1`, `split == "test"`) | 3,123 | 2,417 | 8,956 |
| Judgments per hard test query | 28.4 | 27.3 | 20.3 |
| Median query length | 2 words, 9 characters | 3 words, 20 characters | 3 words, 20 characters |

Every hard test query has at least one Exact or Substitute product in all three locales.

Query examples:
- jp: `空調服 ズボン`, `カシオソーラー電波腕時計 メンズ`, `楽しくない仕事はなぜ楽しくないのか`, `azul by moussy レディース`, `satoshi kon`
- es: `rinconera ducha sin taladro`, `filtro cafetera delonghi`, `calefactor eléctrico inteligente`, `telefonos moviles para mayores con tapa`

Japanese queries mix scripts: 93% contain kana or kanji, 7% are Latin only (brands, titles), and 57% contain a space.

### 3.3 Loader

The existing `irbench.datasets.amazon_esci` loader gains a `locale` parameter (`us`, `es`, `jp`) in `dataset.yaml`. Two new dataset folders use it:

- `datasets/amazon-esci-jp/dataset.yaml`: `locale: jp`, `corpus: full`, the two ESCI parquet files, no AmazonQAC shard, no prefix query set.
- `datasets/amazon-esci-es/dataset.yaml`: the same with `locale: es`.

Everything else follows §3.1 and §3.3 of the US spec:
- documents: `id`, `title`, `brand`, `color`, `bullet_points`, `description` (HTML stripped); a product with no title is dropped with its judgments;
- queries: a seeded sample of 2,000 from the hard test queries (seed `20261020`); id `esci-<query_id>`; whitespace normalised;
- qrels: grades per §2.

Text is kept as Amazon wrote it: no Unicode normalisation beyond what the US loader does, no width folding of full-width Latin letters (`ＡＢＣ`), no kana conversion. How an engine handles those is part of what is measured.

The products parquet holds all three locales. If the fetch cache keys raw files by dataset id, the three datasets would download it three times; the builder should share one raw copy (by URL and sha256) if the fetch layer allows it without a redesign, and otherwise accept the duplicate download.

### 3.4 New perturbation: `accents` (Spanish)

Shoppers often type Spanish without accents: `electrico` for `eléctrico`, `telefonos` for `teléfonos`, `nino` for `niño`. The `accents` kind removes every diacritic from the query (Unicode NFD, drop combining marks, NFC; `ñ` becomes `n`, `ü` becomes `u`). A query with no diacritic gets no variant, as `synonym` does today for queries with no synonym.

10.7% of the hard Spanish test queries contain a diacritic (6.1% an `ñ`), so the variant covers about 210 of the 2,000 queries. The scenario page states that n.

`accents` is a perturbation kind available to any scenario, listed in `perturbations.kinds`. The English scenarios do not list it.

### 3.5 Perturbations left out of Japanese

- `typo` swaps letters for their QWERTY neighbours. Japanese is typed in romaji and converted by an input method, so real mistakes are wrong conversions (the right sound, the wrong kanji), not neighbouring keys. Python's `str.isalpha()` is true for kana and kanji, so the current generator would edit them and produce strings no one types. It is off for `ja`.
- `prefix` stays: cutting the last whitespace-separated piece of a query to 40–70% of its characters is close to what an input method sends before the user finishes. The scenario page notes that for unsegmented text the "last word" is often a whole phrase.

## 4. Scenarios

Both scenarios use the US scenario's layout, with their own dataset and language.

```yaml
id: ecommerce/amazon-search-ja
dataset: amazon-esci-jp
language: ja
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
perturbations:
  seed: 20261020
  kinds: [prefix]
perf:
  page_sizes: [20, 100]
```

`ecommerce/amazon-search-es` is the same with `dataset: amazon-esci-es`, `language: es` and `kinds: [typo, prefix, accents]`.

- `index_timeout_s`: the builder sets it from measured indexing times; both catalogues are a fraction of the US one.
- The filtered-mode filter is chosen as on the other scenarios (`harness/irbench/tasks.py`). It is not changed here, even though it can pick a filter value from a Substitute product; that is a separate issue across all scenarios.

## 5. Harness changes

### 5.1 Embedding model

- Register `multilingual-e5-small` in `harness/irbench/embed/models.py`, with `hf_name` `intfloat/multilingual-e5-small` and dimension 384.
- E5 expects `query: ` before queries and `passage: ` before documents. `SentenceTransformerEmbedder` has a `query_prefix` only; add a `doc_prefix` (default empty, so bge-small vectors are unchanged).
- Runs carry the model in their id already (`<scenario>__<engine>__<model>__<track>__<hardware>`). The report must show these scenarios with their model and must not assume every scenario uses bge-small.

### 5.2 Scenario language and where the engine settings live

- `Scenario.language: Literal["en", "ja", "es"] = "en"`. The harness uses it for scenario validation (§3.5, §5.3) and the report label. Adapters do not switch on it.
- The engine-side language settings of §6.1 live in each engine config's `default` block, as `index` or `search` keys, like Meilisearch's `rankingRules` today (my call). That way:
  - each one is a config key, so the existing `unjustified-setting` lint rule requires its `justification:` entry;
  - the report's *Engine configurations* page and `diff_vs_default` show it with no new data source;
  - the tuned block replaces it by the existing deep-merge (kuromoji's analyzer over `cjk`), so there is no precedence rule to invent.
- Some of today's English choices are hard-coded in adapters, not config keys: Postgres's `english` text search configuration, Typesense's stop-word locale `en`, Elasticsearch's analyzer built by the adapter. Each adapter gains a config key that overrides its hard-coded value (for example `index.text_search_config` for Postgres, `index.analyzer` for Elasticsearch; the builder names them). When the key is absent the adapter renders exactly what it renders today.
- Lint gains one check (my call): a scenario whose `language` is not `en` must have a config per engine whose `default` block sets that engine's language key, so a config cannot silently fall back to English.
- A test renders one request per mode for every English scenario and checks it is identical to the request rendered before the change, so this changes nothing for English.

### 5.3 Embedding model pinned per scenario

- `Scenario.embedding_model: str | None = None`. When set, `embed` and `run` fail with a clear message if `--model` differs (my call). The two new scenarios set `multilingual-e5-small`; the English scenarios leave it unset and behave as today.
- `text-embedding-3-small` is not run on these scenarios: no published run uses it today, and it costs money per call.

### 5.4 Engine image per track

The tuned track may need a different engine image (Elasticsearch with kuromoji installed, Postgres with PGroonga). Today one `engines/<engine>/docker/compose.yaml` serves every track.

- `TrackConfig.compose: str | None`: the name of another compose file in `engines/<engine>/docker/`, used for that track instead of `compose.yaml`. It is set only in the tuned block of a scenario that needs it.
- Like any departure from the shipped engine, it needs a `justification.compose` entry (the `unjustified-setting` lint rule).
- The file may use `build:` with a Dockerfile next to it. The `image-pin` lint rule extends to that Dockerfile: its `FROM` must pin a version, and the plugin version must be tied to it (kuromoji ships with Elasticsearch and must match its version exactly).
- The run manifest records the image actually run, as it does today.

### 5.5 Report

- Scenario pages for the two new scenarios state the language, the model, and what each engine was given for that language on each track (read from the configs, §5.2).
- Publishing (my call, following the user's choice that the results show both tracks): default-track runs of these two scenarios are published to `results/`. Default runs of the English scenarios stay unpublished, as today. `docs/methodology/parity.mdx` says so.
- Default runs appear in one place only: a table on each of the two scenario pages, keyword, semantic and hybrid nDCG@10 per engine, default next to tuned, with the change. They are left out of:
  - each column's best-score marking;
  - the paired significance tests;
  - the *Engine configurations* page's tuned listing, the trade-off charts and every cross-scenario view.
  The report filters them by track before any of these are computed.

## 6. Engine configs

Twelve config files (6 engines × 2 scenarios), default and tuned.

### 6.1 Default track: built-in language setting only

The rule: each engine gets the scenario's language through the language setting it documents and ships, if it has one; nothing that has to be installed. If the engine has no built-in setting for the language, it runs as it would on any text, and the README says so.

| Engine | Japanese | Spanish |
|---|---|---|
| Meilisearch | `localizedAttributes` with locale `jpn` on all searchable fields, and `locales: ["jpn"]` on search requests (stops kanji-only text being read as Chinese) | `localizedAttributes` with `spa` |
| Elasticsearch | built-in `cjk` analyzer (splits Japanese into overlapping two-character pieces, not words) | built-in `spanish` analyzer (stemming and stop words) |
| Typesense | field `locale: "ja"` | field `locale: "es"` |
| Qdrant | the BM25 inference options for the language, as documented for v1.19.2 (the builder checks which of `language`, a multilingual tokenizer and ASCII folding exist) | the same with Spanish |
| ParadeDB | pg_search's built-in Japanese tokenizer (Lindera) | pg_search's Spanish stemmer |
| Postgres | `simple` text search configuration: Postgres has no Japanese configuration, so a Japanese phrase with no spaces stays one token | built-in `spanish` configuration |

Notes:
- The English scenarios do not give Elasticsearch its `english` analyzer today (it runs the standard tokenizer). Spanish gets the `spanish` analyzer because that is the documented way to tell Elasticsearch the text is Spanish. The inconsistency is recorded in the Elasticsearch README; changing the English scenarios is out of scope.
- Each setting in this table is a config key in the engine's `default` block (§5.2), a departure from the shipped config, and carries a `justification:` entry that cites this rule.

### 6.2 Tuned track: official plugins allowed

The tuned track is found as before: a sweep of each engine's documented search settings, scored on nDCG@10 for each scenario and base, keeping the best (parity doc). For these scenarios the sweep may also use the vendor's official language plugin:

| Engine | Japanese | Spanish |
|---|---|---|
| Elasticsearch | `analysis-kuromoji` (official Elastic plugin, installed with `elasticsearch-plugin install`, version-locked to 9.5.5); the sweep tries kuromoji's analyzer and its search mode | no plugin; the sweep may add built-in filters such as `asciifolding` |
| Postgres | PGroonga, if an image with both PGroonga and pgvector can be built from official packages | no plugin |
| Meilisearch, Typesense, Qdrant, ParadeDB | no plugin exists or is needed; usual sweep | usual sweep |

- "Official" means published by the engine's vendor or by the extension's maintainers as a supported package. Community forks and hand-built tokenizers are not allowed.
- The plugin is tried, not imposed (my call): the sweep runs the plugin variants next to the built-in ones and keeps the best on nDCG@10, as for every other setting. If `cjk` beats kuromoji, the tuned track keeps `cjk`, and the scenario page gives kuromoji's best score next to it. The anti-goal is met by kuromoji having been tried and reported, not by forcing it.
- Analyzer and tokenizer variants change the index, so each variant re-indexes the catalogue (339k or 260k products). The sweep for these keys is index-time, unlike the search-settings sweeps of the English scenarios; the builder keeps the variant list short (for kuromoji: its default analyzer and its `search` mode).
- If no image with PGroonga and pgvector can be built from official packages, Postgres's tuned Japanese track keeps the `simple` configuration, and the scenario page says why. The builder does not write a custom build of either extension.
- Each plugin is a `compose` override (§5.4) plus the settings that use it, all justified.

## 7. Runs

This change is built after the US ESCI plan (`docs/superpowers/plans/2026-10-21-irbench-amazon-esci.md`) has landed. On this branch only its Task 1 (the loader) is done. This plan assumes the rest exists: `scenarios/ecommerce/amazon-search`, its engine configs (copied and adapted here), the `scenario.py` changes, and the US sweep approach (my call). If the US plan is dropped, this plan has to take over its scenario and lint changes first.

On the laptop (`local` profile), after the US ESCI runs:

1. Prepare both datasets; embed 339k + 260k products and 2 × 2,000 queries (with their variants) with multilingual-e5-small.
2. Default track for all 6 engines on both scenarios.
3. Sweeps, then the tuned track, sequentially.
4. Report rebuilt; scenario pages written.

Risks:
- **Unjudged results**, as on the US scenario: ESCI judges about 28 products per query, all from Amazon's own ranking. judged@10 is reported with every nDCG.
- **The embedding model is weaker in English than bge-small.** That is irrelevant here (no English scenario uses it), but semantic and hybrid scores on these two scenarios must not be compared with the English ones.
- **Plugin images.** The PGroonga + pgvector combination may not exist as an official package (§6.2 says what happens then).
- **Qdrant's options for Japanese** may not include word splitting. If so, Qdrant's Japanese keyword scores reflect that, which is a result, and its README says so.

## 8. Testing

- **Loader:** `locale` filter for `jp` and `es` on fixture parquet files; the 2,000-query sample is stable for a seed; text is not width-folded or kana-converted.
- **Perturbations:** `accents` strips diacritics (`eléctrico` → `electrico`, `niño` → `nino`), returns nothing for a query without diacritics, and is deterministic; a `ja` scenario with `typo` in its kinds is rejected by scenario validation with a message saying why.
- **Embedding:** `doc_prefix` is applied to documents only; bge-small vectors for an English fixture are unchanged.
- **Adapters:** for each of the 6, a rendered request or index body for `ja` and `es` contains the setting in §6.1; for `en`, every rendered request equals the one before the change (§5.2).
- **Scenario:** `embedding_model` set and `--model` different fails `embed` and `run`; unset behaves as today. Lint reports a non-English scenario whose engine config sets no language key.
- **Report:** default runs of these scenarios appear in the default-vs-tuned table only, never in best-score marking, significance tests or other views.
- **Compose override:** a tuned track with `compose` starts that file; lint rejects an unpinned `FROM`, a missing `justification.compose`, and a compose file that does not exist.

## 9. Deliverables

- `datasets/amazon-esci-jp/dataset.yaml`, `datasets/amazon-esci-es/dataset.yaml`; `locale` parameter in `harness/irbench/datasets/amazon_esci.py`.
- Harness: `Scenario.language`; the `accents` perturbation; `multilingual-e5-small` and `doc_prefix`; `TrackConfig.compose` and the lint changes; report pages showing both tracks for these scenarios.
- Scenarios: `scenarios/ecommerce/{amazon-search-ja,amazon-search-es}/scenario.yaml`.
- Engine configs: 12 files; the Elasticsearch kuromoji Dockerfile and compose file; the Postgres PGroonga ones if §6.2 allows.
- Each engine README: what it is given per language and track, and why.
- Docs: one Mintlify page per scenario (source, license, counts, the language settings table, the `accents` coverage, the unjudged-results caveat).

## 10. Later

- Chinese product search: the C-MTEB EcomRetrieval set (1,000 queries, about 100k product titles) would test a third script, once its license is clear (the Multi-CPR repository has none).
- Japanese typing mistakes as input methods make them (wrong kanji for the right sound) would need a conversion dictionary; no generator is planned.
- Giving Elasticsearch its `english` analyzer on the English scenarios, to match §6.1.

## 11. Open questions

None from the user. Calls made in this spec, to check while reading:
- multilingual-e5-small as the model, pinned on the scenario; text-embedding-3-small not run (§2, §5.1, §5.3);
- full catalogues and 2,000 queries per locale (§2);
- `accents` for Spanish, no `typo` for Japanese (§3.4, §3.5);
- engine language settings are config keys in each `default` block, with a lint check that a non-English scenario sets them (§5.2);
- the per-engine default settings in §6.1, especially Elasticsearch's `cjk` analyzer and Postgres's `simple` configuration for Japanese;
- plugins are tried in the sweep and kept only if they win, with their score reported either way (§6.2);
- default runs of these two scenarios are published, shown in one table per scenario page, and kept out of winner marking, significance tests and other views (§5.5);
- Postgres's tuned Japanese track falls back to `simple` if no official PGroonga + pgvector image exists (§6.2);
- built after the US ESCI plan lands (§7).
