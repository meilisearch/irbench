# IRBench — Home Depot product search with real misspellings

Date: 2026-10-31
Status: Draft; internal-only use decided 2026-11-01, source still open (§11, Q2)
Parent specs: `2026-10-09-irbench-design.md`, `2026-10-20-irbench-amazon-esci-design.md` (grade scale, loader and scenario layout)

Example queries in this document are invented, written in the style of the data. The real queries are Kaggle competition data and are not reproduced in this public repository (§3).

## 1. Goal

Three of the four published scenarios send long natural-language questions to document collections. WANDS and the planned ESCI scenarios send product queries, but neither says which of its queries are misspelled. The only typing mistakes the benchmark measures are synthetic: the `typo` perturbation swaps letters for their keyboard neighbours. Nobody has checked that this resembles the mistakes shoppers actually make.

Misspelled queries are a large share of real store traffic. In this dataset, Google's spelling correction changes 13.9% of the queries shoppers typed into homedepot.com.
- The queries are short (median 3 words) and full of brands, part numbers and sizes.
- Their mistakes are mostly not keyboard slips. They are phonetic spellings (`guage` for `gauge`), misspelled brand names, and words glued together or split apart (`outdoorlights`, `dish washer`).

This change adds one scenario, `ecommerce/home-depot-search`, built on the Home Depot Product Search Relevance data (Kaggle, 2016):

| What it adds | How |
|---|---|
| Short store queries as typed, with real misspellings at their natural share | 4,000 real queries; each is labelled `misspelled` or `clean` |
| The cost of a real misspelling, measured on the same intent | Each misspelled query is also run with its spelling corrected; the two share their judgments, so the difference is the cost of the mistake |
| A check of the synthetic `typo` perturbation | The synthetic `typo` loss on clean queries is shown next to the real-misspelling loss |

Results stay internal: the data's terms do not allow publishing it (§2, §3).

### Non-goals

- Spelling correction as a feature to benchmark (suggesting "did you mean"). Engines are scored on the results they return for the query as typed.
- As-you-type: the data has complete queries only.
- Images: raters saw product images; the data has none.

### Anti-goals

- Anything that publishes or redistributes the data, its derived vectors, or results computed from it (§3).
- A misspelling score that only measures how well an engine handles the synthetic generator. The real misspellings are the point; the synthetic ones are only the comparison.

## 2. Decisions

| Topic | Decision | By |
|---|---|---|
| Licence and use | Internal only. Never published: no runs in `results/`, no published vectors, no docs page, no scores in committed files. A new `distribution: internal` flag on the dataset enforces what it can (§6.1) | user, 2026-11-01 |
| Source | The original Kaggle files, downloaded by each person who runs the scenario with their own Kaggle account, after accepting the competition rules. The Hugging Face mirror is not used (§3.2) | my call, the user must confirm (§11, Q2) |
| Corpus | The 54,667 products judged in the training set, with title, brand and description (§4.4) | my call |
| Attributes file | Used for brand only. Its bullet points repeat the description (§4.1) | my call |
| Grades | Averaged 1–3 rating rounded to the nearest rating, then mapped onto WANDS's 0–2 scale (§4.3) | my call |
| Queries | Seeded sample of 4,000 of the 11,700 usable queries, keeping the natural share of misspellings | my call |
| Misspellings | A query group (`misspelled`, `clean`) on every query, plus a data-supplied variant kind `corrected` for the misspelled ones, built from the Google spelling dictionary posted during the competition (§5) | my call |
| Headline score | The usual pooled row (originals plus the synthetic variants, as on every scenario), with `corrected` kept out of it (§6.4) | my call |
| Modes | keyword, semantic, hybrid, filtered and faceted on brand. No sorted mode: the data has no price, rating or date | my call |

## 3. Licence

### 3.1 What the terms say

- The data is distributed only through Kaggle, under the competition rules ("Subject to Competition Rules" on the data page). There is no data licence.
- The rules limit use to the purpose and duration of the competition, and participants agree "not to transmit, duplicate, publish, redistribute" the data (Kaggle, Home Depot Product Search Relevance, Competition Rules). They must also keep it from people who have not accepted the rules.
- The competition ended on 2016-04-25; late submissions are still open, so the data is still downloadable after accepting the rules.
- The Hugging Face copies (`bstds/home_depot` with no licence; `Ukhushn/home-depot` marked AFL-3.0; `harisarang/home_depot` marked MIT) are uploads by third parties. They cannot grant a licence the sponsor never gave.
- The rules let participants share code publicly, and code shared that way is licensed MIT. The spelling dictionary (§5.1) was shared that way, but its content is competition query text. This spec treats that text as Kaggle data: it is fetched from its public location, never copied into the repository.

### 3.2 Consequences for this spec

- **Nothing derived from the data is published**: no run bundle in `results/`, no report on the public site, no packed vectors in a GitHub release, no Mintlify page.
- **The repository is public (`meilisearch/irbench`), so committed files carry no data and no results.** They are:
  - the loader code;
  - `dataset.yaml`: the competition slug, file names and checksums, the dictionary's URL;
  - the scenario file and the engine configs;
  - synthetic test fixtures;
  - this spec.
- Committed files hold no query text, product text or scores. Two existing habits change for this scenario:
  - **Tuned-config justifications:** `docs/methodology/parity.mdx` asks each tuned setting's justification to state its measured gain, and existing configs quote nDCG values. For an internal scenario the justification says what the setting does and that the sweep chose it, with no numbers. The numbers stay in the internal sweep notes, outside the repository. Lint does not check for numbers, so this is a documented rule only.
  - **No results README:** the scenario folder gets no README with results. Findings such as the grade-sensitivity check (§4.3) go on the internal preview report.
- **The source is Kaggle itself, not a mirror.** Each person who prepares the scenario downloads the files with their own Kaggle account, which requires accepting the rules. That meets the rules' condition that only people who accepted them have the data, and keeps the public repository from pointing at an unlicensed copy.
- Even internal benchmarking is outside "the purpose of the competition" read strictly. The user accepted internal use on 2026-11-01 (§11, Q1).

## 4. Data

### 4.1 Files

From the Kaggle competition `home-depot-product-search-relevance`, pinned by sha256 in `datasets/home-depot/dataset.yaml`:

| File | Used for |
|---|---|
| `train.csv.zip` | Query–product pairs with their averaged rating: `id, product_uid, product_title, search_term, relevance` |
| `product_descriptions.csv.zip` | `product_uid, product_description` |
| `attributes.csv.zip` | `product_uid, name, value`; only the `MFG Brand Name` rows are read |
| Google spelling dictionary (§5.1) | Misspelled flag and corrected text |

Not used:
- `test.csv`: its relevance labels were never released, so its pairs cannot be scored.
- The bullet-point attributes (`Bullet01` and so on). They repeat text the description already contains: Home Depot descriptions end with the bullet list, its items run together with no space or full stop between them.
- The technical specifications (`Product Width (in.)` and so on), left for later (§10).

The Kaggle `train.csv` is expected to be Latin-1, not UTF-8 (competitors read it with `encoding="ISO-8859-1"`). The builder checks the encoding on the real file and decodes accordingly.

### 4.2 Counts

Measured 2026-10-31 on the Hugging Face mirror `bstds/home_depot` (revision `e4c0747e`), which is `train.csv` joined with the descriptions. The mirror was used only to inspect the data. The builder confirms the same counts on the Kaggle files (74,067 pairs, 11,795 queries, 54,667 products) and stops if they differ.

| | |
|---|---|
| Judged query–product pairs | 74,067 |
| Distinct queries | 11,795 |
| Distinct products | 54,667 (53,489 distinct titles) |
| Judgments per query | mean 6.3, median 7, max 16; 1,399 queries have a single judgment |
| Query length | median 3 words, 19 characters; 644 one-word queries |
| Queries with a digit (sizes, part numbers, voltages) | 26% |
| Queries with a capital letter | 3% |
| Description length | median 734 characters; no HTML tags; 9,446 with HTML entities |

Ratings (each pair rated by at least three people, 1 = not relevant, 2 = somewhat, 3 = highly; the file holds the average):

| Average | 1.00 | 1.25 | 1.33 | 1.50 | 1.67 | 1.75 | 2.00 | 2.25 | 2.33 | 2.50 | 2.67 | 2.75 | 3.00 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pairs | 2,105 | 4 | 3,006 | 5 | 6,780 | 9 | 11,730 | 11 | 16,060 | 19 | 15,202 | 11 | 19,125 |

### 4.3 Grades

The average is rounded to the nearest rating, and the rating is mapped as WANDS maps its labels (Irrelevant 0, Partial 1, Exact 2):

| Average | Rating | Grade | Pairs |
|---|---|---|---|
| below 1.5 | 1, not relevant | 0 | 5,115 |
| 1.5 to below 2.5 | 2, somewhat relevant | 1 | 34,595 |
| 2.5 and above | 3, highly relevant | 2 | 34,357 |

- Why rounding (my call): it reads the average back onto the scale the raters used, with no threshold of our own. The average alone cannot tell (2, 2, 1) from (3, 1, 1), so no rule based on the majority vote is possible.
- Grade-0 pairs stay in the qrels as judged non-relevant, so judged@10 counts them, as on ESCI.
- The scores sit on the same 0–2 scale as WANDS and ESCI.
- Ties at exactly .5 (24 pairs) round up.
- Sensitivity check: the alternative reading, where only averages of 2 and above count as relevant (1.67 becomes 0), is scored once on the tuned runs by rescoring, with no new runs. If the order of engines changes on any base, the internal preview report says so.

### 4.4 Documents

- `id`: `product_uid` as a string.
- `title`: `product_title`, stripped. Fifteen products carry two different titles across their rows; the title of the row with the smallest `id` is kept.
- `brand`: the first `MFG Brand Name` value for the product, stripped; null when absent.
- `description`: `product_description` with HTML entities unescaped (`html.unescape`) and whitespace collapsed.
  - The run-together bullets are left as they are: every engine sees the same text.
  - Splitting on case changes would also split product names written in camel case.
- Corpus: every product judged in `train.csv` (54,667). Every product is judged for at least one query; a product judged only for other queries is a distractor.
- Why not the 124,428 products of `product_descriptions.csv` (my call):
  - The test set reuses some training queries with other products, whose labels are hidden.
  - With those products in the index, an engine that returns them is scored as if they were wrong, for queries where they may be the best answer.
  - Keeping to judged products avoids that bias. §10 keeps the larger corpus as an option.
- The builder measures brand coverage on the corpus. If fewer than half the products have a brand, the filtered and faceted modes are dropped and the internal preview report says why.

### 4.5 Queries

- Usable queries: the 11,700 queries with at least one product of grade 1 or 2. The 95 with none are dropped and counted in `prepare.manifest.json`.
- Sample: 4,000 of them, seeded (`20261031`), as loader constants like ESCI's (my call).
  - The sample keeps the natural share of misspellings, so about 555 misspelled queries are expected.
  - 4,000 rather than ESCI's 2,000 so that the misspelled group has enough queries for a useful confidence interval.
  - The run cost stays small because the corpus is 55k products.
- id: `hd-<n>`, where `n` is the smallest Kaggle row `id` among the query's pairs. The data has no query id, and this one is stable.
- text: whitespace-normalised, case kept, as for WANDS and ESCI. 344 queries contain a double space; it is part of no mistake an engine should see.
- `group`: `misspelled` or `clean` (§5.2).
- `corrected`: for misspelled queries only, the dictionary's correction, whitespace-normalised.

## 5. Misspellings

### 5.1 The dictionary

During the competition a participant (Kaggle user `steubk`, kernel "Fixing typos") ran every search term through Google's spelling correction and posted the result as a dictionary. The third-place team kept a copy in their public repository:

- `github.com/ChenglongChen/Kaggle_HomeDepot`, file `Code/Chenglong/google_spelling_checker_dict.py`, commit `55c1033d0af3b6cf2f033fe4bcf3e1e0ffda3445`, repository licence MIT;
- sha256 `76edf5ee8de0da6cd9385e5713f0866ac6f38d76aae1c80eb8b921a876615fd2`, 3,364 entries covering training and test queries;
- it maps a whole query to its corrected query.

`dataset.yaml` lists it as a `url` file pinned to that commit (§3.1 for why it is fetched, not copied).

It is a Python file. The loader parses it with `ast.parse` and reads the dict literal with `ast.literal_eval`. It never imports or executes it.

Measured against the training queries:

| | |
|---|---|
| Training queries in the dictionary (exact string match on the raw query) | 1,640 of 11,795 (13.9%) |
| Respelled, same number of words | 1,442 (about 1,360 of them change exactly one word) |
| Only spaces change (two words glued, or one word split) | 97 |
| Words added or removed as well as respelled (a unit added to a number, for example) | 101 |
| Edit distance 1 / 2 / 3 / 4+ | 1,086 / 415 / 100 / 39 |

Quality, checked by hand on samples:
- Recall is high: 11 of 12 misspellings picked at random from the training queries are in it.
- Most corrections are right. A few are wrong or go further than spelling, for example a misspelled plural corrected to the wrong noun, or a connector type changed.

So `misspelled` means "Google's speller changed it". A few clean queries still contain mistakes, and a few corrections are wrong. The scenario's preview page states both.

### 5.2 How the misspellings are used

Both views, because they answer different questions:

1. **Group.** Every query is `misspelled` or `clean`, and scores are reported per group.
   - The original queries, as typed, carry real traffic's 14% of mistakes.
   - The group split shows how much worse an engine does on the misspelled ones. That comparison mixes in other differences between the two sets of queries.
2. **Paired variant `corrected`.** Each misspelled query also runs with its corrected text, with the same judgments. The difference between the two, on the same queries and the same intents, is the cost of the misspelling and nothing else.

The variant goes in that direction, original as typed and variant corrected, because the raters judged the query as typed: the qrels belong to the misspelled text. The reverse (corrected text as the original) would turn 14% of the original queries into text no one typed.

The synthetic perturbations still run (`typo`, `prefix`) on every query. On the scenario page:
- real misspelling cost: `corrected` minus `original` on the misspelled group;
- synthetic typo cost: `original` minus `typo` on the clean group.

If an engine loses little to synthetic typos but much to real ones, the synthetic perturbation is not measuring what shoppers do. That holds for every scenario that uses it.

## 6. Harness changes

### 6.1 Internal-only datasets

The harness has no such mechanism today: `irbench publish` refuses only the `local` hardware profile, `irbench vectors pack` publishes any scenario's vectors, and the report renders whatever runs it is given.

- `DatasetSpec.distribution: Literal["public", "internal"] = "public"`. A scenario is internal when its dataset is. `datasets/home-depot/dataset.yaml` sets `internal`, and its `license` field reads "Kaggle competition rules, internal use only".
- `irbench publish` refuses a run of an internal scenario, whatever its hardware profile, with a message that names the dataset and its licence.
- `irbench vectors pack` refuses an internal scenario. `irbench embed` never looks for published vectors for one.
- `irbench report`:
  - refuses to build when `--results` is the repository's `results/` directory and contains an internal run (the public site is built from there);
  - for previews (`--results .irbench/runs`), shows a banner "Internal: not for publication (Kaggle competition data)" on the index and on every page that shows an internal scenario.
- New lint rule `internal-data`: fails if `results/` holds a run of an internal scenario, if `embeddings/published.yaml` lists one, or if `docs/docs.json` links a page for one.
- CI does not prepare internal datasets: it has no Kaggle account, by design.
  - For a config whose scenario is internal and not prepared, the `not-prepared` finding becomes a notice that does not fail lint.
  - The builder runs `irbench lint` locally after `prepare`, so `query-leak` is still checked before configs are committed.
- Tests for internal datasets use synthetic fixtures only.
- `docs/methodology/parity.mdx` gains a short "Internal scenarios" paragraph:
  - what the flag means;
  - that internal results are never published;
  - that their tuned justifications carry no scores (§3.2).

The mechanism is generic, so other datasets with restrictive terms can use it.

### 6.2 Manually downloaded raw files

Kaggle files need a signed-in account that has accepted the rules, so they cannot be fetched by URL.

- `FileSpec` gains `manual: str | None`, an instruction shown to the runner, used instead of `url`. Exactly one of `url` and `manual` is set.
- `fetch_dataset` looks for the file under the dataset's raw directory and checks its sha256. If the file is missing, it fails with the instruction. For Home Depot:
  - accept the rules at `kaggle.com/competitions/home-depot-product-search-relevance/rules`;
  - run `kaggle competitions download -c home-depot-product-search-relevance -p <raw dir>`, or download the three zips from the data page;
  - then rerun `prepare`.
- The harness handles no Kaggle credentials (my call): the runner downloads the files, the harness only checks them.
- The sha256 values are not known yet. The builder records them from the first download. The zips have not changed since 2016.
- The loader reads the CSVs from inside the zips (`zipfile`, standard library).

### 6.3 Data-supplied variant: `corrected`

- `Query.corrected: str | None = None`, read by `prepare` from `queries.jsonl` like `group`.
- `Perturbation` and `PerturbationSpec.kinds` gain `corrected`.
  - Its generator returns the query's `corrected` text, or nothing when the query has none (as `accents` does for a query with no diacritic).
  - The existing check in `make_variants` still applies: it drops a variant equal to its original once lower-cased and whitespace-collapsed.
- Variant tasks keep the original's qrels and group, as every variant does today.

### 6.4 Scores per group and perturbation, and the headline

The report's headline nDCG, its confidence interval, judged@10 and the winner marking all read the `(mode, base, "all")` row. Today `score_relevance` builds that row from every task: originals and synthetic variants together. That stays as it is for every scenario.

- **`corrected` is kept out of the `all` row**, and out of the per-group `all` rows. It is a reference measurement (the query as it should have been typed), not a perturbation of what users type. Pooling it would put about 555 cleaned-up queries into the headline. It gets its own `(mode, base, "corrected")` row, like any perturbation.
- The tuned sweep's objective is the same `all` row as on the other scenarios, so `corrected` does not reach the sweep either (§8).
- Group rows are also emitted per perturbation: `(mode, base, group, perturbation)` rows appear next to the current `(mode, base, group, all)` rows, so an `original` row exists per group. Existing readers of the `all` rows are unchanged.
- The paired difference for `corrected` and its 95% confidence interval come from per-task nDCG (`per_task_ndcg`), pairing each variant task with its original.

### 6.5 Report

The scenario page (preview only, §6.1) adds a misspelling table per base. One row per engine, with:
- nDCG@10 on misspelled queries as typed, and with `corrected`;
- the difference, with its confidence interval;
- the synthetic `typo` difference on clean queries.

It also shows:
- the number of queries in each group;
- judged@10 next to every nDCG;
- the dictionary's caveats (§5.1);
- the grade-sensitivity result (§4.3).

## 7. Scenario

```yaml
id: ecommerce/home-depot-search
dataset: home-depot
index_fields:
  searchable: [title, brand, description]
  filterable: [brand]
  sortable: []
  facets: [brand]
modes: [keyword, semantic, hybrid, filtered, faceted]
structured_bases: [keyword, semantic]
embedding_fields: [title, brand, description]
filter_templates:
  - {field: brand, op: eq}
sort_templates: []
top_k: 100
index_timeout_s: 1800
perturbations:
  seed: 20261031
  kinds: [typo, prefix, corrected]
perf:
  page_sizes: [20, 100]
```

- No `synonym`: there is no vocabulary file for this catalogue, and building one is not in scope.
- `description` is in the embedding text, unlike ESCI:
  - Home Depot descriptions are short enough (median 734 characters) to fit bge-small's 512 tokens;
  - they carry the product's features.
- The embedding model is the default bge-small; the scenario does not pin one.

## 8. Engine configs and runs

- Six config files, default and tuned, following the existing justification rules, with no scores in the justifications (§3.2).
- **Tuned track:** the usual sweep of documented search settings, scored on nDCG@10 per base on the `all` row (§6.4).
  - The sweep does not score the misspelled group alone: that would reward typo settings at the cost of clean queries.
  - Typo-tolerance settings that the parity rules already allow are part of the sweep, as on the other scenarios (for example `fuzziness` in Elasticsearch, `fuzzy` in ParadeDB).
- **Runs** are on the laptop, with the `local` profile. The scenario is never published, so no reference-machine runs are needed.
  1. Download the Kaggle files (§6.2), prepare, and embed 54,667 products and the query variants.
  2. Default track on the 6 engines, then sweeps, then the tuned track.
  3. Preview report from `.irbench/runs`, and the grade-sensitivity rescoring (§4.3).

Risks:
- **Unjudged results.** About 6 products are judged per query, chosen by Home Depot's own search. Engines that return other relevant products are scored as wrong. judged@10 is reported with every nDCG.
- **The dictionary is a machine's corrections.** A wrong correction makes `corrected` look worse than it should. The preview page gives the measured share of edits by size (§5.1), so a reader can judge.
- **Misspelled and clean queries differ in more than spelling.** The group comparison is indicative only; the paired `corrected` comparison is the measurement.

## 9. Testing

- **Loader** (synthetic CSV zips and a synthetic dictionary file, all strings invented):
  - grades at the boundaries (1.33 → 0, 1.5 → 1, 2.33 → 1, 2.5 → 2, 3 → 2);
  - a query with no grade ≥ 1 is dropped and counted;
  - title from the smallest row id when titles differ;
  - brand from `MFG Brand Name` only, null when absent;
  - HTML entities unescaped;
  - query ids `hd-<min row id>`;
  - the 4,000 sample is stable for a seed;
  - the dictionary is matched on the raw query before whitespace normalisation;
  - `group` and `corrected` are set only for matched queries;
  - the dictionary file is parsed without executing it (a fixture containing a function call is rejected);
  - Latin-1 input is decoded.
- **Fetch:** a `manual` file that is missing fails with its instruction; a wrong checksum fails; `url` and `manual` together are rejected.
- **Perturbation:** `corrected` returns the query's correction, nothing without one, and is dropped when it equals the original after normalisation.
- **Scores:**
  - `corrected` tasks are absent from the `all` rows and present in their own row;
  - the `all` rows of a scenario without `corrected` are unchanged;
  - group rows exist per perturbation;
  - the paired `corrected` difference pairs the right tasks.
- **Internal-only:**
  - `publish` refuses an internal run on a reference profile;
  - `vectors pack` refuses an internal scenario;
  - `report` refuses `results/` with an internal run, and shows the banner on a preview;
  - the `internal-data` lint rule catches each of its three cases;
  - `not-prepared` is a notice for an internal scenario only.

## 10. Later

- The full 124,428-product catalogue, as distractors, once the bias from hidden test labels (§4.4) is measured.
- The technical attributes as a searchable field: many queries are sizes and part numbers (26% contain a digit).
- A finer grade scale (the averages carry thirds) if the 0–2 mapping turns out to hide differences.
- Real misspellings for the other e-commerce scenarios: the `corrected` mechanism works for any dataset that ships corrections.

## 11. Open questions for the user

1. **Use and publication.** Answered 2026-11-01: internal only. Results, vectors and docs never leave Meilisearch; only code, configs (with no scores) and checksums go in the public repository.
2. **Source.** This spec has each person who runs the scenario download the files from Kaggle with their own account, after accepting the rules.
   - The simpler alternative is the unlicensed Hugging Face mirror, fetched by URL.
   - The mirror has no attributes, so brand filtering and facets would go.
   - The public repository would point at a copy made against the rules.
   - Do you accept the Kaggle download step?

Calls made in this spec, to check while reading:
- judged-products corpus (54,667), not the full catalogue (§4.4);
- grades by rounding the average (§4.3);
- 4,000 queries (§4.5);
- misspellings as both a group and a `corrected` variant, original as typed (§5.2);
- the Google dictionary as the definition of "misspelled", with its known errors; it is fetched by URL from the third-place team's repository and its text never copied into this one (§3.1, §5.1);
- `corrected` kept out of the headline `all` row and the sweep objective (§6.4);
- brand from the attributes file, bullets and specifications not used (§4.1);
- the internal-only mechanism; lint's `not-prepared` as a notice for internal scenarios in CI; no scores in committed justifications (§3.2, §6.1);
- the harness never handles Kaggle credentials (§6.2).
