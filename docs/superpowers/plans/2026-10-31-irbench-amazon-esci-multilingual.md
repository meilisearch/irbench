# Japanese and Spanish Product Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ecommerce/amazon-search-ja` and `ecommerce/amazon-search-es` (2,000 real Amazon.co.jp and Amazon.es queries each, full locale catalogues) and run every engine on both, with built-in language settings on the default track and official language plugins allowed on the tuned track.

**Architecture:** The existing `amazon_esci` normaliser gains a `locale` and runs without AmazonQAC; two new datasets reuse the US raw files. Scenarios gain `language` and `embedding_model`; a new `accents` perturbation covers Spanish. Engine language settings are config keys (three adapters gain one override key each); a track can name its own compose file, which carries the plugin images. The report keeps default runs out of every ranking view and shows them in one default-vs-tuned table.

**Tech Stack:** Python 3.12, pydantic, pyarrow, sentence-transformers, Jinja2 report, Docker Compose, uv, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-31-irbench-amazon-esci-multilingual-design.md`

**Depends on:** `docs/superpowers/plans/2026-10-21-irbench-amazon-esci.md` Tasks 2–7 (scenario query sets, `scenarios/ecommerce/amazon-search`, its 6 default configs, the shared-dataset lint check, report group order). On this branch only its Task 1 has landed. Do not start Task 7 of this plan until those have.

## Global Constraints

- Locales `jp` and `es`; scenario ids `ecommerce/amazon-search-ja`, `ecommerce/amazon-search-es`; dataset ids `amazon-esci-jp`, `amazon-esci-es`; `language` values `en`, `ja`, `es`.
- Same ESCI commit, files and sha256 as `datasets/amazon-esci/dataset.yaml`; seed `20261020`; sample 2,000; grades E 2, S 1, C 0, I 0; `corpus: full`.
- Query text is normalised exactly as the US loader does (`_normalize_key`); no width folding, no kana conversion, no other Unicode normalisation.
- Embedding model id `multilingual-e5-small`, `intfloat/multilingual-e5-small`, 384 dimensions, query prefix `query: `, document prefix `passage: `.
- Perturbations: es `[typo, prefix, accents]`; ja `[prefix]`. `typo` is rejected for `language: ja`.
- English scenarios must render byte-identical requests and identical vectors after every task (the golden tests in `harness/tests/fixtures/render_golden/` stay unchanged).
- Default track: built-in language setting only. Tuned track: also the vendor's official plugin (Elasticsearch `analysis-kuromoji` matching 9.5.5 exactly; PGroonga with pgvector, both from official packages, or not at all).
- No new Python dependency. Never add `Co-Authored-By` lines to commits.
- `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .` and `uv run irbench lint` pass after every task.

## Review Focus

1. A Spanish query already in decomposed form (`e` + U+0301) must give the same `accents` variant as the composed form, and a query whose only diacritic is `ñ` must get a variant. Task 2 tests both.
2. A Japanese query with no spaces (`楽しくない仕事はなぜ楽しくないのか`) must get a `prefix` variant cut on a character boundary, never empty, and a two-character query (`時計`) none. Task 2 tests it.
3. Running `embed` or `run` on a Japanese scenario with `--model bge-small-en-v1.5` must fail before any download or container start, naming both models. Task 2 tests it.
4. A run on a locally built plugin image has no repo digest: the manifest records `digest: null`, and `publish` and the report must accept it. Task 4 tests it.
5. A scenario where an engine has only a default run (its tuned run not done yet) must still show that engine in the ranking views, so a half-finished results folder is not silently missing an engine. Task 8 tests it.

---

### Task 1: locale-aware ESCI loader and the two datasets

**Files:**
- Modify: `harness/irbench/datasets/amazon_esci.py` (`locale` parameter; QAC optional)
- Modify: `harness/irbench/datasets/registry.py` (`DatasetSpec.raw_from`)
- Modify: `harness/irbench/stages/prepare.py:49` (fetch into `ws.raw_dir(spec.raw_from or spec.id)`)
- Create: `datasets/amazon-esci-jp/dataset.yaml`, `datasets/amazon-esci-es/dataset.yaml`
- Modify: `harness/irbench/datasets/__init__.py` (register both ids to `normalize_amazon_esci`)
- Test: `harness/tests/test_amazon_esci.py`, `harness/tests/test_prepare.py`

**Interfaces:**
- Produces: `normalize_amazon_esci(raw: dict[str, Path], out: Path, corpus: Literal["full", "pool"] = "full", locale: Literal["us", "es", "jp"] = "us") -> NormalizedDataset`. Raw keys: `examples`, `products`, `sources`, and `qac` only when `locale == "us"`.
- Produces: `DatasetSpec.raw_from: str | None = None`, a dataset id whose raw directory is reused.

- [ ] **Step 1: Write the failing tests** (fixture parquet files built in `tmp_path`, as the existing tests do):
  - `test_locale_jp_keeps_only_jp`: with `locale="jp"`, a `us` and an `es` product and their judgments are absent; the jp ones are present.
  - `test_non_us_needs_no_qac`: `raw` without `qac` and `locale="es"` succeeds; the pool is not filtered by QAC sessions; `query_sets == {}`; no `queries.prefixes.jsonl` is written.
  - `test_us_still_requires_qac`: `locale="us"` without `qac` raises `KeyError`.
  - `test_text_not_width_folded`: a jp query `ＡＢＣ　時計` comes out as `ａｂｃ 時計` (lower-cased and whitespace-normalised by `_normalize_key`, full-width letters kept full-width). Adjust the expected string if `_normalize_key` treats U+3000 differently; the assertion that matters is that no character becomes ASCII.
  - `test_group_from_sources_for_es`: an es query's `group` is its `sources.csv` value.
  - `test_raw_from_shares_directory` (in `test_prepare.py`): a spec with `raw_from: amazon-esci` fetches into `raw_dir("amazon-esci")`.

- [ ] **Step 2: Run** `uv run pytest harness/tests/test_amazon_esci.py harness/tests/test_prepare.py -q`. Expected: FAIL (unexpected keyword `locale`, unknown field `raw_from`).

- [ ] **Step 3: Implement.** Replace the two `== "us"` filters with `== locale`; run the QAC block, the `eligible` filter and the prefix rows only when `locale == "us"` (for other locales `eligible = deduped`). Write both `dataset.yaml` files: copy `datasets/amazon-esci/dataset.yaml` without the `qac` file, `params: {corpus: full, locale: jp}` (or `es`), `raw_from: amazon-esci`, `license: "Apache-2.0"`.

- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS, US tests unchanged.

- [ ] **Step 5: Commit.** `git commit -m "datasets: ESCI Japanese and Spanish locales, sharing the US raw files"`

### Task 2: scenario language, pinned model, `accents` perturbation

**Files:**
- Create: `harness/irbench/perturb/accents.py`
- Modify: `harness/irbench/perturb/variants.py` (`_apply` handles `accents`)
- Modify: `harness/irbench/scenario.py` (`language`, `embedding_model`, `accents` kind, ja/typo check)
- Modify: `harness/irbench/stages/embed.py` (`embed_scenario`), `harness/irbench/stages/run.py` (`run_pipeline`): model check first
- Test: `harness/tests/test_perturb.py` (or the existing perturbation test file), `harness/tests/test_scenario.py`, `harness/tests/test_embed_stage.py`, `harness/tests/test_run.py` (or wherever `run_pipeline` is tested)

**Interfaces:**
- Produces: `accents(text: str, rng: random.Random) -> str | None` (rng unused, kept for the `_apply` signature).
- Produces: `Scenario.language: Literal["en", "ja", "es"] = "en"`, `Scenario.embedding_model: str | None = None`, `PerturbationSpec.kinds: list[Literal["typo", "prefix", "synonym", "accents"]]`.
- Produces: `check_model(scenario: Scenario, model_id: str) -> None` in `scenario.py`, raising `ValueError(f"{scenario.id} is embedded with {scenario.embedding_model}, not {model_id}")`.

- [ ] **Step 1: Write the failing tests.**
  - `test_accents`: `"calefactor eléctrico"` → `"calefactor electrico"`; `"niño"` → `"nino"`; `"pingüino"` → `"pinguino"`; `"filtro cafetera"` → `None`; `unicodedata.normalize("NFD", "eléctrico")` → `"electrico"`.
  - `test_prefix_on_unspaced_japanese`: `prefix("楽しくない仕事はなぜ楽しくないのか", random.Random(1))` is a non-empty proper prefix of the input; `prefix("時計", ...)` is `None`.
  - `test_ja_rejects_typo`: a scenario with `language: ja` and `kinds: [typo]` fails validation with a message containing `typo` and `ja`.
  - `test_language_defaults_to_en`: every existing scenario under `scenarios/` loads with `language == "en"` and `embedding_model is None`.
  - `test_embed_rejects_other_model` and `test_run_rejects_other_model`: with `embedding_model: multilingual-e5-small`, calling with `bge-small-en-v1.5` raises `ValueError` naming both, and the fake embedder / fake runtime records no call.

- [ ] **Step 2: Run them.** Expected: FAIL.

- [ ] **Step 3: Implement.** `accents`: NFD, drop characters with `unicodedata.combining(c)`, NFC; return `None` when unchanged. Call `check_model` at the top of `embed_scenario` and `run_pipeline`, before any fetch or `up`.

- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS; English variant files unchanged (the existing variant tests pin them).

- [ ] **Step 5: Commit.** `git commit -m "harness: scenario language and pinned embedding model; accents perturbation"`

### Task 3: `multilingual-e5-small` and document prefixes

**Files:**
- Modify: `harness/irbench/embed/models.py`
- Test: `harness/tests/test_embed_models.py` (create if absent; stub `SentenceTransformer` so no model downloads)

**Interfaces:**
- Produces: `SentenceTransformerEmbedder(model_id: str, hf_name: str, dim: int, query_prefix: str = "", doc_prefix: str = "")`.
- Produces: `MODELS["multilingual-e5-small"]` with the values in Global Constraints.

- [ ] **Step 1: Write the failing tests.** With `sentence_transformers.SentenceTransformer` monkeypatched to a stub that records its inputs:
  - `test_doc_prefix`: `embed(["a"], "doc")` sends `["passage: a"]`; `embed(["a"], "query")` sends `["query: a"]` for `multilingual-e5-small`.
  - `test_bge_unchanged`: for `bge-small-en-v1.5`, docs are sent with no prefix and queries with the existing bge prefix.

- [ ] **Step 2: Run them.** Expected: FAIL (unknown model).

- [ ] **Step 3: Implement** the parameter and the registry entry.

- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS.

- [ ] **Step 5: Commit.** `git commit -m "embed: multilingual-e5-small, with document prefixes"`

### Task 4: per-track compose file

**Files:**
- Modify: `harness/irbench/engines/config.py` (`TrackConfig.compose`)
- Modify: `harness/irbench/runtime/docker.py` (`DockerRuntime(..., compose: str = "compose.yaml")`)
- Modify: `harness/irbench/stages/run.py` (pass `ctx.config.track(spec.track).compose`)
- Modify: `harness/irbench/lint.py` (`_settings` yields `compose`; new checks below)
- Test: `harness/tests/test_runtime.py`, `harness/tests/test_lint.py`, `harness/tests/test_report_data.py`

**Interfaces:**
- Produces: `TrackConfig.compose: str | None = None` (a file name in `engines/<engine>/docker/`, no path separators).
- Produces: lint rule names `compose-missing` and reuse of `image-pin` and `unjustified-setting`.

- [ ] **Step 1: Write the failing tests.**
  - `test_runtime_uses_track_compose`: a `DockerRuntime` built with `compose="compose.kuromoji.yaml"` puts that path after `-f` in every compose command.
  - `test_compose_needs_justification`: a tuned block with `compose:` and no `justification.compose` gives `unjustified-setting`.
  - `test_compose_must_exist`: `compose: nope.yaml` gives `compose-missing`; `compose: ../x.yaml` is rejected at config load.
  - `test_build_compose_pins_from`: a compose file whose engine service uses `build:` passes `image-pin` when the Dockerfile's first `FROM` has a version tag, and fails on `FROM docker.elastic.co/elasticsearch/elasticsearch` or `:latest`.
  - `test_null_digest_accepted`: a run manifest with `engine_image: {image: "irbench-es-kuromoji:9.5.5", digest: null}` loads in `load_results` and `publish_run` without error.

- [ ] **Step 2: Run them.** Expected: FAIL.

- [ ] **Step 3: Implement.** `_compose_image` returns the build's Dockerfile `FROM` image when the service has `build:`; the existing `IMAGE_RE` check then applies. A compose service with `build:` must also set `image:` (the local tag the manifest records).

- [ ] **Step 4: Run** `uv run pytest -q` and `uv run irbench lint`. Expected: PASS, 0 findings.

- [ ] **Step 5: Commit.** `git commit -m "harness: a track may run its own compose file"`

### Task 5: adapter language keys and the lint check

**Files:**
- Modify: `engines/elasticsearch/adapter.py` (`index.analyzer`)
- Modify: `engines/postgres/adapter.py` (`index.text_search_config`)
- Modify: `engines/postgres-paradedb/adapter.py` (`index.tokenizer`)
- Modify: `harness/irbench/engines/base.py` (`EngineAdapter.language_keys: tuple[str, ...] = ()`), and each of the 6 adapters sets it
- Modify: `harness/irbench/lint.py` (`language-setting` rule)
- Test: `harness/tests/test_language_settings.py` (create), `harness/tests/test_lint.py`

**Interfaces:**
- Produces, per adapter, the key that carries the language (each is popped from `config.index` before the rest is merged, like Typesense's `searchable_fields`):
  - Elasticsearch `index.analyzer: str`: every searchable text field gets `"analyzer"` and `"search_analyzer"` set to it; the custom `irbench_search` analyzer is still built when synonyms exist and then wins as `search_analyzer`.
  - Postgres `index.text_search_config: str`: the scenario's configuration is created with `COPY = <value>` instead of `COPY = english`.
  - ParadeDB `index.tokenizer: str`: a pg_search tokenizer expression cast onto each searchable column in the `bm25` index (`(title::<expr>)`). Check the exact 0.26 syntax for Lindera Japanese and the Spanish stemmer against the pg_search docs before writing the test.
  - Meilisearch, Typesense, Qdrant: no code; their existing pass-through already carries it (`index.localizedAttributes` plus `search.<base>.locales`; `index.searchable_fields.locale`; `index.bm25_options`).
- Produces: `language_keys`: meilisearch `("index.localizedAttributes",)`, elasticsearch `("index.analyzer",)`, typesense `("index.searchable_fields.locale",)`, qdrant `("index.bm25_options",)`, postgres `("index.text_search_config",)`, postgres-paradedb `("index.tokenizer",)`.

- [ ] **Step 1: Write the failing tests.**
  - `test_es_analyzer_key`: with `index: {analyzer: cjk}`, every searchable field in `index_body()["mappings"]` has `analyzer == search_analyzer == "cjk"`, and `analyzer` does not appear under `settings`.
  - `test_pg_text_search_config_key`: with `index: {text_search_config: spanish}`, `setup_ddl()` contains `COPY = spanish`.
  - `test_paradedb_tokenizer_key`: with `index: {tokenizer: <expr>}`, `text_index_ddl()` casts each searchable column to `<expr>`.
  - `test_meili_typesense_qdrant_pass_through`: the rendered settings / collection body / inference options contain the configured `localizedAttributes`, `locale: "ja"` and `bm25_options` values.
  - `test_english_unchanged`: the existing golden render tests pass with no fixture change (run them; do not edit fixtures).
  - `test_lint_language_setting`: a scenario with `language: es` whose engine config's `default` block sets none of that engine's `language_keys` gives a `language-setting` finding naming the engine; an English scenario never does.

- [ ] **Step 2: Run them.** Expected: FAIL.

- [ ] **Step 3: Implement.** Lint loads each engine's class with `load_adapter_class` (as `stages/context.py` does) to read `language_keys`.

- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS, golden fixtures untouched (`git status harness/tests/fixtures` shows nothing).

- [ ] **Step 5: Commit.** `git commit -m "engines: language settings as config keys; lint non-English scenarios"`

### Task 6: plugin images (kuromoji, PGroonga)

**Files:**
- Create: `engines/elasticsearch/docker/Dockerfile.kuromoji`, `engines/elasticsearch/docker/compose.kuromoji.yaml`
- Create (only if Step 3 succeeds): `engines/postgres/docker/Dockerfile.pgroonga`, `engines/postgres/docker/compose.pgroonga.yaml`
- Modify (only if Step 3 succeeds): `engines/postgres/adapter.py` (`index.text_search: pgroonga`)
- Test: `harness/tests/test_language_settings.py`, plus docker-marked tests in `harness/tests/test_adapter_elasticsearch.py` and `harness/tests/test_adapter_postgres.py`

**Interfaces:**
- Consumes: `TrackConfig.compose` (Task 4), `index.analyzer` (Task 5).
- Produces (if PGroonga is possible): Postgres `index.text_search: Literal["tsvector", "pgroonga"]`, default `tsvector`. With `pgroonga`, the text index is `USING pgroonga` over the searchable columns and `keyword_match` uses `&@~` with `pgroonga_score`.

- [ ] **Step 1: Write the Elasticsearch image.** `FROM` the same image and tag as `engines/elasticsearch/docker/compose.yaml`, then `RUN bin/elasticsearch-plugin install --batch analysis-kuromoji`. The compose file copies `compose.yaml` and swaps `image:` for `build:` plus a local `image: irbench-elasticsearch-kuromoji:9.5.5`.

- [ ] **Step 2: Docker test** `test_kuromoji_analyzer` (marked `docker`): start the kuromoji compose file, create an index with `index.analyzer: kuromoji`, and check `_analyze` splits `空調服ズボン` into more than one token.

- [ ] **Step 3: PGroonga feasibility.** Try a Dockerfile `FROM` the Postgres image in `engines/postgres/docker/compose.yaml`, installing PGroonga from its official apt repository and pgvector from the PGDG package for the same major version. If both install and `CREATE EXTENSION pgroonga; CREATE EXTENSION vector;` succeed, continue. Otherwise delete the files, write one line in `engines/postgres/README.md` saying which package was missing, and skip Steps 4–5.

- [ ] **Step 4: PGroonga adapter path.** Test `test_pgroonga_render`: with `index: {text_search: pgroonga}` the DDL has `USING pgroonga` and the keyword SQL has `&@~`; without it, the golden render is unchanged. Docker test `test_pgroonga_japanese`: a document `空調服 ズボン メンズ` is found by the query `ズボン`.

- [ ] **Step 5: Run** `uv run pytest -q` and `uv run pytest -m docker -k "kuromoji or pgroonga"`. Expected: PASS.

- [ ] **Step 6: Commit.** `git commit -m "engines: kuromoji image for Elasticsearch; PGroonga image for Postgres"` (drop the PGroonga part of the message if Step 3 failed).

### Task 7: the two scenarios and their default configs

Requires the US plan's Tasks 2–5.

**Files:**
- Create: `scenarios/ecommerce/amazon-search-ja/scenario.yaml`, `scenarios/ecommerce/amazon-search-es/scenario.yaml`
- Create: `engines/<engine>/configs/ecommerce/amazon-search-ja.yaml` and `amazon-search-es.yaml` for the 6 engines

- [ ] **Step 1: Write the scenarios.** Copy `scenarios/ecommerce/amazon-search/scenario.yaml`, then set `id`, `dataset`, `language`, `embedding_model: multilingual-e5-small` and `perturbations.kinds` from Global Constraints. Leave `index_timeout_s` at the US value until Task 9 measures it.

- [ ] **Step 2: Write the 12 configs.** Copy each engine's `amazon-search.yaml` default block and add the language setting from spec §6.1, each with a `justification:` entry citing the default-track rule:
  - Meilisearch: `localizedAttributes: [{attributePatterns: ["*"], locales: [jpn]}]` (`spa`), and `search.<keyword|hybrid>.locales: [jpn]` (`spa`);
  - Elasticsearch: `analyzer: cjk` / `analyzer: spanish`;
  - Typesense: `searchable_fields: {locale: ja}` / `es`;
  - Qdrant: `bm25_options` with the language option documented for v1.19.2 (check the server accepts it: start Qdrant and send one inference request);
  - Postgres: `text_search_config: simple` / `spanish`;
  - ParadeDB: `tokenizer:` the Lindera Japanese / Spanish stemmer expressions verified in Task 5.
  No `tuned:` block yet.

- [ ] **Step 3: Run** `uv run irbench lint` (it reports `not-prepared` until Task 9 prepares the scenarios; every other rule must give 0 findings) and `uv run pytest -q`.

- [ ] **Step 4: Commit.** `git commit -m "scenarios: Japanese and Spanish Amazon search, default configs"`

### Task 8: report, default runs kept out of rankings

**Files:**
- Modify: `harness/irbench/report/data.py` (`ScenarioResults.ranked_runs`)
- Modify: `harness/irbench/report/render.py` (ranking views use `ranked_runs`; new `track_comparison_view`)
- Modify: `harness/irbench/report/templates/scenario.html.j2` ("Built-in settings vs tuned" section)
- Modify: `docs/methodology/parity.mdx` (default runs published for these two scenarios)
- Test: `harness/tests/test_report_render.py`, `harness/tests/test_report_data.py`

**Interfaces:**
- Produces: `ScenarioResults.ranked_runs -> list[RunResult]`: every run except a `default` run whose engine and model also have a `tuned` run in the scenario.
- Produces: `track_comparison_view(s: ScenarioResults) -> list[dict]`: one row per engine and model that has both tracks, with keys `engine, model, keyword_default, keyword_tuned, semantic_default, semantic_tuned, hybrid_default, hybrid_tuned` (nDCG@10, `None` when N/A), plus `default_setting` and `tuned_setting`: the values of the engine's `language_keys` (Task 5, read with `load_adapter_class`) found in each run manifest's `config.effective.index`, as compact JSON, or `"—"` when none is set. Empty when no engine has both.

- [ ] **Step 1: Write the failing tests.**
  - `test_default_runs_not_ranked`: a scenario with Elasticsearch default 0.70 and tuned 0.60 keyword: the best-score mark goes to another engine's tuned run, not to Elasticsearch default; the significance notes never compare against a default run.
  - `test_default_only_engine_still_ranked`: an engine with only a default run appears in the relevance table.
  - `test_track_comparison_section`: a scenario page with both tracks for an engine has "Built-in settings vs tuned" and one row per engine; a page with tuned runs only has no such section.
  - `test_configs_page_tuned_only`: the configurations page lists tuned configs only when both tracks are present.
  - `test_comparison_shows_settings`: a default manifest with `config.effective.index.analyzer = "cjk"` and a tuned one with `"kuromoji"` give a row with `default_setting == '"cjk"'` and `tuned_setting == '"kuromoji"'`.
  - `test_non_bge_model_renders`: a scenario whose runs all use `multilingual-e5-small` renders its page and the index page, with the model named in the scenario header.

- [ ] **Step 2: Run them.** Expected: FAIL.

- [ ] **Step 3: Implement.** Switch the relevance cells, perf and trade-off views, configs page and cross-scenario views from `s.runs` to `s.ranked_runs`; run pages keep every run. Section copy: "Each engine with only its built-in language setting (default) and with the best settings found, official language plugins allowed (tuned). nDCG@10." In `parity.mdx`, add after the Tracks list: "For `ecommerce/amazon-search-ja` and `ecommerce/amazon-search-es`, the default runs are published too, shown only in a default-vs-tuned table on their scenario pages, because how much setup a language needs is part of the result."

- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS; the report for the existing results folder renders unchanged (no scenario there has both tracks published).

- [ ] **Step 5: Commit.** `git commit -m "report: default runs out of rankings, default-vs-tuned table"`

### Task 9: docs and READMEs

**Files:**
- Create: `docs/scenarios/amazon-search-ja.mdx`, `docs/scenarios/amazon-search-es.mdx`
- Modify: `docs/docs.json`, each `engines/<engine>/README.md` (a "Languages" section)

- [ ] **Step 1: Write the two pages** from the spec: source and licence, counts (§3.2), fields, the per-engine language table for both tracks (§6.1, §6.2), the `accents` coverage (about 210 of 2,000 queries; replace with the measured n after Task 10), why there is no `typo` for Japanese, the unjudged-results caveat (about 28 judged products per query), and that semantic scores use a different model from the English scenarios and are not comparable with them.

- [ ] **Step 2: Write the README sections**: what the engine is given per language and track, and the Elasticsearch note that English scenarios do not use its `english` analyzer.

- [ ] **Step 3: Check** `docs/docs.json` is valid JSON and lists both pages. Commit: `git commit -m "docs: Japanese and Spanish Amazon search"`

---

## Runs (controller, after Task 9; not dispatched to implementers)

Benchmarks run for hours, one at a time, on a quiet machine.

### Task 10: prepare, embed, default runs

- [ ] `uv run irbench prepare` for both scenarios. Expected: about 339,059 jp and 259,973 es docs (fewer only by titleless products), 2,000 queries each; record the `accents` variant count.
- [ ] `uv run irbench embed --model multilingual-e5-small` for both.
- [ ] Default track for the 6 engines on both scenarios. Record indexing time and peak memory; set `index_timeout_s` in both scenario files to 3× the slowest measured build, and commit.
- [ ] Check every manifest: succeeded, 0 errors, no violations.

### Task 11: sweeps and tuned configs

- [ ] Sweep each engine per scenario with the scratch sweep tool used for the earlier tuned tracks. Plugin variants (kuromoji's default analyzer and `search` mode; PGroonga if Task 6 built it) are tried next to the built-in ones and re-index per variant.
- [ ] Write each `tuned:` block with numbers in its justifications. Where a plugin lost, keep the built-in setting and put the plugin's best score in the justification text so the scenario page can show it. `irbench lint` gives 0 findings.

### Task 12: tuned runs, publish both tracks, report

- [ ] Tuned track for the 6 engines on both scenarios; check every manifest.
- [ ] `irbench publish` both tracks for these two scenarios only.
- [ ] Rebuild the report, check both scenario pages and the configurations page in a browser, and update the results dashboard artifact with the two scenarios.
