# Amazon ESCI and As-You-Type Scenarios Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ecommerce/amazon-search` (2,000 real Amazon queries) and `ecommerce/as-you-type` (7,429 real typed prefixes of the same queries) on the full 1.2M-product ESCI catalogue, run every engine on both, and publish the results.

**Architecture:** One dataset, `amazon-esci`, whose normaliser joins ESCI (products, queries, judgments) with one AmazonQAC train shard (typing sessions). It writes the complete-query set and a second query set, `prefixes`. Both scenarios share the documents and document vectors; a new `Scenario.query_set` picks the query set. The report learns a fixed group order and a completion line chart.

**Tech Stack:** Python 3.12, pydantic, pyarrow (already a dependency), Jinja2 report, Docker engines, uv, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-20-irbench-amazon-esci-design.md`

## Global Constraints

- Locale `us` only; grades Exact 2, Substitute 1, Complement 0, Irrelevant 0. Complement and Irrelevant stay in the qrels.
- Dataset seed `20261020`; complete-query sample size 2,000; loader constants, not run settings.
- Prefix text is sent exactly as typed, trailing space included. Complete queries are whitespace-normalised.
- Prefix groups, in this order: `≤25%`, `25–50%`, `50–75%`, `75–<100%`, `complete`, `diverges`.
- Pins: ESCI commit `7916cdf6ab75a462e77f20ab40428a10923998d5`, AmazonQAC revision `975cbfac4623ffb054c59631e06e39d0831e47fb`.
- sha256: `shopping_queries_dataset_examples.parquet` `4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a`; `shopping_queries_dataset_products.parquet` `25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265`; `shopping_queries_dataset_sources.csv` `a5fed8ecc016443de40bf3c63098f0e3f23bbe4daa4236f1c38b8c3184778c50`; AmazonQAC `train/part-00000-1d4931ef-16a0-4dc1-8034-ddaa3f7ddec6-c000.snappy.parquet` `85ea09504cec52c80bbc84912ae0c1f3c4e274fd55581c2023b573ef6eec780a`.
- No new dependency. HTML stripping uses the standard library.
- Never add `Co-Authored-By` lines to commits.
- `uv run pytest -q`, `uv run ruff check .` and `uv run irbench lint` must pass after every task.

## Review Focus

1. A product whose description is HTML-only or entity-only (`<p>&nbsp;</p>`) must end up with an empty description, never a stray `&nbsp;` or tag text. Task 1 tests it.
2. A typing session whose prefixes repeat (users delete and retype) must not give two prefixes in one group, and a prefix equal to the final query must count as `complete`, not `75–<100%`. Task 1 tests both.
3. ESCI queries that differ only by case or spacing ("iPhone  case" vs "iphone case") must map to one normalised key. If two such queries are in the pool, the one with the smaller `query_id` wins, so ids stay unique. Task 1 tests it.
4. A scenario with `perturbations.kinds: []` must produce exactly one task per query and mode, with no crash in variant generation or the perturbation report table. Task 2 tests it.
5. A query whose text ends with a space must reach every engine's request with the space intact, or the engine's documented tokenizer drops it. Task 4 tests every adapter.

---

### Task 1: `amazon-esci` dataset and normaliser

**Files:**
- Create: `datasets/amazon-esci/dataset.yaml`
- Create: `harness/irbench/datasets/amazon_esci.py`
- Modify: `harness/irbench/datasets/__init__.py` (register `"amazon-esci": normalize_amazon_esci`)
- Modify: `harness/irbench/datasets/registry.py` (`DatasetSpec.params`)
- Modify: `harness/irbench/datasets/wands.py` (`NormalizedDataset.query_sets`)
- Modify: `harness/irbench/stages/prepare.py` (pass `**spec.params` to the normaliser)
- Test: `harness/tests/test_amazon_esci.py` (fixture parquet files built in the test with pyarrow, in `tmp_path`)

**Interfaces:**
- Produces: `DatasetSpec.params: dict[str, Any] = {}`.
- Produces: `NormalizedDataset.query_sets: dict[str, tuple[Path, Path]] = {}`, mapping set name to `(queries path, qrels path)`.
- Produces: `normalize_amazon_esci(raw: dict[str, Path], out: Path, corpus: Literal["full", "pool"] = "full") -> NormalizedDataset`. Raw keys: `examples`, `products`, `sources`, `qac`.
- Produces: files `docs.jsonl`, `queries.jsonl`, `qrels.tsv`, `queries.prefixes.jsonl`, `qrels.prefixes.tsv` in `out`.
- Produces: doc fields `id, title, brand, color, bullet_points, description`. Query rows `{"qid", "text", "group"}`.

- [ ] **Step 1: Write `dataset.yaml`.** `id: amazon-esci`, `version: "7916cdf6ab75a462e77f20ab40428a10923998d5"`, `license: "Apache-2.0 (ESCI); CDLA-Permissive-2.0 (AmazonQAC)"`, `source: https://github.com/amazon-science/esci-data`, `params: {corpus: full}`. Files, with the sha256 values from Global Constraints:
  - `examples` and `products`: `https://media.githubusercontent.com/media/amazon-science/esci-data/<commit>/shopping_queries_dataset/<file>`;
  - `sources`: `https://raw.githubusercontent.com/amazon-science/esci-data/<commit>/shopping_queries_dataset/shopping_queries_dataset_sources.csv`;
  - `qac`: `https://huggingface.co/datasets/amazon/AmazonQAC/resolve/975cbfac4623ffb054c59631e06e39d0831e47fb/train/part-00000-1d4931ef-16a0-4dc1-8034-ddaa3f7ddec6-c000.snappy.parquet`.

- [ ] **Step 2: Write the failing tests** in `test_amazon_esci.py`. Each builds tiny parquet and csv inputs with the real column names (spec §3.1, §3.2) and calls `normalize_amazon_esci`:
  - `test_us_only_and_full_corpus`: an `es` product and its judgments are absent; with `corpus="full"` an unjudged US product is in `docs.jsonl`; with `corpus="pool"` it is not.
  - `test_doc_fields`:
    - `<p>Soft <b>cotton</b>&amp;wool</p>` becomes `"Soft cotton&wool"`;
    - `<p>&nbsp;</p>` becomes `None`;
    - a missing brand is `None`;
    - a product with an empty title is dropped, its judgments are dropped, and `dropped == {"docs": 1, "judgments": n}`.
  - `test_grades`: E→2, S→1, C→0, I→0 in `qrels.tsv`. A query with only C/I judgments is not in `queries.jsonl`.
  - `test_pool_and_sample`:
    - only small-version test queries that have a QAC session are sampled;
    - with 2,500 eligible fixture queries, exactly 2,000 come out;
    - two runs give identical files;
    - qid is `esci-<query_id>`;
    - `group` is the `sources.csv` source.
  - `test_case_spacing_duplicates`: `"iPhone  case"` (id 7) and `"iphone case"` (id 3) are both eligible; only `esci-3` is kept.
  - `test_prefix_groups`: for final query `"yeti mug"` with session prefixes `["y", "ye", "yet", "yeti", "yeti ", "yeti m", "yeti m", "yeti mug", "yettim"]`, the prefix set holds at most one row per group. Groups and texts are drawn from `≤25%`: `y`, `ye`; `25–50%`: `yet`, `yeti`; `50–75%`: `yeti `, `yeti m`; `75–<100%`: none (no literal prefix of 7 characters); `complete`: `yeti mug`; `diverges`: `yettim`. Ids are `esci-<id>-<group>`, and the trailing space of `"yeti "` is kept if it is chosen.
  - `test_prefix_qrels`: every prefix row's qrels equal its final query's qrels.
  - `test_spec_params`: `DatasetSpec` accepts `params`, and `prepare` passes them (use the existing `test_prepare.py` fake normaliser pattern).

- [ ] **Step 3: Run the tests to verify they fail.** `uv run pytest harness/tests/test_amazon_esci.py -q`. Expected: import errors.

- [ ] **Step 4: Implement.**
  - Read parquet with `pyarrow.parquet`, column-projected. Stream the QAC shard with `ParquetFile.iter_batches(columns=["prefixes", "final_search_term"], batch_size=500_000)`, keeping only rows whose normalised term is in the pool keys.
  - Normalised key: `" ".join(text.lower().split())`.
  - Choose sessions and prefixes with one `random.Random(20261020)`, iterating the pool in sorted qid order so the output is stable.
  - Group rule (spec §3.2), with `final` the normalised ESCI query:
    - `diverges` if `not final.startswith(prefix.lower())`;
    - else skipped if `prefix.strip().lower() == final`, because the `complete` row is added from the ESCI query text itself, once per query;
    - else by `len(prefix) / len(final)` with upper bounds 0.25, 0.5, 0.75, and `75–<100%` for the rest.
  - HTML: an `html.parser.HTMLParser` subclass collecting data, then `html.unescape`, whitespace collapsed, empty → `None`.

- [ ] **Step 5: Run the tests.** `uv run pytest harness/tests/test_amazon_esci.py harness/tests/test_prepare.py -q` passes. Ruff is clean.

- [ ] **Step 6: Commit.** `git commit -m "datasets: amazon-esci (ESCI products and judgments, AmazonQAC prefixes)"`.

### Task 2: query sets, group order, empty perturbation kinds

**Files:**
- Modify: `harness/irbench/scenario.py` (`query_set`, `group_order`)
- Modify: `harness/irbench/stages/prepare.py` (read the named query set; manifest `query_set`)
- Modify: `harness/irbench/stages/score_run.py` (write `group_order` into `scores.json` when set)
- Test: `harness/tests/test_scenario.py`, `harness/tests/test_prepare.py`, `harness/tests/test_score_relevance.py` (or the test that covers `score_run`)

**Interfaces:**
- Consumes: `NormalizedDataset.query_sets` (Task 1).
- Produces: `Scenario.query_set: str | None = None` and `Scenario.group_order: list[str] | None = None`.
- Produces: `scores.json` key `"group_order": list[str]`, present only when the scenario sets it.

- [ ] **Step 1: Write the failing tests.**
  - `test_query_set_selects_files`: a fake normaliser returns `query_sets={"prefixes": (q, r)}`. A scenario with `query_set: prefixes` yields tasks from `q` and qrels from `r`, and the manifest records `"query_set": "prefixes"`.
  - `test_unknown_query_set`: `query_set: nope` raises `ValueError` naming `nope` and the available sets.
  - `test_empty_kinds`: `perturbations: {seed: 1, kinds: []}` gives exactly one task per query and mode, all with `perturbation == "original"`.
  - `test_group_order_in_scores`: scoring a run of a scenario with `group_order: [b, a]` writes `"group_order": ["b", "a"]` to `scores.json`. Without it, the key is absent.

- [ ] **Step 2: Run the tests to verify they fail.** Expected: validation errors on unknown fields, and assertion failures.

- [ ] **Step 3: Implement.** `prepare` uses `normalized.query_sets[scenario.query_set]` when set, else `(normalized.queries, normalized.qrels)`. If `make_variants` or the report perturbation table fails on empty kinds, fix them here.

- [ ] **Step 4: Run** `uv run pytest -q`. Everything passes, and the WANDS golden tests are unchanged.

- [ ] **Step 5: Commit.** `git commit -m "harness: scenario query sets, group order, scenarios without perturbations"`.

### Task 3: the two scenarios and the shared-dataset lint check

**Files:**
- Create: `scenarios/ecommerce/amazon-search/scenario.yaml` (spec §4.1, verbatim, plus `id_field: id`, `request_timeout_s: 10`, `max_error_rate: 0.001`, `max_facet_values: 1000`, `facet_verify_limit: 1000`, and the WANDS perf protocol apart from `page_sizes`)
- Create: `scenarios/ecommerce/as-you-type/scenario.yaml` (spec §4.2, with `index_fields` and `embedding_fields` written out identical to amazon-search, and `group_order` from Global Constraints)
- Modify: `harness/irbench/lint.py`
- Test: `harness/tests/test_lint.py`

**Interfaces:**
- Consumes: `Scenario.query_set`, `Scenario.group_order` (Task 2).
- Produces: a lint finding when two scenarios with the same `dataset` differ in `index_fields`, `embedding_fields` or `id_field`. Also a finding when an engine's configs for two such scenarios differ once parsed (YAML comments ignored).

- [ ] **Step 1: Write the failing tests.**
  - `test_shared_dataset_fields_must_match`: two scenario files on one dataset with different `searchable` give one finding naming both paths.
  - `test_shared_dataset_engine_configs_must_match`: the same engine config contents with different comments pass. A differing `tuned.search` gives a finding naming the engine and both scenario ids.

- [ ] **Step 2: Run the tests to verify they fail.**

- [ ] **Step 3: Implement the check in `lint_repo`.** Write both scenario files.

- [ ] **Step 4: Run.** `uv run pytest harness/tests/test_lint.py -q` and `uv run python -c "from pathlib import Path; from irbench.scenario import load_scenario; [load_scenario(Path('.'), s) for s in ('ecommerce/amazon-search','ecommerce/as-you-type')]"` both succeed. `irbench lint` now reports the missing engine configs, which Task 5 adds; that finding is expected until then.

- [ ] **Step 5: Commit.** `git commit -m "scenarios: ecommerce/amazon-search and ecommerce/as-you-type; lint shared-dataset scenarios"`.

### Task 4: query text reaches every engine as typed

**Files:**
- Test: `harness/tests/test_trailing_space.py`
- Modify: only an adapter that strips the text, if the test shows one
- Modify: `engines/postgres/README.md`, `engines/qdrant/README.md` (a "Query text" line on how their tokenizer treats a trailing space)

**Interfaces:**
- Consumes: each adapter's `render(task, vector, top_k)`. Use `harness/tests/adapter_helpers.py` to build adapters for a scenario.

- [ ] **Step 1: Write the test.** It is parametrized over the six engines and renders keyword and hybrid tasks with text `"yeti m "`:
  - for HTTP engines, the request body JSON contains the string `"yeti m "`;
  - for the two Postgres engines, the SQL parameters contain `"yeti m "`.

- [ ] **Step 2: Run it.** If an adapter fails, change it to pass the text unchanged, and run that adapter's golden tests.

- [ ] **Step 3: Write the README lines.** Postgres `websearch_to_tsquery` and Qdrant's BM25 tokenizer ignore a trailing space: the last word is matched whole either way. That is engine behaviour, measured as such.

- [ ] **Step 4: Commit.** `git commit -m "adapters: pin that query text reaches each engine as typed"`.

### Task 5: engine configs, default track (12 files)

**Files:**
- Create: `engines/<engine>/configs/ecommerce/amazon-search.yaml` and `engines/<engine>/configs/ecommerce/as-you-type.yaml` for `meilisearch, elasticsearch, typesense, qdrant, postgres, postgres-paradedb`

**Interfaces:**
- Consumes: the scenarios (Task 3), and the lint shared-dataset check (the as-you-type file must parse equal to amazon-search).

- [ ] **Step 1: Write the configs.** Copy each engine's `ecommerce/product-search.yaml` header and default block. Drop settings that exist only for modes this scenario lacks: Meilisearch's `sort`-first `rankingRules` (there is no `sorted` mode), and any synonym or sort-only entries. Keep corpus-size settings (HNSW, `ef_search`) at their WANDS values. Write no `tuned:` block yet; Task 9 adds it.

- [ ] **Step 2: Run** `uv run irbench lint`. Expected: 0 findings, with the configs count up by 12.

- [ ] **Step 3: Commit.** `git commit -m "engines: default configs for the Amazon scenarios"`.

### Task 6: report, group order and completion chart

**Files:**
- Modify: `harness/irbench/report/data.py` (`RunResult.group_order`, read from `scores.json`)
- Modify: `harness/irbench/report/render.py` (`query_group_tables` ordering; `completion_view`)
- Modify: `harness/irbench/report/charts.py` (`category_line_chart`)
- Modify: `harness/irbench/report/templates/scenario.html.j2` (a "While typing" section)
- Test: `harness/tests/test_report_charts.py`, `harness/tests/test_report_render.py`

**Interfaces:**
- Consumes: `scores.json` `group_order` (Task 2).
- Produces: `category_line_chart(title: str, categories: list[str], series: list[LineSeries]) -> Markup | None`, with `LineSeries(label: str, slot: int, values: list[float | None])`. It draws a line through the ordered categories, leaves a gap at `None`, uses a y axis from 0 to `min(1, nice_max(max * 1.1))`, and labels each line at its right end.
- Produces: `completion_view(s: ScenarioResults) -> list[dict]`, one chart per retrieval base. Categories are `group_order` minus `diverges`; `diverges` is listed in a small table under the chart. Empty when no run has `group_order`.

- [ ] **Step 1: Write the failing tests.**
  - `test_group_tables_follow_group_order`: with `group_order: ["≤25%", "complete"]` the table names come out in that order, not alphabetically. Without it, the order stays sorted.
  - `test_category_line_chart`: three categories and two series render two `<path>` elements and three x labels. A `None` value splits the path. An empty series list returns `None`.
  - `test_completion_section`: a scenario page for runs with `group_order` contains "While typing" and one chart per base. A page without it has no such section.

- [ ] **Step 2: Run the tests to verify they fail.**

- [ ] **Step 3: Implement.** Section copy: "nDCG@10 by how much of the query had been typed, over the same 2,000 queries; `complete` is the full query. Queries typed with a mistake (`diverges`) are listed below the chart."

- [ ] **Step 4: Run.** `uv run pytest -q` passes. Ruff is clean.

- [ ] **Step 5: Commit.** `git commit -m "report: query groups in scenario order, while-typing chart"`.

### Task 7: docs

**Files:**
- Create: `docs/scenarios/amazon-search.mdx`, `docs/scenarios/as-you-type.mdx`
- Modify: `docs/docs.json` (navigation), `docs/superpowers/specs/2026-10-09-irbench-design.md` (Later list: Amazon ESCI done, see the 2026-10-20 spec)

- [ ] **Step 1: Write the two pages,** modelled on `docs/scenarios/covid-papers.mdx`:
  - sources and licenses;
  - corpus size and fields;
  - how prefixes are chosen and scored (against the final query's judgments);
  - the groups;
  - the unjudged-results caveat (about 16 judged products per query, all from Amazon's own ranking; judged@10 is shown);
  - the fit-test rule (§7.1).

- [ ] **Step 2: Check** that `docs/docs.json` is valid JSON and lists both pages.

- [ ] **Step 3: Commit.** `git commit -m "docs: Amazon search and as-you-type scenarios"`.

---

## Runs (controller, after Task 7; not dispatched to implementers)

These steps run benchmarks for hours and must run one at a time on a quiet machine. The controller runs them and records each outcome in the ledger.

### Task 8: prepare, embed, fit test

- [ ] `uv run irbench prepare --scenario ecommerce/amazon-search` and `... --scenario ecommerce/as-you-type`. Expected counts:
  - 1,215,854 docs (fewer only by dropped titleless products);
  - 2,000 complete queries;
  - 7,429 prefix queries, within a few, from the shard measurement.
- [ ] `uv run irbench embed --scenario ecommerce/amazon-search --model bge-small-en-v1.5`, then the same for as-you-type (doc vectors are shared; only the query vectors are new). This takes about 4 hours.
- [ ] Fit test (spec §7.1): for each engine, default track, `irbench run --scenario ecommerce/amazon-search --engine <e> --model bge-small-en-v1.5 --track default`. Sample `docker stats` every 10 s into `scratchpad/esci/fit-<engine>.log`, and record the peak and the build time.
- [ ] If an engine fails to fit:
  - first raise its memory settings within the VM, with a justification, and retry once;
  - if it still fails, set `params: {corpus: pool}` in `dataset.yaml`, re-prepare and re-embed, and run all engines on the fallback corpus;
  - write the failure into the scenario docs page.

### Task 9: sweeps and tuned configs

- [ ] Sweep each engine on `ecommerce/amazon-search` with the scratchpad `gsweep.py`, using the grids of the knowledge-base sweeps adapted to these fields. Keyword and hybrid run on the default index; index-level variants are added only where an engine's earlier sweeps showed index settings matter (Qdrant BM25).
- [ ] Write each engine's `tuned:` block in both Amazon configs (identical files apart from comments), with justifications carrying the numbers. `irbench lint` gives 0 findings.

### Task 10: tuned runs, report, write-ups

- [ ] Run the tuned track for all 6 engines on both scenarios, sequentially. Check every manifest: succeeded, 0 errors, no violations.
- [ ] Rebuild the report from the tuned runs only, check the while-typing chart and the configs page in a browser, and publish to `https://claude.ai/artifact/Dje2agmosdc6a65QUmcvcH`.
- [ ] Add the Amazon scenarios to the prototype report only if the prototype is re-run on them; otherwise leave it as is and say so.
- [ ] Push `plan3-loadgen`.
