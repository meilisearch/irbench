# Chinese Title Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ecommerce/title-search-zh` and `media/video-search-zh` (C-MTEB EcomRetrieval and VideoRetrieval plus Multi-CPR train queries), with one-judgment scoring, Chinese engine settings on both tracks, and runs kept internal.

**Architecture:** A new `cmteb` normaliser writes docs/queries/qrels for two datasets. The scenario gains `zh`, `headline_metric` and `publishable`; the score stage records the headline metric's per-task values and intervals for five metrics, which the report uses for its matrix, ties and a second table. Engine Chinese settings are config keys (the multilingual plan's mechanism); two plugin images (`compose.zh.yaml` for Elasticsearch and Postgres) carry the tuned-track plugins.

**Tech Stack:** Python 3.12, pydantic, pyarrow, ranx, Jinja2, Docker Compose, uv, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-31-irbench-chinese-title-search-design.md`

**Base:** this branch already merges the multilingual plan's Tasks 1–6, 8, 9 (`f5afeaa`): `multilingual-e5-small` with `doc_prefix`, `TrackConfig.compose`, `language_keys` + lint, the kuromoji and PGroonga images, `ScenarioResults.ranked_runs`.

## Global Constraints

- Dataset ids `cmteb-ecom`, `cmteb-video`; scenario ids `ecommerce/title-search-zh`, `media/video-search-zh`.
- Loader constants: seed `20261031`, train sample `3000`; query ids `<set>-dev-<qid>` / `<set>-train-<qid>` where `<set>` is `ecom` or `video`; groups `dev` / `train`.
- Pins (spec §3.1): HF revisions `687de13dc7294d6fd9be10c6945f9e8fec8166b9` (EcomRetrieval), `39c90699b034ec22ac45b3abf5b0bbb5ffd421f9` (EcomRetrieval-qrels), `58c2597a5943a2ba48f4668c3b90d796283c5639` (VideoRetrieval), `faa71382b6a29cf1778d1f436b963e75cb5b927c` (VideoRetrieval-qrels); Multi-CPR commit `a4e467182a3e2c110528a1575d79b33cc449d2c3`. sha256: ecom corpus `ad50bc86e74f45ac57ad5b2f475e564a6baef4b286d0af803ac673e26ba0e059`, queries `ddf99eba426a21182755979e5c8f3b0afe93808096437779a7ab2d6a316df686`, qrels `189c199c2a3c615f8ec476549039e00f8e49f18905ede786d4003237d71d2933`, train queries `2a36f72063a3be3dea04944fb0174ea82473d66eb585b4fa3bcacc8c94c13008`, train qrels `2b0404478e3994f5e57ee0634239fbd89bf32faff087990bd4aac945a5e9ba04`; video corpus `02c3000524f7ff40c2a1d5a209f5cde9b81c7f8eadee41f6311bf2b9eeb483c0`, queries `0f3df85f0550833d7f5a009b43da777edfb9a298327ee020e02d19f47a3b1173`, qrels `ce205fa71e98f9d42a7c438e762484ee167ea8a38c97ab2e6f6bdfd494336e95`, train queries `bbd72d6848065ce86422b0ec5d6d5cc3d92ed5e7e2e51f04dd1344a5d1ecc4a0`, train qrels `788b418dfdb1c61a635623c5c359f7530b87ca27672f8b3877da3966173e6da3`.
- Plugin pins: IK `https://get.infini.cloud/elasticsearch/analysis-ik/9.5.5`, sha256 `76d1437d1ad1d2e3b68e7fe9938ef38194e1d517ef3d150cf67f0e329633e051`; zhparser commit `2e995c4df672563992b4d7a147b8fa2d0d4cda6c`; SCWS commit `b5ae33f6a71ecbf5442a2e485703af4d372f3341`.
- No perturbations (`kinds: []`); `embedding_model: multilingual-e5-small`; `headline_metric: mrr@10`; `publishable: false`.
- Every existing scenario must render and score exactly as before (defaults `ndcg@10`, `publishable: true`).
- Commit messages: no `Co-Authored-By` line (user rule).

## Review Focus

1. A bundle scored before this change (no `headline_metric`, no `ci95`) must still load and render with nDCG@10 — Task 4 test `test_old_bundle_renders_ndcg`.
2. A scenario page mixing runs whose bundles disagree on `headline_metric` (should not happen, but a stale bundle can): use the metric of the majority and never crash — Task 4 test `test_mixed_headline_uses_majority`.
3. A train query whose judged title was merged as a duplicate must keep a judgment on the surviving id, and a query left with two judgments on one merged id counts once — Task 2 test `test_duplicate_merge_keeps_one_judgment`.
4. `irbench publish` on a run whose scenario file was since deleted must fail with a clear message, not a stack trace — Task 5 test `test_publish_missing_scenario`.
5. The sweep helper with a variant identical to the default (p = 1) or an empty variant set returns the default — Task 6 tests `test_identical_variant_keeps_default`, `test_no_variants_keeps_default`.

---

### Task 1: scenario fields (`zh`, `headline_metric`, `publishable`)

**Files:**
- Modify: `harness/irbench/scenario.py`
- Test: `harness/tests/test_scenario.py`

**Interfaces:**
- Produces: `Scenario.language: Literal["en", "ja", "es", "zh"]`; `Scenario.headline_metric: Literal["ndcg@10", "mrr@10"] = "ndcg@10"`; `Scenario.publishable: bool = True`.

- [ ] **Step 1: Write the failing tests**: `test_zh_language_accepted`; `test_zh_rejects_typo` (`pytest.raises(ValueError, match="'typo' is not supported for language 'zh'")`); `test_headline_metric_default_ndcg`; `test_headline_metric_rejects_other` (`"recall@10"` fails); `test_publishable_default_true`.
- [ ] **Step 2: Run** `uv run pytest harness/tests/test_scenario.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** The `typo` check applies to `ja` and `zh` with the same comment (input methods; `str.isalpha()` is true for Han characters).
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `harness: scenario language zh, headline metric, publishable flag`.

### Task 2: `cmteb` loader and the two datasets

**Files:**
- Create: `harness/irbench/datasets/cmteb.py`, `datasets/cmteb-ecom/dataset.yaml`, `datasets/cmteb-video/dataset.yaml`
- Modify: `harness/irbench/datasets/__init__.py` (register both ids)
- Test: `harness/tests/test_cmteb.py`

**Interfaces:**
- Produces: `normalize_cmteb(raw: dict[str, Path], out: Path, domain: Literal["ecom", "video"]) -> NormalizedDataset`. Raw keys: `corpus`, `queries`, `qrels` (parquet; columns `id,text` / `id,text` / `qid,pid,score`), `train_queries` (TSV `qid\ttext`), `train_qrels` (TSV `qid\t0\tpid\t1`). Writes `docs.jsonl` rows `{"id", "title"}`, `queries.jsonl` rows `{"qid", "text", "group"}`, `qrels.tsv` grade 1. `dropped` keys: `docs` (merged duplicates), `judgments` (0), `moved_judgments`.

- [ ] **Step 1: Write the failing tests** (fixture parquet/TSV built in `tmp_path` with pyarrow):
  - `test_dev_queries_all_kept`: 3 dev queries in → 3 rows with ids `ecom-dev-<qid>`, group `dev`.
  - `test_train_sample_only_eligible_and_stable`: with train queries whose judged pid is or is not in the corpus, only eligible ones appear, ids `ecom-train-<qid>`; the sample is identical across two calls; with fewer eligible than 3000, all are kept.
  - `test_duplicate_merge_keeps_one_judgment`: titles `1:"甲"`, `5:"甲"`; a judgment on `5` becomes one on `1`; `docs.jsonl` has no id `5`; `dropped["docs"] == 1`, `dropped["moved_judgments"] == 1`; two queries judged on `1` and `5` each keep exactly one judgment on `1`.
  - `test_text_kept_as_written`: title `"ＡＢＣ小米WiFi"` and query `" 老肉， "` come out as `"ＡＢＣ小米WiFi"` and `"老肉，"` (whitespace normalised only).
- [ ] **Step 2: Run** `uv run pytest harness/tests/test_cmteb.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** Sample with `random.Random(SEED).sample(sorted(eligible_qids, key=int), min(SAMPLE_SIZE, len(eligible)))`. Write the two `dataset.yaml` files with the five files each (HF `resolve/<revision>/data/...` URLs, Multi-CPR `raw.githubusercontent.com/Alibaba-NLP/Multi-CPR/<commit>/data/<domain>/...`), `params: {domain: ecom|video}`, `license: "not specified (Multi-CPR has no licence; internal only, see the Chinese title search spec §7)"`, `version` the five pins joined.
- [ ] **Step 4: Run** `uv run pytest harness/tests/test_cmteb.py harness/tests/test_prepare.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `datasets: cmteb-ecom and cmteb-video (C-MTEB titles, Multi-CPR train queries)`.

### Task 3: score stage records the headline metric and five intervals

**Files:**
- Modify: `harness/irbench/score/relevance.py`, `harness/irbench/stages/score_run.py`
- Test: `harness/tests/test_score_relevance.py`

**Interfaces:**
- Consumes: `Scenario.headline_metric` (Task 1).
- Produces: `GroupScore.ci95: dict[str, tuple[float, float]] | None` for `ndcg@10, mrr@10, success@1, recall@10, recall@100` (`ndcg_ci95` kept); `per_task_scores(tasks, qrels, run, metric: str) -> dict[str, float]` (`per_task_ndcg` stays as a wrapper); `scores.json` gains `headline_metric: str`, `single_judgment: bool`, and `per_task[<headline>]` next to `per_task["ndcg@10"]`.

- [ ] **Step 1: Write the failing tests**: `test_ci95_for_five_metrics` (a 3-task fixture: each key present, lo ≤ mean ≤ hi, the mean equal to `metrics[...]`); `test_per_task_mrr` (ranks 1, 3, absent → 1.0, 1/3, 0.0); `test_single_judgment_flag` (true when every task's qrels has exactly one entry, false otherwise; function `single_judgment(qrels, task_ids) -> bool`); a score-stage test that `scores.json` has `headline_metric == "mrr@10"` and `per_task["mrr@10"]` for a scenario fixture with that headline.
- [ ] **Step 2: Run** `uv run pytest harness/tests/test_score_relevance.py harness/tests/test_run_pipeline.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** ranx gives per-query scores in `ranx_run.scores[metric]` after `evaluate`; success@1 is ranx's `hit_rate@1`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `score: headline metric per task, intervals for five metrics, single-judgment flag`.

### Task 4: report uses the headline metric

**Files:**
- Modify: `harness/irbench/report/data.py`, `harness/irbench/report/render.py`, `harness/irbench/report/templates/scenario.html.j2`
- Test: `harness/tests/test_report_render.py`, `harness/tests/test_report_data.py`

**Interfaces:**
- Consumes: Task 3's `scores.json` keys.
- Produces: `RunResult.headline_metric -> str` (`scores.get("headline_metric", "ndcg@10")`); `RunResult.score(group, metric) -> float | None`; `RunResult.ci95(group, metric) -> tuple | None` (falls back to `ndcg_ci95` for nDCG); `RunResult.per_task(group, metric="ndcg@10")`; `RunResult.single_judgment -> bool`; `ScenarioResults.headline_metric -> str` (majority of its runs' values, ties to `ndcg@10`); `metrics_rows(s) -> list[dict]` for the second table.

- [ ] **Step 1: Write the failing tests**:
  - `test_matrix_uses_mrr_when_headline`: two runs with `headline_metric: mrr@10`, run A better on MRR but worse on nDCG → A is `best`; the heading reads `Relevance: MRR@10 by mode`.
  - `test_old_bundle_renders_ndcg`: a fixture bundle without the new keys renders as today (heading `nDCG@10`, same best mark).
  - `test_mixed_headline_uses_majority`.
  - `test_metrics_table_on_headline_pages`: success@1, recall@10, recall@100, nDCG@10 columns with `±` present only when the headline is not nDCG@10.
  - `test_judged_hidden_for_single_judgment`: the judged@10 table is absent and the sentence "Each query has one judged document, so judged@10 is at most 0.1 and is not shown." is present.
- [ ] **Step 2: Run** `uv run pytest harness/tests/test_report_render.py harness/tests/test_report_data.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** `relevance_cell` and `mark_winners` read the scenario's headline metric; the tie test uses `per_task(group, metric)`. Existing callers keep `ndcg(...)`.
- [ ] **Step 4: Run** `uv run pytest -q -k "report"`. Expected: PASS, existing report tests unchanged.
- [ ] **Step 5: Commit** `report: headline metric per scenario, metrics table, judged@10 hidden for one-judgment sets`.

### Task 5: publish refuses internal scenarios

**Files:**
- Modify: `harness/irbench/publish.py`
- Test: `harness/tests/test_publish.py`

**Interfaces:**
- Consumes: `Scenario.publishable` (Task 1); the manifest's `scenario` id.

- [ ] **Step 1: Write the failing tests**: `test_publish_refuses_internal_scenario` (`pytest.raises(ValueError, match="internal")`, message names the scenario and "spec §7"); `test_publish_missing_scenario` (`ValueError` naming the scenario id, no traceback from yaml or file access).
- [ ] **Step 2: Run** `uv run pytest harness/tests/test_publish.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the check after the profile check, before any copy.
- [ ] **Step 4: Run** the same. Expected: PASS.
- [ ] **Step 5: Commit** `publish: refuse runs of internal scenarios`.

### Task 6: sweep significance rule

**Files:**
- Create: `harness/irbench/sweep.py`
- Modify: `harness/irbench/report/stats.py` (add `holm`)
- Test: `harness/tests/test_sweep.py`

**Interfaces:**
- Produces: `holm(pvalues: dict[str, float]) -> dict[str, float]` (adjusted, capped at 1, monotone); `keep_variant(default: dict[str, float], variants: dict[str, dict[str, float]], alpha: float = ALPHA) -> tuple[str | None, dict[str, float]]` — returns the kept variant name (None = keep default) and the Holm-adjusted p per variant. A variant is eligible when its mean exceeds the default's and its adjusted p < alpha; the eligible one with the highest mean wins.

- [ ] **Step 1: Write the failing tests**: `test_holm_known_values` (p `{a: .01, b: .04, c: .03}` → `{a: .03, b: .06, c: .06}`); `test_identical_variant_keeps_default`; `test_no_variants_keeps_default`; `test_worse_significant_variant_not_kept`; `test_spec_worked_case` (standard ≫ cjk, smartcn > standard but smartcn vs cjk significant → `smartcn`; same with smartcn vs cjk not significant → `standard`), with per-query vectors of 400 tasks built from a seeded RNG.
- [ ] **Step 2: Run** `uv run pytest harness/tests/test_sweep.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** using `paired_randomization_p`.
- [ ] **Step 4: Run** the same. Expected: PASS.
- [ ] **Step 5: Commit** `sweep: keep a variant only when significantly better than the default (Holm)`.

### Task 7: Postgres zhparser configuration and PGroonga tokenizer

**Files:**
- Modify: `engines/postgres/adapter.py`
- Test: `harness/tests/test_language_settings.py`

**Interfaces:**
- Produces: `index.text_search_config: zhparser` → setup DDL creates the extension and `CREATE TEXT SEARCH CONFIGURATION <cfg> (PARSER = zhparser); ALTER TEXT SEARCH CONFIGURATION <cfg> ADD MAPPING FOR n,v,a,i,e,l WITH simple;` (no thesaurus; a synonym list with zhparser raises `ValueError`). `index.pgroonga_tokenizer: TokenBigram | TokenNgram` → `WITH (tokenizer='<name>')` on the PGroonga index; any other value raises.

- [ ] **Step 1: Write the failing tests**: `test_pg_zhparser_config` (DDL contains `CREATE EXTENSION IF NOT EXISTS zhparser`, `PARSER = zhparser`, `ADD MAPPING FOR n,v,a,i,e,l WITH simple`, no `COPY =`); `test_pg_zhparser_rejects_synonyms`; `test_pgroonga_tokenizer_option`; `test_pgroonga_tokenizer_rejects_unknown`; `test_pg_simple_config_unchanged` (`COPY = simple`).
- [ ] **Step 2: Run** `uv run pytest harness/tests/test_language_settings.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** Golden render fixtures for existing scenarios stay unchanged.
- [ ] **Step 4: Run** `uv run pytest harness/tests/test_language_settings.py harness/tests/test_adapter_postgres.py harness/tests/test_render_stage.py -q`. Expected: PASS; `git status harness/tests/fixtures` clean.
- [ ] **Step 5: Commit** `engines/postgres: zhparser configuration and PGroonga tokenizer option`.

### Task 8: Chinese plugin images

**Files:**
- Create: `engines/elasticsearch/docker/Dockerfile.zh`, `engines/elasticsearch/docker/compose.zh.yaml`, `engines/postgres/docker/Dockerfile.zh`, `engines/postgres/docker/compose.zh.yaml`
- Test: `harness/tests/test_zh_images.py` (marked `docker`)

**Interfaces:**
- Produces: local images `irbench-elasticsearch-zh:9.5.5` (smartcn, icu, IK) and `irbench-postgres-zh:0.8.7-pg17` (PGroonga 4.0.6-1, SCWS, zhparser), each compose file otherwise identical to its `compose.yaml` (as the kuromoji/pgroonga ones).

- [ ] **Step 1: Write the images.** Elasticsearch: `FROM elasticsearch:9.5.5`; `elasticsearch-plugin install --batch analysis-smartcn analysis-icu`; download IK with curl, check `sha256sum -c`, install from `file://`. Postgres: `FROM pgvector/pgvector:0.8.7-pg17`; PGroonga as in `Dockerfile.pgroonga`; build SCWS and zhparser at the pinned commits with `build-essential postgresql-server-dev-17 git autoconf automake libtool`, then purge the build packages. Header comments state the plugin rule (spec §5.2) and why each plugin qualifies.
- [ ] **Step 2: Docker tests** `test_es_zh_analyzers` (`_analyze` with `smartcn`, `icu_analyzer`, `ik_smart`, `ik_max_word` each gives ≥ 2 tokens for `纯山楂酱`) and `test_pg_zh_parsers` (`CREATE EXTENSION zhparser; CREATE EXTENSION pgroonga; CREATE EXTENSION vector;` succeed; a zhparser configuration gives ≥ 2 lexemes for `纯山楂酱`).
- [ ] **Step 3: Run** `uv run pytest -m docker harness/tests/test_zh_images.py -q` and `uv run irbench lint`. Expected: PASS; no `image-pin` finding.
- [ ] **Step 4: Commit** `engines: Chinese plugin images (smartcn, icu, IK; PGroonga, zhparser)`.

### Task 9: the two scenarios and their default configs

**Files:**
- Create: `scenarios/ecommerce/title-search-zh/scenario.yaml`, `scenarios/media/video-search-zh/scenario.yaml`, `engines/<engine>/configs/ecommerce/title-search-zh.yaml` and `engines/<engine>/configs/media/video-search-zh.yaml` for the 6 engines
- Test: `harness/tests/test_zh_defaults.py` (marked `docker`)

- [ ] **Step 1: Write the scenarios** as spec §4 (`index_timeout_s: 1800` until Task 11 measures), plus `headline_metric: mrr@10`, `publishable: false`.
- [ ] **Step 2: Write the 12 configs.** Copy each engine's `knowledge-base/web-questions.yaml` (single text field) and set the default block per spec §5.1, each with a `justification:` entry citing the default-track rule: Meilisearch `localizedAttributes: [{attributePatterns: ["*"], locales: ["cmn"]}]` and `locales: ["cmn"]` on keyword and hybrid search; Elasticsearch `index.analyzer: cjk`; Typesense `index.searchable_fields.locale: zh`; Qdrant `index.bm25_options: {tokenizer: multilingual}`; ParadeDB `index.tokenizer: pdb.jieba`; Postgres `index.text_search_config: simple`. Tuned blocks are written in Task 11.
- [ ] **Step 3: Docker test** `test_default_finds_title` per engine: index the one title `护肚子肚围婴儿冬季纯棉加厚绒肚兜宝宝保暖新生儿裹腹护肚脐神器` with the scenario's default config, keyword-search `宝宝护肚`, assert it is returned (Postgres: assert it is not). If an engine's setting does not exist as written (ParadeDB tokenizer name, Qdrant `multilingual`), fix the config to the engine's documented name or drop it, and record that in its README (Task 10).
- [ ] **Step 4: Run** `uv run irbench lint` (only `not-prepared` until Task 11) and `uv run pytest -m docker harness/tests/test_zh_defaults.py -q`.
- [ ] **Step 5: Commit** `scenarios: Chinese title and video search, default configs`.

### Task 10: docs, READMEs, parity rule

**Files:**
- Create: `docs/scenarios/title-search-zh.mdx`, `docs/scenarios/video-search-zh.mdx`
- Modify: `docs/docs.json`, `docs/methodology/parity.mdx`, `engines/*/README.md`

- [ ] **Step 1: Write the two pages** from the spec: source and licence (internal, §7), counts (§3.2, no query or title text beyond the spec's examples), one judgment per query and what that means (§6.1), the measured noise (§6.2, replaced by the real figure after Task 11), the per-engine table for both tracks (§5.1, §5.2), modes N/A, the model and why.
- [ ] **Step 2: Parity doc**: the tuned-track plugin rule with its four conditions (spec §5.2), and the significance rule for one-judgment scenarios.
- [ ] **Step 3: READMEs**: what each engine gets for `zh` per track, and what was not available.
- [ ] **Step 4: Check** `python -I -c "import json; json.load(open('docs/docs.json'))"` and both pages listed. Commit `docs: Chinese title and video search`.

### Task 11: prepare, embed, runs, sweeps

- [ ] `uv run irbench prepare` for both. Expected: 100,902 ecom docs and 100,579 video docs (after merges), 4,000 queries each; record moved judgments.
- [ ] `uv run irbench embed --model multilingual-e5-small` for both.
- [ ] Default track, 6 engines × 2 scenarios, sequentially. Check each manifest: succeeded, 0 errors. Set `index_timeout_s` to 3× the slowest build; commit.
- [ ] Sweeps per engine and scenario with `keep_variant` (Task 6): index-time variants on keyword, then search settings per base. Variants per spec §5.2.
- [ ] Write the `tuned:` blocks with gain, Holm-adjusted p and variants tried in each justification; tuned runs; `irbench lint` 0 findings beyond `not-prepared`.
- [ ] Rebuild the local report; update the docs pages' measured noise; commit `runs: Chinese title search tuned configs`.
