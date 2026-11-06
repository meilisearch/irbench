# DBpedia-Entity Entity Search Scenario Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `entity-search/dbpedia`: 467 DBpedia-Entity v2 queries against a 1M-entity sample of DBpedia (every judged entity plus seeded random distractors), with configs for all 6 engines, then run, tune and report it.

**Architecture:** One new dataset, `dbpedia-entity`, whose normaliser streams the 1.8 GB BEIR corpus twice: once to count non-judged entities, once to write the judged ones plus a seeded draw of distractors. It reuses the BEIR helpers in `harness/irbench/datasets/beir.py`. A small harness change lets a normaliser report sampling stats into `prepare.manifest.json`. Everything else (query groups, three-mode scenarios, HF-pinned files) already exists.

**Tech Stack:** Python 3.12, pydantic, standard library only for the loader (`hashlib`, `random`, `json`), uv, pytest, ruff, Mintlify docs.

**Spec:** `docs/superpowers/specs/2026-10-31-irbench-dbpedia-entity-design.md`

## Global Constraints

- Source: `huggingface.co/datasets/mteb/dbpedia`, revision `7d8780d8ef0d0a55d13fa797fe39224ec806592c`. Files and sha256:
  - `corpus.jsonl` `218fd9274382857e1424ae3f3ad664424636766df2673ec0caf75830c60fb1ad`
  - `queries.jsonl` `7573758032694ac960d59b3c59746cf7ff49a01dfc32f12a47af87181640482f`
  - `qrels/test.tsv` `7939c61fa8f0f99c43bd04e2c118bd3d97ec7ee4d67d05c63d14fde103bedc96`
  - `qrels/dev.tsv` `316a5c4bab1356cd49dca79175e15f0bbd16cb028063371fe08a69a136505a73`
- License line: `MIT (mteb/dbpedia card and github.com/iai-group/DBpedia-Entity); entity text from DBpedia 2015-10 / Wikipedia, CC BY-SA 3.0`.
- Corpus seed `20261031`, a loader constant `SEED`. Corpus size `1000000`, a `dataset.yaml` param `corpus_size`; fallback `500000` (spec §7.1).
- The draw, exactly: number the non-judged entities 0 … n−1 in corpus file order; keep those whose number is in `random.Random(SEED).sample(range(n), corpus_size - n_judged)`. Documents are written in corpus file order.
- Document id: `"dbp-" + hashlib.sha1(uri.encode("utf-8")).hexdigest()[:16]`; `uri` kept as a stored, non-indexed field.
- Query groups from the `_id` prefix (text before the last `-`): `SemSearch_ES` → `SemSearch ES`; `INEX_LD` → `INEX-LD`; `INEX_XER`, `SemSearch_LS`, `TREC_Entity` → `ListSearch`; `QALD2_tr`, `QALD2_te` → `QALD2`. All 467 queries kept (user decision, 2026-11-01).
- Grades as published (0, 1, 2); grade 0 stays in the qrels.
- Scenario: `entity-search/dbpedia`, modes `[keyword, semantic, hybrid]`, searchable `[title, text]`, embedding fields `[title, text]`, perturbations `[typo, prefix]` with seed `20261031`, `index_timeout_s: 7200`, `page_sizes: [20, 100]`.
- No new dependency. Never add `Co-Authored-By` lines to commits.
- `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check harness engines` and `uv run irbench lint` must pass after every task (lint after Task 2 needs the scenario prepared locally).

## Review Focus

1. An entity whose URI has non-ASCII characters (`<dbpedia:Zürich>`, `<dbpedia:1908–09_Birmingham_F.C._season>`; 2,555 judged URIs are non-ASCII) must get the same id in `docs.jsonl` and `qrels.tsv`. Task 1 tests it.
2. A `corpus_size` smaller than the number of judged entities must fail with a message naming both numbers, not silently keep only some judged entities. Task 1 tests it.
3. A `corpus_size` larger than the corpus must fail, not loop or return fewer documents than asked. A `corpus_size` equal to the corpus keeps every entity. Task 1 tests both.
4. Query text with doubled spaces (`July, 1850  president died …`, a real query) must be whitespace-normalised, and a query whose `_id` prefix is not in the group map must fail the loader. Task 1 tests both.
5. The judged-entity set must come from both qrels files: an entity judged only in `dev.tsv` must be kept and never counted as a distractor. Task 1 tests it.

---

### Task 1: `dbpedia-entity` dataset and normaliser

**Files:**
- Create: `datasets/dbpedia-entity/dataset.yaml`
- Create: `harness/irbench/datasets/dbpedia_entity.py`
- Modify: `harness/irbench/datasets/__init__.py` (register `"dbpedia-entity": normalize_dbpedia_entity`)
- Modify: `harness/irbench/datasets/wands.py` (`NormalizedDataset.stats`)
- Modify: `harness/irbench/stages/prepare.py` (write `params` and `stats` into the manifest's `dataset` block)
- Test: `harness/tests/test_dbpedia_entity.py`; `harness/tests/test_prepare.py` (manifest)

**Interfaces:**
- Consumes: `beir.read_jsonl_rows(path)`, `beir.read_qrels_tsv(path) -> Qrels`, `beir.write_normalized(out_dir, docs, queries, qrels, text_fields) -> NormalizedDataset`.
- Produces: `NormalizedDataset.stats: dict[str, int] = {}` (defaults empty, so other loaders are unchanged).
- Produces: `normalize_dbpedia_entity(raw: dict[str, Path], out: Path, corpus_size: int = 1_000_000) -> NormalizedDataset`. Raw keys: `corpus`, `queries`, `qrels_test`, `qrels_dev`.
- Produces: `entity_id(uri: str) -> str` and constants `SEED = 20261031`, `GROUPS: dict[str, str]` (prefix → group, per Global Constraints).
- Produces: doc rows `{"id", "title", "text", "uri"}`; query rows `{"qid", "text", "group"}`; `stats = {"corpus_size", "corpus_seed", "judged_docs", "distractor_docs", "source_docs"}`.
- Produces: `prepare.manifest.json` `dataset` block gains `"params": spec.params` and `"stats": normalized.stats`.

- [ ] **Step 1: Write `dataset.yaml`.** `id: dbpedia-entity`, `version: "mteb-dbpedia@7d8780d8ef0d0a55d13fa797fe39224ec806592c"`, the license line and sha256 values from Global Constraints, `source: https://github.com/iai-group/DBpedia-Entity`, `params: {corpus_size: 1000000}`. Files `corpus`, `queries`, `qrels_test`, `qrels_dev`, each `https://huggingface.co/datasets/mteb/dbpedia/resolve/<revision>/<path>`. The two qrels files are named `test.tsv` and `dev.tsv` on disk, so they do not collide in the raw directory.

- [ ] **Step 2: Write the failing tests** in `test_dbpedia_entity.py`, on fixture files written in `tmp_path` with the real formats (`{"_id", "title", "text"}` corpus rows, `{"_id", "text"}` query rows, `query-id\tcorpus-id\tscore` qrels with a header). The fixture: 30 entities with URIs `<dbpedia:E00>` … `<dbpedia:E29>` plus `<dbpedia:Zürich>`; queries `SemSearch_ES-1`, `INEX_LD-2`, `INEX_XER-3`, `SemSearch_LS-4`, `TREC_Entity-5`, `QALD2_tr-6`, `QALD2_te-7` (dev); judged entities E00–E05 in `test.tsv` with grades 0, 1, 2, and E06 plus `Zürich` in `dev.tsv` only.
  - `test_keeps_every_judged_entity_and_fills_to_size`: with `corpus_size=12`, `docs.jsonl` has 12 rows; all 8 judged ids are present (grade 0 included); `stats == {"corpus_size": 12, "corpus_seed": 20261031, "judged_docs": 8, "distractor_docs": 4, "source_docs": 31}`.
  - `test_sample_is_seeded_and_in_file_order`: two runs give identical `docs.jsonl`; the ids appear in fixture file order; the 4 distractors equal those chosen by `random.Random(20261031).sample(range(23), 4)` over the non-judged entities numbered in file order.
  - `test_dev_only_judgment_is_judged`: E06 is kept in every run, and is not among the 4 distractors (Review Focus 5).
  - `test_ids_hash_the_uri`: `entity_id("<dbpedia:Zürich>") == "dbp-" + sha1("<dbpedia:Zürich>".encode()).hexdigest()[:16]`; the `Zürich` doc's `id` equals its qrels id and its `uri` field is `<dbpedia:Zürich>` (Review Focus 1).
  - `test_id_collision_fails`: monkeypatch `entity_id` to return a constant; the loader raises `ValueError` mentioning "collide".
  - `test_groups_and_text`: groups are `SemSearch ES`, `INEX-LD`, `ListSearch` ×3, `QALD2` ×2; a query text `"July, 1850  president  died"` comes out `"July, 1850 president died"`; 7 queries written.
  - `test_unknown_group_prefix_fails`: a query `Foo_Bar-1` raises `ValueError` naming `Foo_Bar` (Review Focus 4).
  - `test_qrels_merge_test_and_dev`: `qrels.tsv` holds the judgments of both files, with hashed ids and grades 0, 1, 2 unchanged.
  - `test_missing_judged_entity_fails`: a qrels row for `<dbpedia:Absent>` raises `ValueError` naming the URI.
  - `test_size_below_judged_fails`: `corpus_size=5` raises `ValueError` whose message contains `5` and `8` (Review Focus 2).
  - `test_size_bounds`: `corpus_size=31` keeps all 31 entities; `corpus_size=32` raises `ValueError` (Review Focus 3).

  In `test_prepare.py`, add `test_manifest_records_dataset_params_and_stats`, in the style of the existing tests there that `monkeypatch.setitem(NORMALIZERS, "wands", ...)`: wrap the WANDS normaliser so it returns `dataclasses.replace(result, stats={"judged_docs": 1})`; after prepare, `manifest["dataset"]["stats"] == {"judged_docs": 1}` and `manifest["dataset"]["params"] == {}` (WANDS has no params). A plain WANDS prepare writes `"stats": {}`.

- [ ] **Step 3: Run the tests to verify they fail.** `uv run pytest harness/tests/test_dbpedia_entity.py harness/tests/test_prepare.py -q`. Expected: import error for `irbench.datasets.dbpedia_entity`, and the manifest test failing on a missing key.

- [ ] **Step 4: Implement.**
  - `NormalizedDataset.stats: dict[str, int] = field(default_factory=dict)`; prepare adds `"params": spec.params, "stats": normalized.stats` to the manifest's `dataset` dict.
  - `normalize_dbpedia_entity`: read both qrels with `read_qrels_tsv` (URIs as keys), build the judged-URI set, then pass 1 over `read_jsonl_rows(raw["corpus"])` counts entities and non-judged entities and checks every judged URI was seen. Validate the size, draw the distractor numbers into a `set[int]` with the exact call from Global Constraints, then pass 2 yields doc rows lazily into `write_normalized(out, docs, queries, qrels, ["title", "text"])`, so the corpus is never held in memory. Keep a `dict[str, str]` from id to URI only to detect collisions. Map qrels URIs to ids. Return the `NormalizedDataset` from `write_normalized` with `stats` set (`dataclasses.replace`).
  - A module docstring citing the spec sections (§3.3, §3.4), in the style of `amazon_esci.py`.
  - Register in `__init__.py`.

- [ ] **Step 5: Run the tests to verify they pass.** `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check harness engines`. Expected: all pass, and the existing loader tests are unchanged.

- [ ] **Step 6: Commit.**

```bash
git add datasets/dbpedia-entity harness/irbench/datasets harness/irbench/stages/prepare.py harness/tests/test_dbpedia_entity.py harness/tests/test_prepare.py
git commit -m "datasets: dbpedia-entity (DBpedia-Entity v2, 1M-entity seeded sample)"
```

### Task 2: scenario, engine configs and CI

**Files:**
- Create: `scenarios/entity-search/dbpedia/scenario.yaml`
- Create: `engines/{meilisearch,elasticsearch,typesense,qdrant,postgres,postgres-paradedb}/configs/entity-search/dbpedia.yaml` (6 files)
- Modify: `.github/workflows/ci.yml` (parity job: `uv run irbench prepare --scenario entity-search/dbpedia` before `irbench lint`)
- Test: `harness/tests/test_scenario.py`

**Interfaces:**
- Consumes: the `dbpedia-entity` dataset (Task 1).
- Produces: scenario id `entity-search/dbpedia`, used by every later task.

- [ ] **Step 1: Write the failing test** in `test_scenario.py`:

```python
def test_entity_search_scenario_loads():
    s = load_scenario(REPO_ROOT, "entity-search/dbpedia")
    assert s.dataset == "dbpedia-entity"
    assert s.index_fields.searchable == ["title", "text"]
    assert s.modes == ["keyword", "semantic", "hybrid"] and s.structured_bases == []
    assert s.embedding_fields == ["title", "text"]
    assert s.perturbations.kinds == ["typo", "prefix"] and s.perturbations.seed == 20261031
    assert s.synonyms is None and s.index_timeout_s == 7200
    assert s.page_sizes == [20, 100] and s.top_k == 100
```

- [ ] **Step 2: Run it to verify it fails.** `uv run pytest harness/tests/test_scenario.py -q -k entity_search`. Expected: FAIL, scenario file not found.

- [ ] **Step 3: Write the scenario.** Copy `scenarios/knowledge-base/web-questions/scenario.yaml` (comments included) and change only what spec §4.1 fixes: `id`, `dataset`, `searchable`, `embedding_fields`, `index_timeout_s`, the perturbation seed. Keep the comment on `searchable` ("Order of `searchable` is the field priority every engine must respect.").

- [ ] **Step 4: Write the 6 engine configs.** For each engine, copy `engines/<engine>/configs/knowledge-base/web-questions.yaml`, change the scenario name in the header comment, and keep the `default` block as it is (the default blocks of `web-questions` and `community-qa` are identical for every engine, and two searchable fields need no extra setting). Write no `tuned:` block; Task 5 adds it.

- [ ] **Step 5: Add the CI line** to the parity job, after the existing `prepare` lines. The dataset cache key already hashes `datasets/**/dataset.yaml`, so the 1.8 GB corpus is downloaded once per change of that file.

- [ ] **Step 6: Prepare locally and run the checks.**
  - `uv run irbench prepare --scenario entity-search/dbpedia` (downloads 1.8 GB on the first run; normalising streams it twice). Expected `prepare.manifest.json` counts: `docs` 1000000, `queries` 467, `skipped_queries` 0, `dropped_docs` 0, `dropped_judgments` 0; `dataset.stats.judged_docs` 45595, `distractor_docs` 954405, `source_docs` 4635922.
  - `uv run pytest -q` and `uv run irbench lint`. Expected: pass, lint 0 findings, the config count up by 6.

- [ ] **Step 7: Commit.**

```bash
git add scenarios/entity-search engines/*/configs/entity-search .github/workflows/ci.yml harness/tests/test_scenario.py
git commit -m "scenarios: entity-search/dbpedia, default engine configs, CI prepare"
```

### Task 3: docs page

**Files:**
- Create: `docs/scenarios/entity-search.mdx`
- Modify: `docs/docs.json` (add `scenarios/entity-search` to the Scenarios group)

- [ ] **Step 1: Write the page,** modelled on `docs/scenarios/covid-papers.mdx`, from the spec:
  - the use case (spec §1) and what each query group is, with examples (§3.2);
  - source, revision and both licenses (§3.1);
  - corpus: the sampling rule, the seed and size, why scores still mean something, and the §3.5 table, labelled as measured on a draw of the same rule with a stand-in BM25, not on the shipped set;
  - the caveats: scores higher than on the full corpus; gaps between engines may look larger and more significant than on the full 4.64M corpus (put this next to how the report marks significance); the keyword-versus-semantic gap may be inflated (§7); unjudged results and judged@10; what the sample cannot show (many entities sharing a name);
  - modes: why filtered, sorted and faceted are not run (§4.2);
  - the fit-test rule and fallback size (§7.1), with a line to be filled with the measured result after Task 4.

- [ ] **Step 2: Check.** `uv run python -c "import json; d=json.load(open('docs/docs.json')); assert 'scenarios/entity-search' in json.dumps(d)"` exits 0.

- [ ] **Step 3: Commit.**

```bash
git add docs/scenarios/entity-search.mdx docs/docs.json
git commit -m "docs: entity search scenario"
```

---

## Runs (controller, after Task 3; not dispatched to implementers)

These steps run for hours and must run one at a time on a quiet machine (the laptop was swapping heavily while several sessions ran on 2026-10-31). The controller runs them and records each outcome.

### Task 4: embed and fit test

- [ ] `uv run irbench embed --scenario entity-search/dbpedia --model bge-small-en-v1.5`. Expected: 1,000,000 document vectors and the query vectors for 467 queries and their variants; about 2–3 hours.
- [ ] Fit test (spec §7.1): for each of the 6 engines, `uv run irbench run --scenario entity-search/dbpedia --engine <engine> --model bge-small-en-v1.5 --track default`. Sample `docker stats` every 10 s into the scratchpad, and record each engine's peak memory and index build time. Pass means: the index builds without an out-of-memory kill or restart within 7,200 s, and the relevance stage stays under the 0.1% error rate.
- [ ] If an engine fails:
  - it may get a memory setting (Elasticsearch heap, Postgres `maintenance_work_mem`), each a new departure with a justification citing the corpus size and the failure, and one retry;
  - if it still fails, set `params: {corpus_size: 500000}` in `dataset.yaml`, re-prepare, re-embed, and rerun every engine on the smaller corpus. The size is never chosen per engine;
  - either way, write the measured peaks, build times and outcome into the docs page (Task 3's fit-test line).
- [ ] Set `index_timeout_s` in the scenario from the slowest measured build (about twice it, rounded up to the next 1,800 s), with the measurement in a comment.

### Task 5: sweeps and tuned configs

- [ ] Sweep each engine on `entity-search/dbpedia` with the scratchpad sweep tool used for the earlier tuned tracks, over the same documented search settings as the knowledge-base sweeps, scored on nDCG@10 over the 467 queries per retrieval base. Index-level variants only where an engine's earlier sweeps showed they matter (Qdrant BM25 options).
- [ ] Write each engine's `tuned:` block with the winning settings, each with a justification carrying the default and tuned nDCG@10, in the style of `engines/meilisearch/configs/knowledge-base/web-questions.yaml`. `uv run irbench lint` gives 0 findings.

### Task 6: tuned runs and report

- [ ] Run the tuned track for the 6 engines, sequentially. Check every run manifest: succeeded, 0 errors, no violations.
- [ ] Rebuild the report from the tuned runs only. Check in a browser that the scenario page shows the four query groups and judged@10.
- [ ] Update the docs page with anything the runs changed (fit-test numbers, final `index_timeout_s`).
