# Home Depot Product Search (internal) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the internal-only scenario `ecommerce/home-depot-search`: 4,000 real Home Depot queries with real misspellings flagged, a `corrected` variant that measures what each misspelling costs, run on all 6 engines and never published.

**Architecture:** The harness gains four things:
- an internal-only flag on datasets, enforced by `publish`, `vectors pack`, `report` and lint;
- manually downloaded raw files, checked by sha256, because Kaggle needs a signed-in account;
- a data-supplied variant kind, `corrected`, which stays out of the headline `all` rows;
- paired variant-minus-original differences in `scores.json`, shown as a misspelling table on the scenario page.

A new `home_depot` normaliser reads the Kaggle zips and the Google spelling dictionary.

**Tech Stack:** Python 3.12, pydantic, ranx, Jinja2 report, typer, uv, pytest, ruff. Standard library only for the new code: `zipfile`, `csv`, `ast`, `html`.

**Spec:** `docs/superpowers/specs/2026-10-31-irbench-home-depot-design.md`

**Depends on:** nothing unlanded. This plan does not use ESCI's `query_set`.

**Overlap with the Chinese plan.** The Chinese title-search spec (branch `qdequele/spec-cmteb-chinese-cb61c4`) adds `Scenario.publishable: bool = True` and has `publish` refuse when it is false.
- If that has landed when Task 1 starts, `is_internal` (Task 1) also returns true for a scenario with `publishable: false`. Every check in this plan then covers both scenarios.
- If it has not landed, leave a note in the Chinese plan's file pointing at `irbench.internal.is_internal`.

## Global Constraints

- Ids:
  - scenario `ecommerce/home-depot-search`, dataset `home-depot`;
  - query ids `hd-<smallest Kaggle row id of the query>`, variant ids `<qid>-corrected`, `<qid>-typo`, `<qid>-prefix`.
- Data rules:
  - seed `20261031`; sample `4000` of the queries that have a grade ≥ 1;
  - grades `< 1.5 → 0`, `1.5 ≤ r < 2.5 → 1`, `≥ 2.5 → 2`;
  - groups `misspelled` and `clean`.
- Kaggle competition slug `home-depot-product-search-relevance`; files `train.csv.zip`, `product_descriptions.csv.zip`, `attributes.csv.zip`. Brand comes from attribute name `MFG Brand Name` only.
- Dictionary:
  - URL `https://raw.githubusercontent.com/ChenglongChen/Kaggle_HomeDepot/55c1033d0af3b6cf2f033fe4bcf3e1e0ffda3445/Code/Chenglong/google_spelling_checker_dict.py`;
  - sha256 `76edf5ee8de0da6cd9385e5713f0866ac6f38d76aae1c80eb8b921a876615fd2`;
  - parsed with `ast`, never imported or executed.
- `dataset.yaml`:
  - `distribution: internal`;
  - `license: "Kaggle competition rules (Home Depot Product Search Relevance), internal use only"`.
- Banner text, exactly: `Internal: not for publication (Kaggle competition data)`.
- Nothing from the data in the repository:
  - no query or product text in code, tests, fixtures, configs or commit messages;
  - test strings are invented;
  - tuned justifications carry no scores.
- The harness never reads Kaggle credentials.
- Existing scenarios must not change:
  - their `scores.json` rows are identical before and after (no `corrected` tasks exist there);
  - the render golden tests in `harness/tests/fixtures/render_golden/` stay unchanged.
- No new Python dependency. Never add `Co-Authored-By` lines to commits.
- `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check harness engines` and `uv run irbench lint` pass after every task.

## Review Focus

1. **Group rows for one group, several perturbations.** With per-perturbation group rows added (Task 6), `RunResult.group_ndcg` and `group_n` take the first row that matches the group, and could return a `typo` row. They must read the `perturbation == "all"` row. Task 6 tests a scores fixture with both kinds of rows.
2. **`corrected` in the run-vs-run significance test.** `per_task` in `scores.json` feeds the paired test between runs. If `corrected` tasks stay in it, the significance test covers tasks the headline excludes. Task 6 tests that `per_task` has no `-corrected:` task ids.
3. **Filters on products without a brand.** If the relevant products of a query have no brand, the query gets no filtered task (`_choose_filter` returns None). It must not get a filter on the string `"None"`. Task 8 tests that no task has a filter whose value is `"None"`.
4. **A runner who has not downloaded the Kaggle files.** `prepare` must fail before any work. The error must name each missing file and print the download instruction, not a stack trace from `zipfile`. Task 4 tests the CLI exit code and message.
5. **Dictionary keys with double spaces.** A dictionary key with a double space must match the query as typed, before whitespace normalisation, and the stored `text` must still be normalised. Task 8 tests a fixture query with a double space.

---

### Task 1: internal-only flag, `publish` and `vectors pack` refuse

**Files:**
- Create: `harness/irbench/internal.py`
- Modify: `harness/irbench/datasets/registry.py` (`DatasetSpec.distribution`)
- Modify: `harness/irbench/stages/prepare.py:91-96` (`"distribution"` in the manifest's `dataset` block)
- Modify: `harness/irbench/publish.py`, `harness/irbench/embed/publish.py` (`pack_vectors`), `harness/irbench/stages/embed.py:63` (skip `find_published` for internal scenarios)
- Test: `harness/tests/test_publish.py`, `harness/tests/test_vector_publish.py`, `harness/tests/test_embed_stage.py`, `harness/tests/test_prepare.py`

**Interfaces:**
- Produces: `DatasetSpec.distribution: Literal["public", "internal"] = "public"`.
- Produces: `irbench.internal.is_internal(root: Path, scenario_id: str) -> bool`. It loads the scenario, then its dataset spec. A scenario or dataset that no longer exists in the repository counts as not internal; the manifest check still applies.
- Produces: `irbench.internal.manifest_is_internal(manifest: dict[str, Any]) -> bool` (`manifest.get("dataset", {}).get("distribution") == "internal"`).
- Produces: `class InternalDataError(ValueError)`, raised by `publish_run` and `pack_vectors`. Its message names the dataset id and its `license`.
- Produces: `prepare.manifest.json` → `dataset.distribution` (copied into every run manifest by `_describe_inputs`, unchanged code).

- [ ] **Step 1: Write the failing tests**
  - `test_distribution_defaults_public`: a `dataset.yaml` without the key loads with `distribution == "public"`; an unknown value fails validation.
  - `test_prepare_manifest_records_distribution`: preparing a fixture dataset with `distribution: internal` writes `"distribution": "internal"` in `prepare.manifest.json`.
  - `test_publish_refuses_internal_run`: a run on a reference profile (not `local`) whose manifest has `dataset.distribution: internal` raises `InternalDataError`; nothing appears under `results/`.
  - `test_publish_refuses_internal_by_spec`: same, with an old manifest that has no `distribution` key but whose scenario's dataset spec is internal.
  - `test_pack_vectors_refuses_internal`: `pack_vectors` on an internal scenario raises `InternalDataError` before writing anything to `out_dir` or `embeddings/published.yaml`.
  - `test_embed_internal_never_downloads`: `embed_scenario` on an internal scenario computes locally even when the registry has an entry for it. Use a fake registry entry and a client that fails on any request.
  - `test_cli_publish_internal_exit_1`: `irbench publish <id>` prints the dataset id and exits 1.

- [ ] **Step 2: Run** `uv run pytest harness/tests/test_publish.py harness/tests/test_vector_publish.py harness/tests/test_embed_stage.py harness/tests/test_prepare.py -q`. Expected: FAIL (`distribution` unknown, `irbench.internal` missing).

- [ ] **Step 3: Implement** `DatasetSpec.distribution`, the prepare manifest key, `irbench/internal.py`, and the three call sites.
  - `publish_run` checks `manifest_is_internal(manifest) or is_internal(root, manifest["scenario"])` before the profile check.
  - The CLI `publish` command already turns a `ValueError` into exit 1.

- [ ] **Step 4: Run** the same command. Expected: PASS. Then the full suite and ruff.

- [ ] **Step 5: Commit** `git commit -m "harness: internal-only datasets; publish and vectors pack refuse them"`

### Task 2: report refuses internal runs unless `--internal`, and shows the banner

The spec asks the report to refuse internal runs when it builds from `results/`. A path check would be fragile, so this plan uses an explicit opt-in instead: `irbench report` refuses any internal run unless it gets `--internal`. The public site workflow (`.github/workflows/report.yml`) never passes the flag. This is stricter than the spec and meets the same goal.

**Files:**
- Modify: `harness/irbench/report/data.py` (`RunResult.internal`, `ScenarioResults.internal`)
- Modify: `harness/irbench/report/render.py` (`build_site(..., allow_internal: bool = False)`)
- Modify: `harness/irbench/report/templates/base.html.j2`, `harness/irbench/report/templates/scenario.html.j2`
- Modify: `harness/irbench/cli.py` (`report --internal`)
- Test: `harness/tests/test_report_render.py`, `harness/tests/test_cli.py`

**Interfaces:**
- Consumes: `manifest_is_internal` (Task 1).
- Produces:
  - `RunResult.internal: bool` (property, from `manifest`);
  - `ScenarioResults.internal: bool` (any run internal);
  - `class InternalRunsError(ValueError)`, raised by `build_site` when internal runs are present and `allow_internal` is false. Its message lists the run ids and says to pass `--internal`.
- Produces: the template variable `internal_banner: bool`. It is true on `index.html` when any scenario is internal, and on the scenario and run pages of internal scenarios. When true, the page shows the exact banner text from Global Constraints.

- [ ] **Step 1: Write the failing tests** (build bundles with the helpers in `report_fixtures.py`):
  - `test_build_site_refuses_internal_by_default`: a results dir with one internal run raises `InternalRunsError` naming the run id; `out` is not created or replaced.
  - `test_build_site_internal_banner`: with `allow_internal=True`:
    - the index, the internal scenario page and its run page contain the banner text;
    - a public scenario's page in the same build does not.
  - `test_public_build_unchanged`: a results dir with only public runs renders byte-identical pages with `allow_internal` false or true.
  - `test_cli_report_internal_flag`: `irbench report --results <dir> --out <out>` exits 1 on an internal run; with `--internal` it exits 0.

- [ ] **Step 2: Run** `uv run pytest harness/tests/test_report_render.py harness/tests/test_cli.py -q`. Expected: FAIL.

- [ ] **Step 3: Implement.** In the CLI, add `InternalRunsError` to the exceptions `report` turns into exit 1.

- [ ] **Step 4: Run** the same command. Expected: PASS.

- [ ] **Step 5: Commit** `git commit -m "report: refuse internal runs unless --internal; internal banner"`

### Task 3: lint `internal-data`, `not-prepared` as a notice, parity doc

**Files:**
- Modify: `harness/irbench/lint.py` (`Finding.severity`, rule `internal-data`, `not-prepared` severity)
- Modify: `harness/irbench/cli.py` (`lint` exits 1 on errors only; notices printed with a `notice:` prefix)
- Modify: `docs/methodology/parity.mdx` (section "Internal scenarios"; `internal-data` row in the rules table)
- Test: `harness/tests/test_lint.py`, `harness/tests/test_cli.py`

**Interfaces:**
- Consumes: `is_internal`, `manifest_is_internal` (Task 1).
- Produces: `Finding.severity: Literal["error", "notice"] = "error"`, and `Finding.__str__` prefixes `notice: ` for notices.
- Produces: the `internal-data` checks in `lint_repo`:
  - each `results/*/manifest.json` that is internal (by manifest or by spec);
  - each `embeddings/published.yaml` entry whose scenario is internal;
  - each page in `docs/docs.json` navigation equal to `scenarios/<last path segment of an internal scenario id>`.

- [ ] **Step 1: Write the failing tests**
  - `test_internal_data_results`, `test_internal_data_published_vectors`, `test_internal_data_docs_nav`: one finding each, rule `internal-data`, severity `error`, path pointing at the offending file.
  - `test_not_prepared_internal_is_notice`: a config for an unprepared internal scenario gives `not-prepared` with severity `notice`; for a public scenario it stays `error`.
  - `test_cli_lint_notices_exit_0`: only notices → exit 0, notices printed; one error → exit 1.

- [ ] **Step 2: Run** `uv run pytest harness/tests/test_lint.py harness/tests/test_cli.py -q`. Expected: FAIL.

- [ ] **Step 3: Implement**, and write the parity paragraph. It covers four points:
  - what `distribution: internal` means;
  - internal runs, vectors and docs pages are never published;
  - the report needs `--internal`, and its pages carry the banner;
  - tuned justifications on internal scenarios say what a setting does and that the sweep chose it, with no scores.

- [ ] **Step 4: Run** the same command, then `uv run irbench lint` on the repo (prepared as CI prepares it). Expected: PASS, the same findings as before this task.

- [ ] **Step 5: Commit** `git commit -m "lint: internal-data rule; not-prepared is a notice for internal scenarios"`

### Task 4: manually downloaded raw files

**Files:**
- Modify: `harness/irbench/datasets/registry.py` (`FileSpec`, `fetch_dataset`)
- Modify: `harness/irbench/datasets/fetch.py` (`ManualFileMissing`)
- Modify: `harness/irbench/cli.py` (`prepare` turns `ManualFileMissing` into a message and exit 1)
- Test: `harness/tests/test_fetch.py`, `harness/tests/test_cli.py`

**Interfaces:**
- Produces: `FileSpec` fields `url: str | None = None`, `sha256: str`, `member: str | None = None`, `manual: str | None = None`, `filename: str | None = None`.
  - Validation: exactly one of `url` and `manual`.
  - `manual` requires `filename`; `member` requires `url`.
- Produces: `class ManualFileMissing(FileNotFoundError)`, with attributes `paths: list[Path]` and `instruction: str`.
- Produces: in `fetch_dataset`, a manual file resolves to `raw_dir / filename`.
  - A wrong checksum raises the existing `ChecksumError`.
  - All missing manual files of a spec are collected and raised together as one `ManualFileMissing`, before any URL file is downloaded.

- [ ] **Step 1: Write the failing tests**
  - `test_filespec_url_xor_manual`: both set, or neither, fails validation; `manual` without `filename` fails.
  - `test_manual_file_present`: a file with the right sha256 in `raw_dir` is returned and no HTTP request is made (client that fails on any request).
  - `test_manual_file_missing_lists_all`: two missing manual files → one `ManualFileMissing` whose `paths` holds both and whose message contains the instruction. A `url` file in the same spec is not fetched.
  - `test_manual_file_bad_checksum`: raises `ChecksumError` naming the file.
  - `test_cli_prepare_missing_manual`: `irbench prepare` exits 1 and prints the instruction and both paths, with no traceback.

- [ ] **Step 2: Run** `uv run pytest harness/tests/test_fetch.py harness/tests/test_cli.py -q`. Expected: FAIL.

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run** the same command. Expected: PASS.

- [ ] **Step 5: Commit** `git commit -m "datasets: raw files a runner downloads by hand, checked by sha256"`

### Task 5: `corrected`, a data-supplied variant

**Files:**
- Modify: `harness/irbench/models.py` (`Perturbation` gains `"corrected"`; `REFERENCE_PERTURBATIONS`; `Query.corrected`)
- Modify: `harness/irbench/scenario.py` (`PerturbationSpec.kinds` gains `"corrected"`)
- Modify: `harness/irbench/perturb/variants.py` (`_apply` takes the `Query`)
- Modify: `harness/irbench/stages/prepare.py:55-64` (read `corrected`)
- Test: `harness/tests/test_variants.py`, `harness/tests/test_prepare.py`

**Interfaces:**
- Produces: `REFERENCE_PERTURBATIONS: frozenset[str] = frozenset({"corrected"})` in `irbench.models`. These are variants that are not perturbations of what users type. Task 6 keeps them out of `all` rows.
- Produces: `Query.corrected: str | None = None`. It is set on originals only; variants have `None`.
- Produces: `_apply(kind: str, query: Query, rng: random.Random, vocabulary: list[list[str]]) -> str | None`. For `corrected` it returns `query.corrected`. The other kinds use `query.text` as before.

- [ ] **Step 1: Write the failing tests**
  - `test_corrected_variant`: a query with `corrected="kohler toilet seat"` (invented) gets the variant `<qid>-corrected` with that text, the original's `group`, `variant_of` the original, and `corrected=None`.
  - `test_corrected_absent`: a query without `corrected` gets no `corrected` variant.
  - `test_corrected_equal_after_norm_dropped`: `corrected` equal to the text up to case and whitespace gives no variant.
  - `test_other_kinds_unchanged`: for a fixed seed, the `typo`, `prefix` and `synonym` variants of a fixture query set are identical before and after. Compare with variants generated by the current code and stored in the test.
  - `test_prepare_reads_corrected`: `queries.jsonl` rows with `corrected` reach `queries.variants.jsonl`.

- [ ] **Step 2: Run** `uv run pytest harness/tests/test_variants.py harness/tests/test_prepare.py -q`. Expected: FAIL.

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run** the same command. Expected: PASS.

- [ ] **Step 5: Commit** `git commit -m "perturb: corrected, a variant supplied by the dataset"`

### Task 6: scores keep `corrected` out of `all`; group rows per perturbation; paired differences

**Files:**
- Modify: `harness/irbench/score/relevance.py`
- Modify: `harness/irbench/stages/score_run.py` (`per_task` without reference kinds; `paired` block)
- Modify: `harness/irbench/report/data.py` (`group_ndcg`, `group_n` read `perturbation == "all"`; `RunResult.paired(...)`; manifest validation accepts `paired`)
- Test: `harness/tests/test_score_relevance.py`, `harness/tests/test_report_data.py`

**Interfaces:**
- Consumes: `REFERENCE_PERTURBATIONS` (Task 5).
- Produces: `score_relevance(tasks, qrels, run) -> list[GroupScore]`.
  - Tasks whose perturbation is in `REFERENCE_PERTURBATIONS` get their own `(mode, base, perturbation)` row only. They are absent from `(mode, base, "all")` and from every group row with perturbation `all`.
  - Group rows are emitted for `all` and for each perturbation present: `GroupScore(group=g, perturbation=p)`.
- Produces: `class PairedDelta(BaseModel)` with fields `mode, base, perturbation, group: str | None, n: int, delta: float, ci95: tuple[float, float]`.
  - `delta` is the mean of (variant nDCG@10 − original nDCG@10) over pairs; the interval is `ci95` on the differences.
  - Pairs are formed per task: variant task `<variant_of>-<kind>:<mode>:<base>` with original task `<variant_of>:<mode>:<base>`, both supported.
- Produces: `paired_deltas(tasks: list[QueryTask], per_task: dict[str, float]) -> list[PairedDelta]` in `score/relevance.py`.
  - One row per (mode, base, perturbation ≠ original, group), plus one with `group=None` over all groups.
  - Its per-task input includes reference kinds.
- Produces: in `scores.json`, `"paired": [...]` (dumped with `exclude_none`) and `per_task.ndcg@10` without reference-kind tasks.
- Produces: `RunResult.paired(group: str, perturbation: str, query_group: str | None) -> dict[str, Any] | None`.
- Produces: `RunResult.group_ndcg(group: str, query_group: str, perturbation: str = "all")` and `RunResult.group_n(query_group: str, perturbation: str = "all")`. Both match on the perturbation too.

- [ ] **Step 1: Write the failing tests**
  - `test_corrected_not_in_all`: tasks with originals, `typo` and `corrected` in groups `misspelled`/`clean`:
    - the `all` row's `n` counts originals and `typo` only;
    - a `corrected` row exists;
    - the `(misspelled, all)` row excludes `corrected`;
    - `(misspelled, corrected)` and `(clean, typo)` rows exist.
  - `test_existing_rows_unchanged`: for tasks without reference kinds, the rows that existed before (`all`, per perturbation, `(group, all)`) are equal to the current output. Pin the current output in the test.
  - `test_paired_deltas`: with hand-set per-task values, two pairs (0.2 → 0.6 and 0.5 → 0.3) give `delta == 0.1`, `n == 2`. An original with no variant is not paired.
  - `test_per_task_excludes_reference`: `scores.json` `per_task` has no task id containing `-corrected:`.
  - `test_group_ndcg_reads_all_row`: the scores have `(misspelled, typo)` before `(misspelled, all)`; `group_ndcg` and `group_n` return the `all` row's values (Review Focus 1).

- [ ] **Step 2: Run** `uv run pytest harness/tests/test_score_relevance.py harness/tests/test_report_data.py -q`. Expected: FAIL.

- [ ] **Step 3: Implement.** Bundles without `paired` (all existing ones) stay valid.

- [ ] **Step 4: Run** the same command, then the full suite. Expected: PASS; the render golden tests unchanged.

- [ ] **Step 5: Commit** `git commit -m "score: corrected out of the headline; group rows per perturbation; paired deltas"`

### Task 7: misspelling table on the scenario page

**Files:**
- Modify: `harness/irbench/report/render.py` (`misspelling_tables(s)`, key `misspellings` in `scenario_view`)
- Modify: `harness/irbench/report/templates/scenario.html.j2`
- Test: `harness/tests/test_report_render.py`

**Interfaces:**
- Consumes: `RunResult.paired`, `group_ndcg`, `group_n`, `judged` (Task 6).
- Produces: `misspelling_tables(s: ScenarioResults) -> list[dict[str, Any]]`. It returns one table per base in `BASE_GROUPS` that has any `corrected` paired row, and `[]` when no run has `corrected` rows.
  - Each row: `run`, nDCG@10 misspelled `original`, nDCG@10 misspelled `corrected`, `corrected` delta and interval, `typo` delta and interval on the `clean` group, judged@10.
  - Each table: `n_misspelled` and `n_clean`, the original queries per group (`group_n(q, "original")`).
  - nDCG@10 cells come from `group_ndcg(base, "misspelled", "original")` and `group_ndcg(base, "misspelled", "corrected")`.

- [ ] **Step 1: Write the failing tests**
  - `test_misspelling_table_values`: a fixture run with paired rows renders a table whose cells equal the fixture's numbers. The heading reads "Misspellings".
  - `test_no_misspelling_table_without_corrected`: a scenario without `corrected` rows has no such section; its page is byte-identical to before.

- [ ] **Step 2: Run** `uv run pytest harness/tests/test_report_render.py -q`. Expected: FAIL.

- [ ] **Step 3: Implement.** Under the table, print the dictionary caveats from spec §5.1 as fixed copy:
  - "Misspelled means Google's speller changed the query; a few clean queries still contain mistakes, and a few corrections are wrong."
  - The measured edit sizes: edit distance 1 / 2 / 3 / 4+ = 1,086 / 415 / 100 / 39 of 1,640.

  Show the copy only when the scenario id is `ecommerce/home-depot-search`, through a per-scenario note dict in `render.py`.

- [ ] **Step 4: Run** the same command. Expected: PASS.

- [ ] **Step 5: Commit** `git commit -m "report: misspelling table (real vs corrected, synthetic typo)"`

### Task 8: the `home_depot` loader and `datasets/home-depot/dataset.yaml`

**Files:**
- Create: `harness/irbench/datasets/home_depot.py`
- Create: `datasets/home-depot/dataset.yaml`
- Modify: `harness/irbench/datasets/__init__.py` (register `"home-depot": normalize_home_depot`)
- Test: `harness/tests/test_home_depot.py` (fixtures built in `tmp_path`: zips of invented CSV rows, written in Latin-1, and an invented dictionary `.py`)

**Interfaces:**
- Consumes: `NormalizedDataset` (`irbench.datasets.wands`), `write_jsonl`, `write_qrels`.
- Produces: `normalize_home_depot(raw: dict[str, Path], out_dir: Path) -> NormalizedDataset`, with raw keys `train`, `descriptions`, `attributes`, `spelling`.
- Produces: loader constants `SEED = 20261031`, `SAMPLE_SIZE = 4000`, `BRAND_ATTRIBUTE = "MFG Brand Name"`.
- Produces: `grade(average: float) -> int`.
- Produces: `load_spelling(path: Path) -> dict[str, str]`. It parses the file with `ast.parse` and accepts exactly one top-level assignment whose value is a dict of string literals. Anything else raises `ValueError`.
- Produces: `docs.jsonl` rows `{"id", "title", "brand", "description"}`, with `brand: null` when unknown. This is what ESCI writes; `filters.field_values` returns `[]` for null, so a null brand never becomes a filter value (Review Focus 3).
- Produces: `queries.jsonl` rows `{"qid", "text", "group"}` plus `"corrected"` for misspelled queries. Rows are sorted by the numeric part of the qid.
- Produces: `dropped` keys `docs`, `judgments` and `queries_no_positive`.

- [ ] **Step 1: Write the failing tests**
  - `test_grades_boundaries`: 1.0→0, 1.33→0, 1.5→1, 1.67→1, 2.33→1, 2.5→2, 2.67→2, 3.0→2.
  - `test_query_without_positive_dropped`: a query whose pairs all average below 1.5 is absent and counted in `dropped["queries_no_positive"]`.
  - `test_qid_is_min_row_id`: a query on rows 7, 3 and 12 gets `hd-3`.
  - `test_title_from_smallest_row`: one product with two titles keeps the title of its smallest row id.
  - `test_brand_only_from_mfg_brand_name`: a product with only `Color Family` has `brand: null`. A product with two `MFG Brand Name` rows keeps the first in file order.
  - `test_description_entities_unescaped`: `&amp;` becomes `&`; runs of whitespace collapse to one space.
  - `test_spelling_match_raw_then_normalise`: the fixture query `drywal  anchors` (two spaces) matches the dictionary key with two spaces. Its `text` is `drywal anchors`, its `group` is `misspelled`, and its `corrected` is the dictionary value, whitespace-normalised.
  - `test_clean_query_has_no_corrected`: an unmatched query is `clean` with no `corrected` key.
  - `test_spelling_parsed_not_executed`: a dictionary file containing `__import__("os")` or a function call raises `ValueError`, and a side-effect sentinel file is not created.
  - `test_latin1_decoded`: a train row with `é` written in Latin-1 reads back as `é`.
  - `test_sample_stable`: with `SAMPLE_SIZE` monkeypatched to 3 over 6 usable queries, two runs pick the same qids, sorted. With 3 usable queries all are kept.
  - `test_corpus_is_judged_products`: a description for a product never judged in `train` is not in `docs.jsonl`.
  - `test_prepare_home_depot_tasks`:
    - `prepare` on a fixture scenario with these docs creates `-corrected` tasks for misspelled queries only;
    - no task has a filter whose value is `"None"`;
    - `prepare.manifest.json` has `distribution: internal`.

- [ ] **Step 2: Run** `uv run pytest harness/tests/test_home_depot.py -q`. Expected: FAIL (module missing).

- [ ] **Step 3: Implement `normalize_home_depot`.**
  - Read the CSVs from inside the zips with `zipfile` and `csv` (`encoding="latin-1"`; check the real file's encoding in Task 9 and change it there if it is not Latin-1).
  - Read attributes in a streaming pass that keeps `MFG Brand Name` rows only. The file has about 2M rows.
  - Sample as `amazon_esci` does: `random.Random(SEED).sample(sorted(eligible), SAMPLE_SIZE)`, then sort.

- [ ] **Step 4: Write `datasets/home-depot/dataset.yaml`.**
  - It has `id`, `version: "kaggle-2016"`, `license` and `distribution` as in Global Constraints, and `source: https://www.kaggle.com/competitions/home-depot-product-search-relevance`.
  - Files:
    - `train`, `descriptions`, `attributes`: `manual` + `filename`, with the instruction text from spec §6.2, which names the rules page and `kaggle competitions download -c home-depot-product-search-relevance -p <raw dir>`;
    - `spelling`: `url` + `sha256` from Global Constraints.
  - The three Kaggle sha256 values are not known until Task 9. Write them as 64 zeros and add the YAML comment `# replaced with the real sha256 in Task 9`. Any `prepare` before Task 9 then fails with `ChecksumError`, never with silently wrong data.

- [ ] **Step 5: Run** the test file, then the full suite. Expected: PASS.

- [ ] **Step 6: Commit** `git commit -m "datasets: home-depot loader (Kaggle files, Google spelling dictionary)"`

### Task 9: real data check (needs the user)

**This task needs the user.** Downloading the files means accepting the Kaggle competition rules with a personal Kaggle account. Ask the user to do it; never use or ask for their Kaggle credentials.

**Files:**
- Modify: `datasets/home-depot/dataset.yaml` (real sha256 values)
- Modify: `harness/irbench/datasets/home_depot.py` (only if the checks below need it)

- [ ] **Step 1: Ask the user** to accept the rules on Kaggle and download `train.csv.zip`, `product_descriptions.csv.zip` and `attributes.csv.zip` into `.irbench/data/home-depot/raw/`. Wait for them to confirm.

- [ ] **Step 2: Record** `shasum -a 256 .irbench/data/home-depot/raw/*.zip` into `dataset.yaml`.

- [ ] **Step 3: Check the data against spec §4.2.**
  - Use a scratch script outside the repository, run with `python -I`, that reads the zips (also outside the repository). Expected:
    - 74,067 train rows, 11,795 distinct search terms, 54,667 product uids;
    - grade counts 5,115 / 34,595 / 34,357;
    - 1,640 queries in the dictionary.
  - Check the encoding: decoding as Latin-1 must give no mojibake on rows with non-ASCII bytes. If the file is UTF-8, switch the loader's encoding and its test.
  - Measure brand coverage over the 54,667 products. If it is under 50%, stop and tell the user: spec §4.4 then drops the filtered and faceted modes.
  - If a count differs, stop and report it. Do not adjust the loader to match the numbers.

- [ ] **Step 4: Commit** `git commit -m "datasets: pin home-depot Kaggle checksums"`. The message holds no counts or data.

### Task 10: scenario, engine configs, runs

**Files:**
- Create: `scenarios/ecommerce/home-depot-search/scenario.yaml` (verbatim from spec §7)
- Create: `engines/{meilisearch,elasticsearch,typesense,qdrant,postgres,postgres-paradedb}/configs/ecommerce/home-depot-search.yaml`
- Modify: each engine `README.md` (one line: the scenario is internal; its tuned justifications carry no scores)

- [ ] **Step 1: Write the scenario file** and `uv run irbench prepare --scenario ecommerce/home-depot-search`.
  - Expected: 54,667 docs, 4,000 queries.
  - The manifest's `dropped.queries_no_positive` is 95.
  - The number of `corrected` variants is near 555.

- [ ] **Step 2: Write the six default configs.**
  - Start from each engine's `ecommerce/product-search.yaml` default block, minus anything there only for the sorted mode. Meilisearch's `rankingRules` change exists for `sort` and goes, with its justification.
  - Keep each header comment's structure, naming this scenario.
  - Run `uv run irbench lint`. Expected: no `query-leak`, and no `not-prepared` finding here, since the scenario is prepared locally.

- [ ] **Step 3: Embed and run the default track.**
  - `uv run irbench embed --scenario ecommerce/home-depot-search --model bge-small-en-v1.5`.
  - Then `irbench run` for each engine with `--track default` and the `local` profile, sequentially.

- [ ] **Step 4: Sweep each engine and write the tuned blocks.**
  - Sweep each engine's documented search settings, scored on nDCG@10 of the `all` row per base. Typo-tolerance switches the parity rules already allow are included.
  - The sweep script is scratch, outside the repository, as for earlier tuned tracks.
  - Write the `tuned:` blocks. Each justification says what the setting does and that the sweep chose it, with no numbers. Keep the numbers in the sweep notes outside the repository.
  - Run the tuned track for each engine.

- [ ] **Step 5: Grade-sensitivity check.**
  - Rescore every tuned run with the alternative grades (averages ≥ 2 relevant, 1.67 → 0) using a scratch script over `run.trec` and the raw judgments.
  - Report to the user whether the order of engines changes on any base, and keep the result in the sweep notes.
  - Nothing is committed from this step.
  - Spec §6.5 also puts this result on the preview page. That would need either a committed result or a new data channel into the report. This plan reports it to the user instead (my call; flagged in the hand-off).

- [ ] **Step 6: Build the preview report.**
  - Run `uv run irbench report --results .irbench/runs --out .irbench/site --internal`.
  - Check that every Home Depot page shows the banner and the misspelling table.
  - Check that `uv run irbench publish <a home-depot run id>` exits 1.

- [ ] **Step 7: Commit** the scenario, configs and README lines: `git commit -m "scenarios: ecommerce/home-depot-search (internal), engine configs"`.
  - Before committing, run `git diff --cached | grep -iE 'ndcg|[0-9]\.[0-9]{3}'` on the new configs. It must find nothing beyond setting values.
