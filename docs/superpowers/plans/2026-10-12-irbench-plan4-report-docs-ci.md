# IRBench Plan 4 — Report, Docs, CI and Published Vectors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish IRBench v0.1. Run results become publishable bundles and a static HTML report. Engine configs are machine-checked against the parity rules. Metrics are cross-checked against `trec_eval`. Vectors are published so reproducing a run needs no model or API key. A Mintlify site documents the methodology, and CI runs every check on each PR.

**Architecture:**
- **Publishing:** `irbench publish` copies a run's small outputs (manifest, scores, perf, index) into `results/<run-id>/`. It refuses runs from the `local` profile.
- **Report:** `irbench report` reads any directory of run folders (`results/` or `.irbench/runs/` for a local preview). It renders Jinja2 templates into plain HTML with inline SVG throughput charts and no JavaScript. Pages are grouped by scenario × hardware profile, so profiles are never compared.
- **Parity lint:** `irbench lint` checks every `engines/*/configs/**.yaml` (justification for every setting, valid N/A keys, no query text or ids, a pinned image).
- **Published vectors:** `irbench vectors pack` writes release assets plus an entry in `embeddings/published.yaml`. `irbench embed` downloads registered vectors (checksum, dataset version, seed and ids verified) before it falls back to computing.

**Tech Stack:**
- Python 3.12 harness (existing): typer, pydantic, httpx, pyarrow.
- New dependencies: `jinja2>=3.1` (runtime) and `pytrec-eval-terrier` (dev, the `trec_eval` C code with Python bindings).
- Mintlify (`docs.json`) for the docs site.
- GitHub Actions for CI and the report deployment.

**Spec:** `docs/superpowers/specs/2026-10-09-irbench-design.md`

**Plan series:** Plan 4 of 4 for v0.1.
- Plans 1–3 built prepare → score, the Meilisearch/Elasticsearch/Qdrant adapters and the Rust loadgen with the perf stage.
- Plan 2 is PR #1 (`plan2-engines`). Plan 3 is PR #2 (`plan3-loadgen`), stacked on #1.
- This plan branches from `plan3-loadgen` as `plan4-report`.

## Global Constraints

- Report: "Static HTML generated from results JSON. No frontend framework." (spec §2). The pages ship no JavaScript; hover uses native SVG `<title>` tooltips, and every chart has a table with the same numbers.
- "Results show relevance broken down by query imperfection (typo, prefix, synonym), and performance (latency, throughput, indexing cost) separately — never blended into one score." (spec §1). The report has no overall score and no ranking.
- "Profiles can be added; results are only compared within a profile." (spec §8). Report pages are per scenario × hardware profile.
- "Only reference-machine results are published." (spec §7). `irbench publish` refuses the `local` profile.
- "A failed stage marks that run `failed` with reason in the manifest; the batch continues; the report shows the failure explicitly." (spec §9)
- "Pinecone … excluded from performance tables and marked 'not comparable'." (spec §2). Pinecone isn't in v0.1, so this plan adds no special case.
- "vectors computed once, cached and published so reproduction needs no API keys. Engines never embed themselves." (spec §2)
- "CI rejects configs containing query IDs or query strings from the query set." (spec §7)
- "ANN parameters start at each engine's documented defaults; any change requires a `justification:` field." (spec §7). The lint applies this to every setting in `index`/`search`, since the default track is "engine as shipped plus the minimum to load the schema".
- Docs: Mintlify site in `docs/` (methodology, parity rules, contributing); parity rules live at `docs/methodology/parity.mdx` (spec §2, §7). IRBench has no HTTP API, so there is no OpenAPI spec.
- Python stays ruff-clean (`uv run ruff check .`, `uv run ruff format --check harness engines`); Rust stays `cargo fmt --check` / `cargo clippy --all-targets -- -D warnings` clean.
- Chart colors use the validated reference categorical palette (8 slots, light and dark steps), one slot per engine, assigned by sorted engine name across the whole site so an engine keeps its color on every page. Engines past the 8th fold into a neutral "other" color; hues are never generated or cycled. The tuned track is drawn dashed, in the same color as that engine's default track.
- Commits must NOT contain `Co-Authored-By` lines (user rule).

## Review Focus

1. **A published bundle is corrupt or half-written** (invalid JSON, a manifest without required keys, or `succeeded` with no `scores.json`). The report still builds and lists the bundle as unreadable or incomplete; `irbench report` exits 1 so CI notices. Tests: Task 4 `test_corrupt_bundle_is_reported_not_fatal` and `test_succeeded_without_scores_is_incomplete`; Task 5 `test_cli_exits_1_on_unreadable_bundles`.
2. **One results directory holds runs from two hardware profiles.** They never appear in the same table or chart. Test: Task 4 `test_profiles_are_never_mixed`.
3. **A config value, N/A reason or error message contains HTML** (`<script>`). It is escaped everywhere, including SVG labels. Tests: Task 5 `test_text_from_results_is_escaped`; Task 6 `test_labels_are_escaped`.
4. **`--out` points at a non-empty directory the report didn't create** (e.g. the repo root). The build refuses and deletes nothing. Test: Task 5 `test_refuses_to_overwrite_a_foreign_directory`.
5. **Published vectors don't match the locally prepared data** (different dataset version or seed, tampered file, or ids out of order). `embed` fails with a message pointing at `--compute` and never leaves mismatched vectors behind. Tests: Task 7 `test_seed_mismatch_is_refused`, `test_tampered_asset_is_refused`, `test_id_mismatch_removes_the_file`.

---

## File Structure

```
.gitignore                                   # + dist/
README.md                                    # new: what IRBench is, quickstart, links
pyproject.toml                               # + jinja2, + pytrec-eval-terrier (dev), ruff first-party
results/README.md                            # bundle layout; only reference-profile runs
embeddings/published.yaml                    # registry of published vectors (written by `vectors pack`)
harness/irbench/
  engines/config.py                          # + Credit, EngineConfig.credits, top-level `credits`
  lint.py                                    # parity lint: Finding, QuerySet, lint_repo
  publish.py                                 # publish_run -> results/<run-id>/
  embed/publish.py                           # registry, pack_vectors, download_published
  stages/embed.py                            # download published vectors before computing
  stages/run.py                              # manifest config gains credits; embed result gains source
  cli.py                                     # + lint, publish, report, vectors pack; embed --compute
  report/__init__.py
  report/data.py                             # RunResult, ScenarioResults, Unreadable, load_results
  report/charts.py                           # Point, Series, nice_max, qps_chart (inline SVG)
  report/render.py                           # view models, formatters, build_site
  report/static/style.css                    # tokens (light/dark), tables, chart marks
  report/templates/base.html.j2, index.html.j2, scenario.html.j2, perf.html.j2, run.html.j2
harness/tests/
  report_fixtures.py                         # write_bundle(): realistic bundle files
  test_trec_eval.py
  test_lint.py
  test_publish.py
  test_report_data.py
  test_report_render.py
  test_report_charts.py
  test_vector_publish.py
  test_engine_config.py                      # + credits tests
  test_run_pipeline.py                       # + credits in manifest
docs/
  docs.json, .mintignore
  index.mdx, quickstart.mdx
  methodology/overview.mdx, parity.mdx, performance.mdx
  contributing/engine-config.mdx, new-engine.mdx
  reference/cli.mdx, results.mdx
.github/workflows/ci.yml                     # python, parity, contract, loadgen, docs, report
.github/workflows/report.yml                 # build results/ report, deploy to GitHub Pages on main
```

---

### Task 1: Cross-check metrics against `trec_eval`

Spec §10: "Metric correctness: our scores vs `trec_eval` on a fixed fixture." Today only hand-computed values are tested.

**Files:**
- Modify: `pyproject.toml` (dev group)
- Test: `harness/tests/test_trec_eval.py`

**Interfaces:**
- Consumes: `score_relevance(tasks, qrels, run) -> list[GroupScore]` (`harness/irbench/score/relevance.py`); metric keys `ndcg@10`, `recall@10`, `recall@100`, `mrr@10`, `success@1`.
- Produces: nothing new for later tasks.

- [ ] **Step 1: Add the dev dependency**

Run: `uv add --dev "pytrec-eval-terrier>=0.5.7"` then `uv run python -c "import pytrec_eval; print(pytrec_eval.supported_measures >= {'ndcg_cut', 'recall', 'recip_rank', 'success'})"`
Expected: `True`. If no wheel installs for this platform, stop and report BLOCKED. Do not hand-roll a `trec_eval` replacement; the point is an independent implementation.

- [ ] **Step 2: Write the cross-check test**

```python
"""Our metrics vs trec_eval (spec §10), through pytrec_eval's bindings to trec_eval's C code."""

import random

import pytest
import pytrec_eval

from irbench.models import QueryTask
from irbench.score.relevance import score_relevance

OURS_TO_TREC = {
    "ndcg@10": "ndcg_cut_10",
    "recall@10": "recall_10",
    "recall@100": "recall_100",
    "mrr@10": "recip_rank",  # computed on the top 10 only, below
    "success@1": "success_1",
}


def _fixture(seed: int = 7, n_tasks: int = 60, n_docs: int = 150):
    rng = random.Random(seed)
    docs = [f"d{i}" for i in range(n_docs)]
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, list[str]] = {}
    for i in range(n_tasks):
        tid = f"t{i}"
        judged = rng.sample(docs, rng.randint(1, 20))
        grades = {d: rng.choice([0, 1, 2]) for d in judged}
        grades[judged[0]] = rng.choice([1, 2])  # prepare drops queries without a relevant doc
        qrels[tid] = grades
        if i % 10 != 9:  # every 10th task returned nothing: it must score 0
            run[tid] = rng.sample(docs, rng.randint(0, 120))
    return qrels, run


def _trec_eval(qrels, run, task_ids):
    # trec_eval orders by score (ties by docid), so give rank r a strictly decreasing score.
    full = {t: {d: float(1000 - r) for r, d in enumerate(run.get(t, []))} for t in task_ids}
    top10 = {t: {d: float(1000 - r) for r, d in enumerate(run.get(t, [])[:10])} for t in task_ids}
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {"ndcg_cut.10", "recall.10", "recall.100", "recip_rank", "success.1"}
    )
    on_full, on_top10 = evaluator.evaluate(full), evaluator.evaluate(top10)
    out = {}
    for ours, trec in OURS_TO_TREC.items():
        source = on_top10 if trec == "recip_rank" else on_full
        # pytrec_eval omits queries with an empty run; they score 0, like ours.
        out[ours] = sum(source.get(t, {}).get(trec, 0.0) for t in task_ids) / len(task_ids)
    return out


def _task(tid: str) -> QueryTask:
    return QueryTask(
        task_id=tid, qid=tid, variant_of=tid, perturbation="original", text="x",
        mode="keyword", base="keyword",
    )


@pytest.mark.parametrize("seed", [7, 11, 2026])
def test_metrics_match_trec_eval(seed):
    qrels, run = _fixture(seed)
    tasks = [_task(t) for t in qrels]
    groups = score_relevance(tasks, qrels, run)
    ours = next(g for g in groups if g.perturbation == "all").metrics
    expected = _trec_eval(qrels, run, list(qrels))
    for name, value in expected.items():
        assert ours[name] == pytest.approx(value, abs=1e-6), name
```

- [ ] **Step 3: Run it**

Run: `uv run pytest harness/tests/test_trec_eval.py -v`
Expected: 3 passed. This test guards existing code, so it should pass the first time. If a metric differs, that is a real metric bug: fix `harness/irbench/score/relevance.py` (not the test), add the failing seed to the ledger with a one-line ruling, and re-run the whole suite.

- [ ] **Step 4: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check harness engines
git add pyproject.toml uv.lock harness/tests/test_trec_eval.py
git commit -m "test: cross-check relevance metrics against trec_eval"
```

---

### Task 2: Config credits and the parity lint

Spec §7: machine-checkable parity rules are enforced in CI, and the report credits tuned configs (§11). This task adds an optional top-level `credits` list to engine configs and `irbench lint`.

**Files:**
- Modify: `harness/irbench/engines/config.py`
- Modify: `harness/irbench/stages/run.py` (`_describe_inputs`)
- Create: `harness/irbench/lint.py`
- Modify: `harness/irbench/cli.py`
- Test: `harness/tests/test_engine_config.py`, `harness/tests/test_run_pipeline.py`, `harness/tests/test_lint.py`

**Interfaces:**
- Consumes: `load_engine_config(root, engine, scenario_id) -> EngineConfig`; `EngineConfig.raw`, `.track(name) -> TrackConfig` (`index`, `search`, `unsupported`, `justification`); `TRACKS`; `Mode`, `BASE_MODES` (`irbench.models`); `Workspace.scenario_dir`.
- Produces:
  - `Credit(name: str, url: str | None)` and `EngineConfig.credits -> list[Credit]`.
  - Run manifests gain `config.credits: list[{"name", "url"}]`, filled for the tuned track and empty for default (read by the report in Task 5).
  - `irbench.lint.Finding(path, rule, message)`, `QuerySet.of(texts, ids)`, `QuerySet.load(root, scenario_id)`, `lint_repo(root, queries=None) -> tuple[int, list[Finding]]`.
  - CLI `irbench lint` (used by CI in Task 9).

- [ ] **Step 1: Write the failing credits tests**

Append to `harness/tests/test_engine_config.py`:

```python
def _write_credit_config(root, data):
    path = root / "engines" / "eng" / "configs" / "ecommerce" / "product-search.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(data))
    return path


def test_credits_are_optional_and_typed(tmp_path):
    _write_credit_config(tmp_path, {
        "default": {},
        "tuned": {"index": {"m": 32}, "justification": {"m": "recall"}},
        "credits": [{"name": "Jane Doe", "url": "https://github.com/jane"}, {"name": "Ann"}],
    })
    config = load_engine_config(tmp_path, "eng", "ecommerce/product-search")
    assert [c.model_dump() for c in config.credits] == [
        {"name": "Jane Doe", "url": "https://github.com/jane"},
        {"name": "Ann", "url": None},
    ]


def test_config_without_credits_has_none(tmp_path):
    _write_credit_config(tmp_path, {"default": {}})
    assert load_engine_config(tmp_path, "eng", "ecommerce/product-search").credits == []


@pytest.mark.parametrize("credits", [{"name": "x"}, [{"url": "https://x"}], [{"name": "x", "x": 1}]])
def test_malformed_credits_name_the_file(tmp_path, credits):
    path = _write_credit_config(tmp_path, {"default": {}, "credits": credits})
    with pytest.raises(ValueError, match="credits") as error:
        load_engine_config(tmp_path, "eng", "ecommerce/product-search")
    assert str(path) in str(error.value)
```

Check the imports at the top of the file include `pytest`, `yaml` and `load_engine_config`; add any that are missing.

Append to `harness/tests/test_run_pipeline.py`:

```python
def test_manifest_credits_tuned_configs_only(tmp_path, wands_raw):
    config = {
        **FAKE_CONFIG,
        "tuned": {"index": {"x": 1}, "justification": {"x": "test"}},
        "credits": [{"name": "Jane Doe", "url": "https://github.com/jane"}],
    }
    root = fake_root(tmp_path, wands_raw, config=config)
    common = dict(runtime_factory=lambda *a: FakeRuntime(), prepare_client=fixture_client(wands_raw))

    tuned = run_pipeline(root, FAKE_SPEC.model_copy(update={"track": "tuned"}), ["prepare"], **common)
    default = run_pipeline(root, FAKE_SPEC, ["prepare"], **common)

    assert tuned["config"]["credits"] == [{"name": "Jane Doe", "url": "https://github.com/jane"}]
    assert default["config"]["credits"] == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest harness/tests/test_engine_config.py harness/tests/test_run_pipeline.py -k credits -v`
Expected: FAIL (`top-level keys must be 'default' and optional 'tuned'`, `AttributeError: ... credits`, `KeyError: 'credits'`).

- [ ] **Step 3: Implement credits**

In `harness/irbench/engines/config.py`, add after `TrackConfig`:

```python
class Credit(BaseModel):
    """Who contributed a tuned config; shown on the report's run page (spec §11)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    url: str | None = None
```

Add to `EngineConfig`:

```python
    @property
    def credits(self) -> list[Credit]:
        items = self.raw.get("credits") or []
        if not isinstance(items, list):
            raise ValueError("credits must be a list of {name, url}")
        return [Credit.model_validate(c) for c in items]
```

In `load_engine_config`, replace the top-level key check and validate credits:

```python
    unknown = sorted(set(raw) - set(TRACKS) - {"credits"})
    if unknown or "default" not in raw:
        raise ValueError(
            f"{path}: top-level keys must be 'default', optional 'tuned' and optional "
            f"'credits', got {sorted(raw)}"
        )
    config = EngineConfig(path, raw)
    for name in TRACKS:
        if name in raw:
            config.track(name)
    try:
        config.credits
    except ValueError as e:
        raise ValueError(f"{path}: credits: {e}") from e
    return config
```

In `harness/irbench/stages/run.py`, `_describe_inputs`, add the credits key to the config dict:

```python
        manifest["config"] = {
            "path": str(ctx.config.path.relative_to(ctx.root)),
            "hash": ctx.config.hash(ctx.spec.track),
            "diff_vs_default": ctx.config.diff(ctx.spec.track),
            # Credits belong to the tuned track (spec §11); the default track is the engine as shipped.
            "credits": [c.model_dump() for c in ctx.config.credits]
            if ctx.spec.track == "tuned"
            else [],
        }
```

- [ ] **Step 4: Run the credits tests**

Run: `uv run pytest harness/tests/test_engine_config.py harness/tests/test_run_pipeline.py -v`
Expected: all pass.

- [ ] **Step 5: Write the failing lint tests**

Create `harness/tests/test_lint.py`:

```python
import json

import pytest
import yaml
from typer.testing import CliRunner

from conftest import REPO_ROOT
from irbench.cli import app
from irbench.lint import QuerySet, lint_repo

SCENARIO = "ecommerce/product-search"
QUERIES = QuerySet.of(
    ["salon chair", "sofa", "smart coffee table", "salon cair"],
    ["0", "0-typo", "0-typo:keyword:keyword"],
)


def make_repo(tmp_path, config, image="getmeili/meilisearch:v1.54.3", skip=()):
    engine = tmp_path / "engines" / "eng"
    (engine / "configs" / "ecommerce").mkdir(parents=True)
    (engine / "docker").mkdir()
    files = {
        "adapter.py": "",
        "README.md": "# eng\n",
        "docker/compose.yaml": yaml.safe_dump({"services": {"engine": {"image": image}}}),
    }
    for name, text in files.items():
        if name not in skip:
            (engine / name).write_text(text)
    (engine / "configs" / "ecommerce" / "product-search.yaml").write_text(yaml.safe_dump(config))
    return tmp_path


def lint(root):
    return lint_repo(root, queries=lambda scenario_id: QUERIES)


def rules(findings):
    return sorted(f.rule for f in findings)


def test_clean_config_has_no_findings(tmp_path):
    n, findings = lint(make_repo(tmp_path, {"default": {"index": {}, "search": {}}}))
    assert (n, findings) == (1, [])


def test_setting_without_justification_is_flagged(tmp_path):
    _, findings = lint(make_repo(tmp_path, {"default": {"index": {"ef": 64}}}))
    assert rules(findings) == ["unjustified-setting"]
    assert "index.ef" in findings[0].message and "[default]" in findings[0].message
    assert findings[0].path == "engines/eng/configs/ecommerce/product-search.yaml"


@pytest.mark.parametrize("key", ["ef", "index.ef", "index", "hnsw"])
def test_justification_by_key_path_or_prefix(tmp_path, key):
    index = {"hnsw": {"ef": 64}} if key == "hnsw" else {"ef": 64}
    config = {"default": {"index": index, "justification": {key: "documented default for v1"}}}
    assert lint(make_repo(tmp_path, config))[1] == []


def test_search_settings_need_justification(tmp_path):
    config = {"default": {"search": {"semantic": {"ef": 128}}, "justification": {"ef": "x"}}}
    assert lint(make_repo(tmp_path, config))[1] == []
    config = {"default": {"search": {"semantic": {"ef": 128}}}}
    _, findings = lint(make_repo(tmp_path / "b", config))
    assert "search.semantic.ef" in findings[0].message


def test_tuned_track_is_checked_after_merge(tmp_path):
    config = {"default": {}, "tuned": {"index": {"m": 32}}}
    _, findings = lint(make_repo(tmp_path, config))
    assert rules(findings) == ["unjustified-setting"]
    assert "[tuned]" in findings[0].message


def test_blank_justification_does_not_count(tmp_path):
    config = {"default": {"index": {"ef": 64}, "justification": {"ef": "  "}}}
    assert rules(lint(make_repo(tmp_path, config))[1]) == ["unjustified-setting"]


@pytest.mark.parametrize("key,ok", [
    ("hybrid", True), ("faceted:semantic", True), ("*:keyword", True),
    ("semantik", False), ("filtered:filtered", False), ("*", False), ("keyword:", False),
])
def test_unsupported_keys_must_name_modes(tmp_path, key, ok):
    _, findings = lint(make_repo(tmp_path, {"default": {"unsupported": {key: "reason"}}}))
    assert rules(findings) == ([] if ok else ["unsupported-key"])


@pytest.mark.parametrize("value", [
    "salon chair",           # a query text
    "Salon  Chair",          # same after normalization
    "best salon chair deals",  # a multi-word query inside a longer string
    "0-typo",                # a variant id
    "0-typo:keyword:keyword",  # a task id
])
def test_query_text_and_ids_are_rejected(tmp_path, value):
    config = {"default": {"search": {"keyword": {"boost": [value]}}, "justification": {"boost": "x"}}}
    _, findings = lint(make_repo(tmp_path, config))
    assert rules(findings) == ["query-leak"]
    assert "search.keyword.boost[0]" in findings[0].message


def test_query_text_as_a_key_is_rejected(tmp_path):
    config = {"default": {"index": {"synonyms": {"sofa": ["couch"]}}, "justification": {"synonyms": "x"}}}
    assert rules(lint(make_repo(tmp_path, config))[1]) == ["query-leak"]


@pytest.mark.parametrize("value", ["salon", "0", "chairs"])
def test_ordinary_values_are_not_leaks(tmp_path, value):
    # A word that is only part of a query, and purely numeric ids, are ordinary settings.
    config = {"default": {"search": {"keyword": {"x": value}}, "justification": {"x": "x"}}}
    assert lint(make_repo(tmp_path, config))[1] == []


def test_prose_fields_are_not_checked_for_leaks(tmp_path):
    config = {"default": {"justification": {"x": "helps salon chair queries"},
                          "unsupported": {"hybrid": "salon chair"}}}
    assert lint(make_repo(tmp_path, config))[1] == []


@pytest.mark.parametrize("image", ["getmeili/meilisearch", "getmeili/meilisearch:latest", "", None])
def test_engine_image_must_pin_a_version(tmp_path, image):
    _, findings = lint(make_repo(tmp_path, {"default": {}}, image=image))
    assert rules(findings) == ["image-pin"]


def test_image_with_digest_is_accepted(tmp_path):
    image = "qdrant/qdrant:v1.19.2@sha256:" + "a" * 64
    assert lint(make_repo(tmp_path, {"default": {}}, image=image))[1] == []


def test_missing_engine_files_are_flagged(tmp_path):
    _, findings = lint(make_repo(tmp_path, {"default": {}}, skip=("README.md",)))
    assert rules(findings) == ["engine-files"]
    assert findings[0].path == "engines/eng/README.md"


def test_unloadable_config_is_a_finding_not_a_crash(tmp_path):
    root = make_repo(tmp_path, {"default": {}})
    (root / "engines/eng/configs/ecommerce/product-search.yaml").write_text("default: [unclosed\n")
    assert rules(lint(root)[1]) == ["config-load"]


def test_unprepared_scenario_is_a_finding(tmp_path):
    root = make_repo(tmp_path, {"default": {}})
    _, findings = lint_repo(root)  # no injected queries: reads .irbench/, which is empty
    assert rules(findings) == ["not-prepared"]
    assert "irbench prepare --scenario ecommerce/product-search" in findings[0].message


def test_repository_configs_pass_the_static_rules():
    n, findings = lint_repo(REPO_ROOT, queries=lambda s: QuerySet.of([], []))
    assert n >= 3 and findings == [], [str(f) for f in findings]


def _write_prepared(root, rows, task_ids):
    d = root / ".irbench" / "scenarios" / "ecommerce__product-search"
    d.mkdir(parents=True)
    (d / "queries.variants.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "tasks.jsonl").write_text("".join(json.dumps({"task_id": t}) + "\n" for t in task_ids))


def test_cli_exit_codes(tmp_path):
    root = make_repo(tmp_path, {"default": {"search": {"keyword": {"q": "salon chair"}},
                                            "justification": {"q": "x"}}})
    _write_prepared(root, [{"qid": "0", "text": "salon chair"}], ["0:keyword:keyword"])
    bad = CliRunner().invoke(app, ["lint", "--root", str(root)])
    assert bad.exit_code == 1
    assert "[query-leak]" in bad.output and "1 configs, 1 findings" in bad.output

    (root / "engines/eng/configs/ecommerce/product-search.yaml").write_text(
        yaml.safe_dump({"default": {}})
    )
    good = CliRunner().invoke(app, ["lint", "--root", str(root)])
    assert good.exit_code == 0 and "1 configs, 0 findings" in good.output
```

- [ ] **Step 6: Run them to verify they fail**

Run: `uv run pytest harness/tests/test_lint.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'irbench.lint'`.

- [ ] **Step 7: Implement the lint**

Create `harness/irbench/lint.py`:

```python
"""Machine-checkable parity rules for engine configs (spec §7), run in CI by `irbench lint`."""

import functools
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import yaml

from irbench.engines.config import TRACKS, TrackConfig, load_engine_config
from irbench.io import read_jsonl
from irbench.models import BASE_MODES, Mode
from irbench.paths import Workspace

MODES = get_args(Mode)
REQUIRED_FILES = ("adapter.py", "README.md", "docker/compose.yaml")
# name:tag, optionally @sha256:<digest>; `latest` is rejected separately.
IMAGE_RE = re.compile(
    r"^[a-z0-9][a-z0-9._/-]*:(?P<tag>[A-Za-z0-9_][A-Za-z0-9_.-]*)(@sha256:[0-9a-f]{64})?$"
)


@dataclass(frozen=True)
class Finding:
    path: str
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: [{self.rule}] {self.message}"


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass(frozen=True)
class QuerySet:
    texts: frozenset[str]
    ids: frozenset[str]

    @classmethod
    def of(cls, texts: list[str], ids: list[str]) -> "QuerySet":
        # Purely numeric ids ("0", "17") are skipped: they collide with ordinary numeric
        # settings. Variant and task ids ("0-typo", "0:keyword:keyword") are distinctive.
        return cls(
            frozenset(normalize(t) for t in texts), frozenset(i for i in ids if not i.isdigit())
        )

    @classmethod
    def load(cls, root: Path, scenario_id: str) -> "QuerySet":
        d = Workspace(root).scenario_dir(scenario_id)
        variants, tasks = d / "queries.variants.jsonl", d / "tasks.jsonl"
        if not variants.exists() or not tasks.exists():
            raise FileNotFoundError(
                f"{d} is not prepared: run `irbench prepare --scenario {scenario_id}` first"
            )
        rows = list(read_jsonl(variants))
        ids = [r["qid"] for r in rows] + [t["task_id"] for t in read_jsonl(tasks)]
        return cls.of([r["text"] for r in rows], ids)

    @functools.cached_property
    def multiword(self) -> list[tuple[str, re.Pattern[str]]]:
        # cached_property writes to __dict__ directly, so it works on a frozen dataclass.
        return [
            (t, re.compile(rf"(?<!\w){re.escape(t)}(?!\w)"))
            for t in sorted(self.texts)
            if " " in t
        ]

    def leak(self, value: str) -> str | None:
        norm = normalize(value)
        if norm in self.texts:
            return f"equals query text {norm!r}"
        if value in self.ids:
            return f"is a query or task id ({value!r})"
        for text, pattern in self.multiword:
            if pattern.search(norm):
                return f"contains query text {text!r}"
        return None


QueryLoader = Callable[[str], QuerySet]


def lint_repo(root: Path, queries: QueryLoader | None = None) -> tuple[int, list[Finding]]:
    """Returns (number of configs checked, findings)."""
    load = functools.cache(queries or (lambda scenario_id: QuerySet.load(root, scenario_id)))
    findings: list[Finding] = []
    n_configs = 0
    engine_dirs = (p for p in (root / "engines").iterdir() if p.is_dir() and p.name[0] not in "._")
    for engine_dir in sorted(engine_dirs):
        findings += _check_engine_files(root, engine_dir)
        for path in sorted((engine_dir / "configs").rglob("*.yaml")):
            n_configs += 1
            scenario_id = path.relative_to(engine_dir / "configs").with_suffix("").as_posix()
            findings += _check_config(root, engine_dir.name, scenario_id, path, load)
    return n_configs, findings


def _check_engine_files(root: Path, engine_dir: Path) -> list[Finding]:
    rel = engine_dir.relative_to(root).as_posix()
    out = [
        Finding(f"{rel}/{name}", "engine-files", "missing: every engine needs an adapter, "
                "a README with parity notes and a pinned compose file")
        for name in REQUIRED_FILES
        if not (engine_dir / name).exists()
    ]
    compose = engine_dir / "docker" / "compose.yaml"
    if compose.exists():
        data = yaml.safe_load(compose.read_text(encoding="utf-8")) or {}
        image = data.get("services", {}).get("engine", {}).get("image")
        match = IMAGE_RE.match(image or "")
        if not match or match["tag"] == "latest":
            out.append(Finding(f"{rel}/docker/compose.yaml", "image-pin",
                               f"services.engine.image must pin a version tag, got {image!r}"))
    return out


def _check_config(
    root: Path, engine: str, scenario_id: str, path: Path, load: QueryLoader
) -> list[Finding]:
    rel = path.relative_to(root).as_posix()
    try:
        config = load_engine_config(root, engine, scenario_id)
    except (ValueError, yaml.YAMLError) as e:
        return [Finding(rel, "config-load", str(e))]

    tracks = [(name, config.track(name)) for name in TRACKS if name in config.raw]
    out: list[Finding] = []
    for name, track in tracks:
        for setting in _settings(track):
            if not _justified(setting, track.justification):
                out.append(Finding(rel, "unjustified-setting",
                                   f"[{name}] {setting} has no `justification:` entry; every "
                                   "setting that departs from the engine as shipped is explained"))
        for key in track.unsupported:
            if not _valid_unsupported_key(key):
                out.append(Finding(rel, "unsupported-key",
                                   f"[{name}] {key!r} is not `<mode>`, `<mode>:<base>` or "
                                   "`*:<base>`"))
    try:
        query_set = load(scenario_id)
    except FileNotFoundError as e:
        return [*out, Finding(rel, "not-prepared", str(e))]
    for name, track in tracks:
        # Only index and search reach the engine; justification and N/A reasons are prose.
        for where, value in _strings({"index": track.index, "search": track.search}, ""):
            reason = query_set.leak(value)
            if reason:
                out.append(Finding(rel, "query-leak",
                                   f"[{name}] {where} {reason}: per-query tuning is not allowed"))
    return out


def _settings(track: TrackConfig) -> Iterator[str]:
    yield from _leaves(track.index, "index")
    yield from _leaves(track.search, "search")


def _leaves(data: dict[str, Any], prefix: str) -> Iterator[str]:
    for key, value in data.items():
        path = f"{prefix}.{key}"
        if isinstance(value, dict):
            yield from _leaves(value, path)  # an empty dict changes nothing
        else:
            yield path


def _justified(setting: str, justification: dict[str, str]) -> bool:
    # A justification key names the setting or a parent, with or without the leading
    # "index." / "search." / "search.<base>.": "index.hnsw.ef" is covered by index,
    # index.hnsw, index.hnsw.ef, hnsw or hnsw.ef; "search.semantic.ef" also by ef.
    parts = setting.split(".")
    starts = (0, 1, 2) if parts[0] == "search" else (0, 1)
    candidates = {".".join(parts[s:e]) for s in starts for e in range(s + 1, len(parts) + 1)}
    return any(justification.get(c, "").strip() for c in candidates)


def _valid_unsupported_key(key: str) -> bool:
    mode, sep, base = key.partition(":")
    if not sep:
        return mode in MODES
    return (mode == "*" or mode in MODES) and base in BASE_MODES


def _strings(value: Any, path: str) -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            yield child, str(key)
            yield from _strings(item, child)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _strings(item, f"{path}[{i}]")
    elif isinstance(value, str):
        yield path, value
```

In `harness/irbench/cli.py`, add the import `from irbench.lint import lint_repo` and the command:

```python
@app.command()
def lint(root: Path = ROOT_OPTION) -> None:
    """Check engine configs against the machine-checkable parity rules (spec §7)."""
    n, findings = lint_repo(root)
    for finding in findings:
        typer.echo(str(finding), err=True)
    typer.echo(f"lint: {n} configs, {len(findings)} findings")
    if findings:
        raise typer.Exit(code=1)
```

- [ ] **Step 8: Run the lint tests and the suite**

Run: `uv run pytest harness/tests/test_lint.py -v` then `uv run pytest`
Expected: all pass. `CliRunner` mixes stderr into `output` by default in the installed typer; if `bad.output` lacks `[query-leak]`, check `CliRunner(mix_stderr=...)` support for the installed click version and assert on `result.stderr` instead.

- [ ] **Step 9: Run the lint on the real repo**

Run: `uv run irbench lint`
Expected: `lint: 3 configs, 0 findings` (WANDS is prepared in `.irbench/` from earlier plans). Any finding on a real config is a real parity problem: report it, do not weaken the rule.

- [ ] **Step 10: Lint and commit**

```bash
uv run ruff check --fix . && uv run ruff format harness engines
git add harness/irbench/engines/config.py harness/irbench/stages/run.py harness/irbench/lint.py \
  harness/irbench/cli.py harness/tests/test_engine_config.py harness/tests/test_run_pipeline.py \
  harness/tests/test_lint.py
git commit -m "feat: parity lint for engine configs and credits for tuned tracks"
```

---

### Task 3: Publish run bundles to `results/`

Spec §4: `results/<run-id>/` holds published outputs. Spec §7: only reference-machine results are published.

**Files:**
- Create: `harness/irbench/publish.py`
- Create: `results/README.md`
- Modify: `harness/irbench/cli.py`
- Test: `harness/tests/test_publish.py`

**Interfaces:**
- Consumes: `Workspace(root).run_dir(run_id)`; run files `manifest.json` (with `hardware.id`, `status`), `scores.json`, `perf.json`, `index.json`.
- Produces: `BUNDLE_FILES`, `publish_run(root, run_id, results_dir=None) -> Path`; CLI `irbench publish RUN_ID... [--results DIR]`; a bundle directory that Task 4's `load_results` reads.

- [ ] **Step 1: Write the failing tests**

Create `harness/tests/test_publish.py`:

```python
import json

import pytest
from typer.testing import CliRunner

from irbench.cli import app
from irbench.publish import BUNDLE_FILES, publish_run

RUN_ID = "ecommerce__product-search__meilisearch__bge-small-en-v1.5__default__aws-c7i-4xlarge"


def make_run(root, run_id=RUN_ID, profile="aws-c7i-4xlarge", status="succeeded",
             files=("scores.json", "perf.json", "index.json")):
    d = root / ".irbench" / "runs" / run_id
    (d / "rendered" / "keyword").mkdir(parents=True, exist_ok=True)
    (d / "rendered" / "keyword" / "0.json").write_text("{}")
    (d / "run.trec").write_text("0 Q0 d1 1 1.0 x\n")
    manifest = {"run_id": run_id, "status": status, "hardware": {"id": profile}}
    (d / "manifest.json").write_text(json.dumps(manifest))
    for name in files:
        (d / name).write_text(json.dumps({"file": name}))
    return d


def test_publish_copies_only_the_bundle(tmp_path):
    make_run(tmp_path)
    dest = publish_run(tmp_path, RUN_ID)
    assert dest == tmp_path / "results" / RUN_ID
    assert sorted(p.name for p in dest.iterdir()) == sorted(BUNDLE_FILES)
    assert json.loads((dest / "perf.json").read_text()) == {"file": "perf.json"}


def test_local_profile_is_never_published(tmp_path):
    make_run(tmp_path, profile="local")
    with pytest.raises(ValueError, match="reference"):
        publish_run(tmp_path, RUN_ID)
    assert not (tmp_path / "results").exists()


def test_failed_run_is_published_with_what_it_has(tmp_path):
    make_run(tmp_path, status="failed", files=())
    dest = publish_run(tmp_path, RUN_ID)
    assert [p.name for p in dest.iterdir()] == ["manifest.json"]


def test_republishing_replaces_stale_files(tmp_path):
    make_run(tmp_path)
    publish_run(tmp_path, RUN_ID)
    make_run(tmp_path, files=("scores.json",))  # rerun without perf/index
    (tmp_path / ".irbench/runs" / RUN_ID / "perf.json").unlink()
    (tmp_path / ".irbench/runs" / RUN_ID / "index.json").unlink()
    dest = publish_run(tmp_path, RUN_ID)
    assert sorted(p.name for p in dest.iterdir()) == ["manifest.json", "scores.json"]


def test_unknown_run_names_the_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="no manifest.json"):
        publish_run(tmp_path, "nope")


def test_cli_publishes_several_runs_and_fails_on_local(tmp_path):
    make_run(tmp_path)
    local_id = RUN_ID.replace("aws-c7i-4xlarge", "local")
    make_run(tmp_path, run_id=local_id, profile="local")
    ok = CliRunner().invoke(app, ["publish", RUN_ID, "--root", str(tmp_path)])
    assert ok.exit_code == 0 and f"results/{RUN_ID}" in ok.output
    bad = CliRunner().invoke(app, ["publish", RUN_ID, local_id, "--root", str(tmp_path)])
    assert bad.exit_code == 1 and "local" in bad.output
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest harness/tests/test_publish.py -v`
Expected: `ModuleNotFoundError: No module named 'irbench.publish'`.

- [ ] **Step 3: Implement**

Create `harness/irbench/publish.py`:

```python
"""Copy a run's outputs into results/<run-id>/ (spec §4). Only reference-machine runs are
published (spec §7); failed runs are published too, so the report shows the failure (§9)."""

import shutil
from pathlib import Path

from irbench.io import read_json
from irbench.paths import Workspace

# Small, self-contained files. rendered/ and run.trec are reproducible from the manifest.
BUNDLE_FILES = ("manifest.json", "scores.json", "perf.json", "index.json")
UNPUBLISHABLE_PROFILES = ("local",)


def publish_run(root: Path, run_id: str, results_dir: Path | None = None) -> Path:
    src = Workspace(root).run_dir(run_id)
    if not (src / "manifest.json").exists():
        raise FileNotFoundError(f"{src} has no manifest.json: not a run directory")
    profile = read_json(src / "manifest.json")["hardware"]["id"]
    if profile in UNPUBLISHABLE_PROFILES:
        raise ValueError(
            f"{run_id} ran on the {profile!r} profile; only reference-machine results are "
            "published. Preview it with `irbench report --results .irbench/runs`."
        )
    dest = (results_dir or root / "results") / run_id
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for name in BUNDLE_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, tmp / name)
    shutil.rmtree(dest, ignore_errors=True)
    tmp.replace(dest)
    return dest
```

In `harness/irbench/cli.py`, add `from irbench.publish import publish_run` and:

```python
@app.command()
def publish(
    run_ids: list[str] = typer.Argument(..., help="Run ids (folder names under .irbench/runs)."),
    results: Path | None = typer.Option(None, "--results", help="Default: <root>/results."),
    root: Path = ROOT_OPTION,
) -> None:
    """Copy finished runs into results/ (reference hardware profiles only)."""
    failed = False
    for run_id in run_ids:
        try:
            dest = publish_run(root, run_id, results)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"{run_id}: {e}", err=True)
            failed = True
        else:
            typer.echo(f"published {dest.relative_to(root) if dest.is_relative_to(root) else dest}")
    if failed:
        raise typer.Exit(code=1)
```

Create `results/README.md`:

```markdown
# Published results

One folder per run, named by run id
(`<scenario>__<engine>__<model>__<track>__<hardware profile>`), written by
`irbench publish`. Each folder holds `manifest.json`, `scores.json`, `perf.json` and
`index.json` when the run produced them. A failed run still has its manifest, so
the report can show the failure.

Only runs on a reference hardware profile are published. Results from different
profiles are never compared. The report at the repository's GitHub Pages site is
built from this folder by `irbench report --results results --out site`.
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest harness/tests/test_publish.py -v` then `uv run pytest`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix . && uv run ruff format harness engines
git add harness/irbench/publish.py harness/irbench/cli.py harness/tests/test_publish.py results/README.md
git commit -m "feat: publish run bundles to results/ for reference profiles only"
```

---

### Task 4: Load result bundles for the report

**Files:**
- Create: `harness/irbench/report/__init__.py` (empty), `harness/irbench/report/data.py`
- Create: `harness/tests/report_fixtures.py`
- Modify: `pyproject.toml` (ruff `known-first-party` gains `"report_fixtures"`)
- Test: `harness/tests/test_report_data.py`

**Interfaces:**
- Consumes: bundle directories (Task 3) or full run directories (`.irbench/runs/<id>/`): `manifest.json` (`run_id`, `scenario`, `engine`, `model`, `track`, `hardware.id`, `status`, `stages`, `config`), `scores.json` (`relevance[]` with `mode`, `base`, `perturbation`, `n`, `metrics`; `not_applicable` and `constraint_violations` keyed `mode:base`; `checks[]`; `perf`), `perf.json` (`limits`, `latency`, `throughput`, `skipped_groups`), `index.json` (`seconds`, `memory_peak_bytes`, `disk_bytes`, `n_docs`).
- Produces (used by Tasks 5 and 6):
  - `MODES`, `PERTURBATIONS = ("original", "typo", "prefix", "synonym")`, `group_key(group) -> tuple[int, int]`.
  - `RunResult` (frozen dataclass): `run_id, scenario, engine, model, track, hardware, status, failure, manifest, scores, perf, index`; `.label`, `.groups`, `.ndcg(group, perturbation="all")`, `.na_reason(group)`, `.violation(group)`. Groups are spelled `"<mode>/<base>"`.
  - `ScenarioResults(scenario, hardware, runs)`: `.slug`, `.models`, `.groups`.
  - `Unreadable(path, error)`.
  - `load_results(results_dir) -> tuple[list[ScenarioResults], list[Unreadable]]`.
  - Test helper `report_fixtures.write_bundle(...)`.

- [ ] **Step 1: Write the fixture helper**

Create `harness/tests/report_fixtures.py`:

```python
"""Realistic result bundles for report tests (shapes copied from real Plan 3 runs)."""

import json
from pathlib import Path

PERTURBATIONS = ("all", "original", "typo", "prefix", "synonym")


def relevance(groups: dict[str, float]) -> list[dict]:
    """{"keyword/keyword": 0.5} -> score rows for every perturbation (typo -0.1, prefix -0.2)."""
    rows = []
    for group, ndcg in groups.items():
        mode, base = group.split("/")
        drop = {"all": 0.0, "original": 0.05, "typo": -0.1, "prefix": -0.2, "synonym": -0.05}
        for p in PERTURBATIONS:
            value = round(ndcg + drop[p], 4)
            rows.append({"mode": mode, "base": base, "perturbation": p, "n": 100, "metrics": {
                "ndcg@10": value, "recall@10": value / 2, "recall@100": value,
                "mrr@10": value, "success@1": value / 3}})
    return rows


def steps(qps_per_client: float, valid_until: int) -> list[dict]:
    out = []
    for c in (1, 2, 4, 8):
        valid = c <= valid_until
        out.append({"concurrency": c, "seconds": 20.0, "requests": int(qps_per_client * c * 20),
                    "errors": 0, "qps": qps_per_client * c, "error_rate": 0.0,
                    "p50_ms": 2.0, "p95_ms": 5.0, "p99_ms": 20.0 if valid else 150.0,
                    "valid": valid})
        if not valid:
            break
    return out


def perf(throughput: dict[str, tuple[float, int]], skipped=(), latency_groups=None) -> dict:
    groups = latency_groups or list(throughput)
    latency = {"all": {"count": 1, "errors": 0, "mean_ms": 1.0, "p50_ms": 1.0, "p95_ms": 1.0,
                       "p99_ms": 1.0, "max_ms": 1.0}}
    for g in groups:
        latency[g] = {"count": 100, "errors": 0, "mean_ms": 3.0, "p50_ms": 2.5, "p95_ms": 7.0,
                      "p99_ms": 12.25, "max_ms": 30.0}
    tp = {}
    for g, (qps, valid_until) in throughput.items():
        s = steps(qps, valid_until)
        valid = [x for x in s if x["valid"]]
        best = max(valid, key=lambda x: x["qps"]) if valid else None
        tp[g] = {"steps": s, "max_qps": best and best["qps"],
                 "max_qps_concurrency": best and best["concurrency"]}
    return {"base_url": "http://localhost:1", "requests": 100,
            "limits": {"p99_ms": 100.0, "max_error_rate": 0.001},
            "warmup": {"requests": 100, "errors": 0, "seconds": 1.0},
            "latency": latency, "throughput": tp, "skipped_groups": list(skipped)}


def write_bundle(
    results: Path, engine: str, *, model="bge-small-en-v1.5", track="default",
    hardware="aws-c7i-4xlarge", scenario="ecommerce/product-search", status="succeeded",
    groups=None, na=None, violations=None, perf_report=None, index=True, failed_stage=None,
    config_diff=None, credits=None,
) -> Path:
    run_id = "__".join([scenario.replace("/", "__"), engine, model, track, hardware])
    d = results / run_id
    d.mkdir(parents=True)
    stages = {"prepare": {"status": "succeeded", "seconds": 1.0}}
    if failed_stage:
        stages[failed_stage] = {"status": "failed", "seconds": 2.0, "error": "RuntimeError: boom"}
    manifest = {
        "run_id": run_id, "scenario": scenario, "engine": engine, "model": model,
        "track": track, "hardware": {"id": hardware}, "status": status, "stages": stages,
        "git_sha": "abc123", "irbench_version": "0.1.0",
        "engine_image": {"image": f"{engine}:1", "digest": f"{engine}@sha256:00"},
        "config": {"path": f"engines/{engine}/configs/{scenario}.yaml", "hash": "f" * 64,
                   "diff_vs_default": config_diff or {}, "credits": credits or []},
        "dataset": {"id": "wands", "version": "3b74dcf4"}, "seed": 20261009,
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    if status == "succeeded":
        scores = {
            "run_id": run_id, "scenario": scenario, "engine": engine, "model": model,
            "track": track,
            "relevance": relevance(groups or {"keyword/keyword": 0.5}),
            "checks": [], "constraint_violations": violations or {},
            "not_applicable": na or {}, "errors": 0, "perf": None,
        }
        (d / "scores.json").write_text(json.dumps(scores))
        if perf_report is not None:
            (d / "perf.json").write_text(json.dumps(perf_report))
        if index:
            (d / "index.json").write_text(json.dumps(
                {"seconds": 42.5, "memory_peak_bytes": 1566838784, "disk_bytes": 727293952,
                 "n_docs": 42994}))
    return d
```

Add `"report_fixtures"` to `known-first-party` in `pyproject.toml`.

- [ ] **Step 2: Write the failing tests**

Create `harness/tests/test_report_data.py`:

```python
import pytest

from irbench.report.data import group_key, load_results
from report_fixtures import write_bundle


def test_runs_are_grouped_by_scenario_and_profile(tmp_path):
    write_bundle(tmp_path, "meilisearch")
    write_bundle(tmp_path, "qdrant")
    write_bundle(tmp_path, "meilisearch", model="text-embedding-3-small")
    scenarios, unreadable = load_results(tmp_path)
    assert unreadable == []
    assert [(s.scenario, s.hardware) for s in scenarios] == [
        ("ecommerce/product-search", "aws-c7i-4xlarge")
    ]
    s = scenarios[0]
    assert [(r.model, r.engine) for r in s.runs] == [
        ("bge-small-en-v1.5", "meilisearch"), ("bge-small-en-v1.5", "qdrant"),
        ("text-embedding-3-small", "meilisearch"),
    ]
    assert s.models == ["bge-small-en-v1.5", "text-embedding-3-small"]
    assert s.slug == "ecommerce__product-search__aws-c7i-4xlarge"


def test_profiles_are_never_mixed(tmp_path):
    write_bundle(tmp_path, "meilisearch", hardware="aws-c7i-4xlarge")
    write_bundle(tmp_path, "meilisearch", hardware="local")
    scenarios, _ = load_results(tmp_path)
    assert [(s.hardware, len(s.runs)) for s in scenarios] == [("aws-c7i-4xlarge", 1), ("local", 1)]


def test_lookups(tmp_path):
    write_bundle(
        tmp_path, "qdrant", track="tuned",
        groups={"semantic/semantic": 0.6, "sorted/semantic": 0.4},
        na={"hybrid:hybrid": "no keyword retrieval"},
        violations={"sorted:semantic": "3 order violations"},
    )
    run = load_results(tmp_path)[0][0].runs[0]
    assert run.label == "qdrant (tuned)"
    assert run.groups == ["semantic/semantic", "hybrid/hybrid", "sorted/semantic"]
    assert run.ndcg("semantic/semantic") == pytest.approx(0.6)
    assert run.ndcg("semantic/semantic", "typo") == pytest.approx(0.5)
    assert run.ndcg("keyword/keyword") is None
    assert run.na_reason("hybrid/hybrid") == "no keyword retrieval"
    assert run.violation("sorted/semantic") == "3 order violations"
    assert run.violation("semantic/semantic") is None


def test_groups_follow_mode_then_base_order():
    groups = ["faceted/keyword", "hybrid/hybrid", "filtered/semantic", "keyword/keyword",
              "filtered/keyword", "semantic/semantic"]
    assert sorted(groups, key=group_key) == [
        "keyword/keyword", "semantic/semantic", "hybrid/hybrid", "filtered/keyword",
        "filtered/semantic", "faceted/keyword",
    ]


def test_failed_run_reports_the_failed_stage(tmp_path):
    write_bundle(tmp_path, "elasticsearch", status="failed", failed_stage="index")
    run = load_results(tmp_path)[0][0].runs[0]
    assert run.status == "failed"
    assert run.failure == "index: RuntimeError: boom"
    assert run.scores is None and run.groups == []


def test_failed_run_without_failed_stage(tmp_path):
    write_bundle(tmp_path, "elasticsearch", status="interrupted")
    run = load_results(tmp_path)[0][0].runs[0]
    assert run.failure == "run status 'interrupted'"


def test_succeeded_without_scores_is_incomplete(tmp_path):
    d = write_bundle(tmp_path, "meilisearch")
    (d / "scores.json").unlink()
    run = load_results(tmp_path)[0][0].runs[0]
    assert run.status == "incomplete"
    assert "scores.json" in run.failure


def test_corrupt_bundle_is_reported_not_fatal(tmp_path):
    write_bundle(tmp_path, "meilisearch")
    bad = write_bundle(tmp_path, "qdrant")
    (bad / "scores.json").write_text("{not json")
    lacking = tmp_path / "lacking"
    lacking.mkdir()
    (lacking / "manifest.json").write_text('{"run_id": "x"}')
    scenarios, unreadable = load_results(tmp_path)
    assert [r.engine for r in scenarios[0].runs] == ["meilisearch"]
    assert sorted(u.path for u in unreadable) == sorted(["lacking", bad.name])
    assert any("lacks" in u.error for u in unreadable)


def test_non_run_entries_are_ignored(tmp_path):
    write_bundle(tmp_path, "meilisearch")
    (tmp_path / "README.md").write_text("x")
    (tmp_path / "empty").mkdir()
    scenarios, unreadable = load_results(tmp_path)
    assert len(scenarios[0].runs) == 1 and unreadable == []


def test_missing_results_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path / "nope")
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest harness/tests/test_report_data.py -v`
Expected: `ModuleNotFoundError: No module named 'irbench.report'`.

- [ ] **Step 4: Implement**

Create an empty `harness/irbench/report/__init__.py` and `harness/irbench/report/data.py`:

```python
"""Load run bundles (results/<run-id>/ or .irbench/runs/<run-id>/) for the report."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

from irbench.io import read_json
from irbench.models import BASE_MODES, Mode
from irbench.paths import slug

MODES: tuple[str, ...] = get_args(Mode)
PERTURBATIONS = ("original", "typo", "prefix", "synonym")
REQUIRED_KEYS = ("run_id", "scenario", "engine", "model", "track", "hardware", "status")


def group_key(group: str) -> tuple[int, int]:
    mode, _, base = group.partition("/")
    return MODES.index(mode), BASE_MODES.index(base)


@dataclass(frozen=True)
class RunResult:
    run_id: str
    scenario: str
    engine: str
    model: str
    track: str
    hardware: str
    status: str
    failure: str | None
    manifest: dict[str, Any]
    scores: dict[str, Any] | None = None
    perf: dict[str, Any] | None = None
    index: dict[str, Any] | None = None

    @property
    def label(self) -> str:
        return self.engine if self.track == "default" else f"{self.engine} ({self.track})"

    @property
    def groups(self) -> list[str]:
        if self.scores is None:
            return []
        found = {f"{g['mode']}/{g['base']}" for g in self.scores["relevance"]}
        found |= {key.replace(":", "/") for key in self.scores["not_applicable"]}
        return sorted(found, key=group_key)

    def ndcg(self, group: str, perturbation: str = "all") -> float | None:
        if self.scores is None:
            return None
        mode, _, base = group.partition("/")
        for g in self.scores["relevance"]:
            if (g["mode"], g["base"], g["perturbation"]) == (mode, base, perturbation):
                return g["metrics"]["ndcg@10"]
        return None

    def na_reason(self, group: str) -> str | None:
        if self.scores is None:
            return None
        return self.scores["not_applicable"].get(group.replace("/", ":"))

    def violation(self, group: str) -> str | None:
        if self.scores is None:
            return None
        return self.scores["constraint_violations"].get(group.replace("/", ":"))


@dataclass(frozen=True)
class ScenarioResults:
    scenario: str
    hardware: str
    runs: list[RunResult]

    @property
    def slug(self) -> str:
        return f"{slug(self.scenario)}__{self.hardware}"

    @property
    def models(self) -> list[str]:
        return sorted({r.model for r in self.runs})

    @property
    def groups(self) -> list[str]:
        return sorted({g for r in self.runs for g in r.groups}, key=group_key)


@dataclass(frozen=True)
class Unreadable:
    path: str
    error: str


def load_results(results_dir: Path) -> tuple[list[ScenarioResults], list[Unreadable]]:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"{results_dir} is not a directory")
    runs: list[RunResult] = []
    unreadable: list[Unreadable] = []
    for d in sorted(p for p in results_dir.iterdir() if (p / "manifest.json").is_file()):
        try:
            runs.append(_load_run(d))
        except (ValueError, KeyError, TypeError) as e:  # JSONDecodeError is a ValueError
            unreadable.append(Unreadable(d.name, f"{type(e).__name__}: {e}"))

    grouped: dict[tuple[str, str], list[RunResult]] = defaultdict(list)
    for run in runs:
        grouped[(run.scenario, run.hardware)].append(run)
    scenarios = [
        ScenarioResults(scenario, hardware, sorted(rs, key=lambda r: (r.model, r.engine, r.track)))
        for (scenario, hardware), rs in sorted(grouped.items())
    ]
    return scenarios, unreadable


def _load_run(d: Path) -> RunResult:
    manifest = read_json(d / "manifest.json")
    missing = [k for k in REQUIRED_KEYS if k not in manifest]
    if missing:
        raise ValueError(f"manifest.json lacks {', '.join(missing)}")
    files = {
        name: read_json(d / f"{name}.json") if (d / f"{name}.json").exists() else None
        for name in ("scores", "perf", "index")
    }
    status, failure = manifest["status"], _failure(manifest)
    if status == "succeeded" and files["scores"] is None:
        status, failure = "incomplete", "the run succeeded but the bundle has no scores.json"
    return RunResult(
        run_id=manifest["run_id"],
        scenario=manifest["scenario"],
        engine=manifest["engine"],
        model=manifest["model"],
        track=manifest["track"],
        hardware=manifest["hardware"]["id"],
        status=status,
        failure=failure,
        manifest=manifest,
        **files,
    )


def _failure(manifest: dict[str, Any]) -> str | None:
    if manifest["status"] == "succeeded":
        return None
    for stage, info in manifest.get("stages", {}).items():
        if info.get("status") == "failed":
            return f"{stage}: {info.get('error', 'no error recorded')}"
    return f"run status {manifest['status']!r}"
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest harness/tests/test_report_data.py -v`
Expected: all pass.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix . && uv run ruff format harness engines
git add pyproject.toml harness/irbench/report harness/tests/report_fixtures.py \
  harness/tests/test_report_data.py
git commit -m "feat: load published result bundles for the report"
```

---

### Task 5: HTML report pages and `irbench report`

Spec §11: per scenario an engines × modes nDCG@10 matrix, the perturbation breakdown and a model comparison; per run the config diff, manifest, N/A reasons and credits; links to the methodology docs. Performance comes in Task 6.

**Files:**
- Modify: `pyproject.toml` (dependency `jinja2>=3.1`)
- Create: `harness/irbench/report/render.py`
- Create: `harness/irbench/report/static/style.css`
- Create: `harness/irbench/report/templates/base.html.j2`, `index.html.j2`, `scenario.html.j2`, `run.html.j2`
- Modify: `harness/irbench/cli.py`
- Test: `harness/tests/test_report_render.py`

**Interfaces:**
- Consumes: Task 4's `load_results`, `RunResult`, `ScenarioResults`, `Unreadable`, `PERTURBATIONS`, `group_key`.
- Produces (Task 6 extends these):
  - `Cell(text, kind, note="")`, where `kind` is one of `score`, `na`, `violated`, `missing`, `failed`.
  - `relevance_cell(run, group) -> Cell`, `scenario_view(s, slots) -> dict`, `run_view(run) -> dict`, `engine_slots(scenarios) -> dict[str, int]`.
  - Formatters `fmt_score`, `fmt_delta`, `fmt_ms`, `fmt_bytes`, `fmt_group`.
  - `build_site(results_dir, out, docs_url=DEFAULT_DOCS_URL) -> SiteResult(pages, unreadable)`; `MARKER = ".irbench-report"`.
  - CLI `irbench report --results DIR --out DIR [--docs-url URL]`.
  - `scenario.html.j2` contains the line `{# performance #}`, which Task 6 replaces with `{% include "perf.html.j2" %}`.

- [ ] **Step 1: Add Jinja2**

Run: `uv add "jinja2>=3.1"`
Expected: `pyproject.toml` and `uv.lock` updated.

- [ ] **Step 2: Write the failing tests**

Create `harness/tests/test_report_render.py`:

```python
import shutil

import pytest
from typer.testing import CliRunner

from irbench.cli import app
from irbench.report.data import load_results
from irbench.report.render import (
    MARKER,
    build_site,
    engine_slots,
    fmt_bytes,
    fmt_delta,
    fmt_group,
    fmt_ms,
    fmt_score,
)
from report_fixtures import write_bundle

PAGE = "ecommerce__product-search__aws-c7i-4xlarge.html"


def site(tmp_path):
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    out = tmp_path / "site"
    return results, out


def read(out, name):
    return (out / name).read_text(encoding="utf-8")


def test_formatters():
    assert fmt_score(0.48072) == "0.481" and fmt_score(None) == "—"
    assert fmt_delta(0.0123) == "+0.012" and fmt_delta(-0.1) == "−0.100" and fmt_delta(None) == "—"
    assert fmt_ms(2.5) == "2.50" and fmt_ms(27.04) == "27.0" and fmt_ms(None) == "—"
    assert fmt_bytes(727293952) == "727.3 MB" and fmt_bytes(1566838784) == "1.6 GB"
    assert fmt_bytes(512) == "512 B" and fmt_bytes(None) == "—"
    assert fmt_group("keyword/keyword") == "keyword"
    assert fmt_group("filtered/semantic") == "filtered (semantic)"


def test_site_pages_and_matrix(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch", groups={"keyword/keyword": 0.481, "semantic/semantic": 0.5})
    write_bundle(results, "qdrant", groups={"semantic/semantic": 0.55},
                 na={"keyword:keyword": "no BM25 in the v0.1 adapter"})
    result = build_site(results, out)

    names = sorted(p.relative_to(out).as_posix() for p in result.pages)
    assert names == sorted([
        "index.html", PAGE,
        "runs/ecommerce__product-search__meilisearch__bge-small-en-v1.5__default__aws-c7i-4xlarge.html",
        "runs/ecommerce__product-search__qdrant__bge-small-en-v1.5__default__aws-c7i-4xlarge.html",
    ])
    assert (out / "style.css").exists() and (out / MARKER).exists()
    assert f'href="{PAGE}"' in read(out, "index.html")

    page = read(out, PAGE)
    assert "0.481" in page
    assert 'title="no BM25 in the v0.1 adapter"' in page and ">N/A<" in page
    assert page.index(">keyword<") < page.index(">semantic<")


def test_violation_and_failure_are_explicit(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch", groups={"sorted/keyword": 0.3},
                 violations={"sorted:keyword": "12 order violations"})
    write_bundle(results, "elasticsearch", status="failed", failed_stage="index")
    build_site(results, out)
    page = read(out, PAGE)
    assert "12 order violations" in page and "⚠" in page
    assert "failed: index: RuntimeError: boom" in page


def test_perturbation_breakdown(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch", groups={"keyword/keyword": 0.5})
    build_site(results, out)
    page = read(out, PAGE)
    # fixture: original 0.55, typo 0.40 (−0.150), prefix 0.30 (−0.250), synonym 0.45 (−0.100)
    assert "0.550" in page and "−0.150" in page and "−0.250" in page and "−0.100" in page


def test_model_comparison_only_with_two_models(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch", groups={"semantic/semantic": 0.5})
    build_site(results, out)
    assert "Embedding models" not in read(out, PAGE)
    write_bundle(results, "meilisearch", model="text-embedding-3-small",
                 groups={"semantic/semantic": 0.6})
    build_site(results, out)
    page = read(out, PAGE)
    assert "Embedding models" in page and "text-embedding-3-small" in page


def test_run_page_has_config_environment_and_credits(tmp_path):
    results, out = site(tmp_path)
    d = write_bundle(results, "qdrant", track="tuned",
                     config_diff={"index.hnsw_config.m": {"default": None, "effective": 32}},
                     credits=[{"name": "Jane Doe", "url": "https://github.com/jane"}],
                     na={"hybrid:hybrid": "needs keyword retrieval"})
    build_site(results, out)
    page = read(out, f"runs/{d.name}.html")
    assert "index.hnsw_config.m" in page and "32" in page
    assert '<a href="https://github.com/jane">Jane Doe</a>' in page
    assert "qdrant@sha256:00" in page and "abc123" in page and "20261009" in page
    assert "needs keyword retrieval" in page


def test_local_profile_is_labelled_not_comparable(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch", hardware="local")
    build_site(results, out)
    assert "not comparable" in read(out, "ecommerce__product-search__local.html")


def test_text_from_results_is_escaped(tmp_path):
    results, out = site(tmp_path)
    d = write_bundle(results, "meilisearch", na={"hybrid:hybrid": "<script>alert(1)</script>"},
                     config_diff={"x": {"default": None, "effective": "<b>x</b>"}})
    build_site(results, out)
    for page in (read(out, PAGE), read(out, f"runs/{d.name}.html")):
        assert "<script>alert" not in page and "<b>x</b>" not in page
    assert "&lt;script&gt;" in read(out, PAGE)


def test_empty_results(tmp_path):
    results, out = site(tmp_path)
    build_site(results, out)
    assert "No published results yet" in read(out, "index.html")


def test_rebuild_removes_stale_pages(tmp_path):
    results, out = site(tmp_path)
    d = write_bundle(results, "meilisearch")
    build_site(results, out)
    stale = out / "runs" / f"{d.name}.html"
    assert stale.exists()
    shutil.rmtree(d)
    build_site(results, out)
    assert not stale.exists()


def test_refuses_to_overwrite_a_foreign_directory(tmp_path):
    results, out = site(tmp_path)
    out.mkdir()
    (out / "precious.txt").write_text("keep me")
    with pytest.raises(FileExistsError, match="not written by"):
        build_site(results, out)
    assert (out / "precious.txt").read_text() == "keep me"


def test_engine_slots_are_global_and_stable(tmp_path):
    results, _ = site(tmp_path)
    write_bundle(results, "qdrant", hardware="a")
    write_bundle(results, "meilisearch", hardware="b")
    write_bundle(results, "elasticsearch", hardware="b")
    scenarios, _ = load_results(results)
    assert engine_slots(scenarios) == {"elasticsearch": 1, "meilisearch": 2, "qdrant": 3}
    many = [f"e{i}" for i in range(10)]
    for e in many:
        write_bundle(results, e, hardware="c")
    slots = engine_slots(load_results(results)[0])
    assert max(slots.values()) == 8 and sorted(slots.values()).count(0) == 5


def test_cli_builds_the_site(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch")
    ok = CliRunner().invoke(app, ["report", "--results", str(results), "--out", str(out)])
    assert ok.exit_code == 0, ok.output
    assert "3 pages" in ok.output  # index, scenario page, one run page


def test_cli_exits_1_on_unreadable_bundles(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch")
    bad = write_bundle(results, "qdrant")
    (bad / "scores.json").write_text("{")
    result = CliRunner().invoke(app, ["report", "--results", str(results), "--out", str(out)])
    assert result.exit_code == 1
    assert bad.name in result.output and (out / "index.html").exists()
    assert bad.name in read(out, "index.html")
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest harness/tests/test_report_render.py -v`
Expected: `ModuleNotFoundError: No module named 'irbench.report.render'`.

- [ ] **Step 4: Implement `render.py`**

Create `harness/irbench/report/render.py`:

```python
"""Static HTML report (spec §11). One page per scenario × hardware profile (results are
only compared within a profile, §8) and one per run. Plain HTML, no JavaScript."""

import shutil
from collections import defaultdict
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import jinja2

from irbench.report.data import (
    PERTURBATIONS,
    RunResult,
    ScenarioResults,
    Unreadable,
    group_key,
    load_results,
)

MARKER = ".irbench-report"
DEFAULT_DOCS_URL = "https://github.com/meilisearch/irbench/tree/main/docs"
BASE_GROUPS = ("keyword/keyword", "semantic/semantic", "hybrid/hybrid")
VECTOR_GROUPS = ("semantic/semantic", "hybrid/hybrid")
UNPUBLISHED_PROFILES = ("local",)
MAX_SLOTS = 8  # categorical palette size; later engines fold into "other" (slot 0)


@dataclass(frozen=True)
class Cell:
    text: str
    kind: str  # score | na | violated | missing | failed
    note: str = ""


@dataclass(frozen=True)
class SiteResult:
    pages: list[Path]
    unreadable: list[Unreadable]


def fmt_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def fmt_delta(value: float | None) -> str:
    return "—" if value is None else f"{value:+.3f}".replace("-", "−")


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}" if value >= 10 else f"{value:.2f}"


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    value = float(n)
    for unit in ("B", "KB", "MB"):
        if value < 1000:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} GB"


def fmt_group(group: str) -> str:
    mode, _, base = group.partition("/")
    return mode if mode == base else f"{mode} ({base})"


def engine_slots(scenarios: list[ScenarioResults]) -> dict[str, int]:
    """Color slot per engine, the same on every page (color follows the entity)."""
    engines = sorted({r.engine for s in scenarios for r in s.runs})
    return {e: i + 1 if i < MAX_SLOTS else 0 for i, e in enumerate(engines)}


def relevance_cell(run: RunResult, group: str) -> Cell:
    reason = run.na_reason(group)
    if reason:
        return Cell("N/A", "na", reason)
    value = run.ndcg(group)
    if value is None:
        return Cell("—", "missing", "not run")
    violation = run.violation(group)
    if violation:
        return Cell(fmt_score(value), "violated",
                    f"constraint violated: {violation}; this score does not measure the mode")
    return Cell(fmt_score(value), "score")


def _delta(value: float | None, base: float) -> float | None:
    return None if value is None else value - base


def perturbation_rows(s: ScenarioResults) -> list[dict[str, Any]]:
    rows = []
    for run in s.runs:
        if run.failure:
            continue
        for group in BASE_GROUPS:
            original = run.ndcg(group, "original")
            if original is None:
                continue
            rows.append({
                "run": run,
                "group": group,
                "original": original,
                "deltas": {p: _delta(run.ndcg(group, p), original) for p in PERTURBATIONS[1:]},
            })
    return rows


def model_comparison(s: ScenarioResults) -> list[dict[str, Any]]:
    if len(s.models) < 2:
        return []
    tables = []
    for group in VECTOR_GROUPS:
        by_label: dict[str, dict[str, Cell]] = defaultdict(dict)
        for run in s.runs:
            if not run.failure:
                by_label[run.label][run.model] = relevance_cell(run, group)
        rows = [
            {"label": label, "cells": [cells.get(m, Cell("—", "missing", "not run"))
                                       for m in s.models]}
            for label, cells in sorted(by_label.items())
        ]
        tables.append({"group": group, "rows": rows})
    return tables


def na_notes(s: ScenarioResults) -> list[dict[str, Any]]:
    notes: dict[tuple[str, str], list[str]] = defaultdict(list)
    for run in s.runs:
        for group in run.groups:
            reason = run.na_reason(group)
            if reason and group not in notes[(run.label, reason)]:
                notes[(run.label, reason)].append(group)
    return [{"label": label, "reason": reason, "groups": groups}
            for (label, reason), groups in sorted(notes.items())]


def scenario_view(s: ScenarioResults, slots: dict[str, int]) -> dict[str, Any]:
    return {
        "columns": s.groups,
        "matrix": [{"run": r, "cells": [relevance_cell(r, g) for g in s.groups]} for r in s.runs],
        "na_notes": na_notes(s),
        "perturbations": perturbation_rows(s),
        "models": model_comparison(s),
        "comparable": s.hardware not in UNPUBLISHED_PROFILES,
    }


def run_view(run: RunResult) -> dict[str, Any]:
    rows = []
    for group in run.groups:
        reason = run.na_reason(group)
        all_metrics = None
        if run.scores is not None and not reason:
            mode, _, base = group.partition("/")
            all_metrics = next(
                (g for g in run.scores["relevance"]
                 if (g["mode"], g["base"], g["perturbation"]) == (mode, base, "all")),
                None,
            )
        rows.append({
            "group": group,
            "na": reason,
            "violation": run.violation(group),
            "n": all_metrics and all_metrics["n"],
            "metrics": all_metrics and all_metrics["metrics"],
            "by_perturbation": {p: run.ndcg(group, p) for p in PERTURBATIONS},
        })
    config = run.manifest.get("config", {})
    return {
        "relevance": rows,
        "checks": sorted((run.scores or {}).get("checks", []),
                         key=lambda c: group_key(f"{c['mode']}/{c['base']}")),
        "diff": config.get("diff_vs_default", {}),
        "credits": config.get("credits", []),
        "stages": run.manifest.get("stages", {}),
    }


def _env() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.PackageLoader("irbench.report", "templates"),
        autoescape=True,
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update(score=fmt_score, delta=fmt_delta, ms=fmt_ms, bytes=fmt_bytes,
                       group=fmt_group)
    return env


def _prepare_out(out: Path) -> None:
    if out.exists():
        if not (out / MARKER).exists() and any(out.iterdir()):
            raise FileExistsError(
                f"{out} is not empty and was not written by `irbench report`; choose another --out"
            )
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / MARKER).write_text("written by `irbench report`; replaced on every build\n")


def build_site(results_dir: Path, out: Path, docs_url: str = DEFAULT_DOCS_URL) -> SiteResult:
    scenarios, unreadable = load_results(results_dir)
    _prepare_out(out)
    env = _env()
    slots = engine_slots(scenarios)
    pages: list[Path] = []

    def write(rel: str, template: str, **context: Any) -> None:
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        html = env.get_template(template).render(
            docs_url=docs_url, root="../" * rel.count("/"), **context
        )
        path.write_text(html, encoding="utf-8")
        pages.append(path)

    write("index.html", "index.html.j2", scenarios=scenarios, unreadable=unreadable)
    for s in scenarios:
        write(f"{s.slug}.html", "scenario.html.j2", s=s, view=scenario_view(s, slots))
        for run in s.runs:
            write(f"runs/{run.run_id}.html", "run.html.j2", run=run, s=s, view=run_view(run))
    style = resources.files("irbench.report").joinpath("static/style.css")
    (out / "style.css").write_text(style.read_text(encoding="utf-8"), encoding="utf-8")
    return SiteResult(pages, unreadable)
```

`scenario_view` takes `slots` now, although only Task 6's performance section uses them. That way `build_site` doesn't change signature twice.

- [ ] **Step 5: Write the templates**

`harness/irbench/report/templates/base.html.j2`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{% block title %}IRBench results{% endblock %}</title>
<link rel="stylesheet" href="{{ root }}style.css">
</head>
<body>
<header class="site">
  <a class="brand" href="{{ root }}index.html">IRBench</a>
  <nav><a href="{{ docs_url }}">Methodology</a></nav>
</header>
<main>
{% block main %}{% endblock %}
</main>
<footer>
  Generated by <code>irbench report</code> from published result bundles. Relevance and
  performance are reported separately and never combined into one score.
</footer>
</body>
</html>
```

`harness/irbench/report/templates/index.html.j2`:

```html
{% extends "base.html.j2" %}
{% block main %}
<h1>IRBench results</h1>
<p class="lede">Relevance and performance of search engines on realistic, imperfect queries.
Results are compared only within one hardware profile.</p>
{% if unreadable %}
<section class="alert">
  <h2>Unreadable result bundles</h2>
  <ul>
  {% for u in unreadable %}
    <li><code>{{ u.path }}</code>: {{ u.error }}</li>
  {% endfor %}
  </ul>
</section>
{% endif %}
{% if scenarios %}
<div class="scroll"><table>
  <thead><tr><th>Scenario</th><th>Hardware profile</th><th>Runs</th><th>Engines</th><th>Models</th></tr></thead>
  <tbody>
  {% for s in scenarios %}
    <tr>
      <th scope="row"><a href="{{ s.slug }}.html">{{ s.scenario }}</a></th>
      <td><code>{{ s.hardware }}</code></td>
      <td class="num">{{ s.runs|length }}</td>
      <td>{{ s.runs|map(attribute="engine")|unique|join(", ") }}</td>
      <td>{{ s.models|join(", ") }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table></div>
{% else %}
<p class="empty">No published results yet.</p>
{% endif %}
{% endblock %}
```

`harness/irbench/report/templates/scenario.html.j2`:

```html
{% extends "base.html.j2" %}
{% block title %}{{ s.scenario }} · {{ s.hardware }} · IRBench{% endblock %}
{% block main %}
<h1>{{ s.scenario }}</h1>
<p class="meta">Hardware profile <code>{{ s.hardware }}</code> · {{ s.runs|length }} runs ·
<a href="{{ docs_url }}">how this is measured</a></p>
{% if not view.comparable %}
<p class="alert">Profile <code>{{ s.hardware }}</code> is a developer machine. These numbers check
the method; they are not comparable with published results.</p>
{% endif %}

<section>
<h2>Relevance: nDCG@10 by mode</h2>
<p class="note">N/A marks a mode the engine does not support (reasons below). ⚠ marks a score
whose results broke the mode's contract (filter or order violations); it does not measure that mode.</p>
<div class="scroll"><table class="matrix">
  <thead><tr><th>Engine</th><th>Model</th>{% for g in view.columns %}<th>{{ g|group }}</th>{% endfor %}</tr></thead>
  <tbody>
  {% for row in view.matrix %}
    <tr>
      <th scope="row"><a href="runs/{{ row.run.run_id }}.html">{{ row.run.label }}</a></th>
      <td>{{ row.run.model }}</td>
      {% if row.run.failure %}
      <td class="failed" colspan="{{ view.columns|length }}">{{ row.run.status }}: {{ row.run.failure }}</td>
      {% else %}
      {% for c in row.cells %}
      <td class="num {{ c.kind }}"{% if c.note %} title="{{ c.note }}"{% endif %}>{% if c.kind == "violated" %}⚠ {% endif %}{{ c.text }}</td>
      {% endfor %}
      {% endif %}
    </tr>
  {% endfor %}
  </tbody>
</table></div>
{% if view.na_notes %}
<h3>Not applicable</h3>
<ul class="notes">
{% for n in view.na_notes %}
  <li><strong>{{ n.label }}</strong> — {{ n.groups|map("group")|join(", ") }}: {{ n.reason }}</li>
{% endfor %}
</ul>
{% endif %}
</section>

<section>
<h2>Imperfect queries</h2>
<p class="note">nDCG@10 on the original queries, and the change when the same queries carry a
typo, are cut short (search as you type) or use a synonym. Perturbation rules are in the methodology.</p>
<div class="scroll"><table>
  <thead><tr><th>Engine</th><th>Model</th><th>Mode</th><th>original</th><th>typo</th><th>prefix</th><th>synonym</th></tr></thead>
  <tbody>
  {% for r in view.perturbations %}
    <tr>
      <th scope="row">{{ r.run.label }}</th><td>{{ r.run.model }}</td><td>{{ r.group|group }}</td>
      <td class="num">{{ r.original|score }}</td>
      {% for p in ["typo", "prefix", "synonym"] %}<td class="num delta">{{ r.deltas[p]|delta }}</td>{% endfor %}
    </tr>
  {% endfor %}
  </tbody>
</table></div>
</section>

{% if view.models %}
<section>
<h2>Embedding models</h2>
{% for t in view.models %}
<h3>{{ t.group|group }}: nDCG@10</h3>
<div class="scroll"><table>
  <thead><tr><th>Engine</th>{% for m in s.models %}<th>{{ m }}</th>{% endfor %}</tr></thead>
  <tbody>
  {% for row in t.rows %}
    <tr><th scope="row">{{ row.label }}</th>{% for c in row.cells %}<td class="num {{ c.kind }}"{% if c.note %} title="{{ c.note }}"{% endif %}>{{ c.text }}</td>{% endfor %}</tr>
  {% endfor %}
  </tbody>
</table></div>
{% endfor %}
</section>
{% endif %}

{# performance #}

<section>
<h2>Runs</h2>
<ul class="runs">
{% for r in s.runs %}
  <li><a href="runs/{{ r.run_id }}.html">{{ r.label }} · {{ r.model }}</a> — {{ r.status }}</li>
{% endfor %}
</ul>
</section>
{% endblock %}
```

`harness/irbench/report/templates/run.html.j2`:

```html
{% extends "base.html.j2" %}
{% block title %}{{ run.label }} · {{ run.model }} · IRBench{% endblock %}
{% block main %}
<p class="crumbs"><a href="../{{ s.slug }}.html">{{ s.scenario }} · {{ s.hardware }}</a></p>
<h1>{{ run.label }} · {{ run.model }}</h1>
<p class="meta">Status <strong>{{ run.status }}</strong> · track <code>{{ run.track }}</code> · profile <code>{{ run.hardware }}</code></p>
{% if run.failure %}<p class="alert">{{ run.failure }}</p>{% endif %}

{% if view.credits %}
<section><h2>Credits</h2><p>Tuned configuration contributed by
{% for c in view.credits %}{% if c.url %}<a href="{{ c.url }}">{{ c.name }}</a>{% else %}{{ c.name }}{% endif %}{% if not loop.last %}, {% endif %}{% endfor %}.</p></section>
{% endif %}

<section>
<h2>Configuration</h2>
<dl class="facts">
  <dt>Config file</dt><dd><code>{{ run.manifest.config.path|default("—") }}</code></dd>
  <dt>Config hash</dt><dd><code>{{ run.manifest.config.hash|default("—") }}</code></dd>
</dl>
{% if view.diff %}
<h3>Changes from the default track</h3>
<div class="scroll"><table>
  <thead><tr><th>Setting</th><th>Default</th><th>This run</th></tr></thead>
  <tbody>
  {% for key, change in view.diff|dictsort %}
    <tr><th scope="row"><code>{{ key }}</code></th><td><code>{{ change.default|tojson }}</code></td><td><code>{{ change.effective|tojson }}</code></td></tr>
  {% endfor %}
  </tbody>
</table></div>
{% else %}
<p class="note">Identical to the default track.</p>
{% endif %}
</section>

<section>
<h2>Environment</h2>
<dl class="facts">
  <dt>Engine image</dt><dd><code>{{ run.manifest.engine_image.image|default("—") if run.manifest.engine_image is defined else "—" }}</code></dd>
  <dt>Image digest</dt><dd><code>{{ run.manifest.engine_image.digest|default("—") if run.manifest.engine_image is defined else "—" }}</code></dd>
  <dt>Harness</dt><dd><code>{{ run.manifest.irbench_version|default("—") }}</code> at <code>{{ run.manifest.git_sha|default("—") }}</code></dd>
  <dt>Dataset</dt><dd><code>{{ run.manifest.dataset.id|default("—") if run.manifest.dataset is defined else "—" }}@{{ run.manifest.dataset.version|default("—") if run.manifest.dataset is defined else "—" }}</code></dd>
  <dt>Perturbation seed</dt><dd><code>{{ run.manifest.seed|default("—") }}</code></dd>
</dl>
<h3>Stages</h3>
<div class="scroll"><table>
  <thead><tr><th>Stage</th><th>Status</th><th>Seconds</th><th>Error</th></tr></thead>
  <tbody>
  {% for name, info in view.stages.items() %}
    <tr><th scope="row">{{ name }}</th><td>{{ info.status }}</td><td class="num">{{ info.seconds|default("—") }}</td><td>{{ info.error|default("") }}</td></tr>
  {% endfor %}
  </tbody>
</table></div>
</section>

{% if view.relevance %}
<section>
<h2>Relevance</h2>
<div class="scroll"><table>
  <thead><tr><th>Mode</th><th>n</th><th>nDCG@10</th><th>Recall@10</th><th>Recall@100</th><th>MRR@10</th><th>Success@1</th><th>original</th><th>typo</th><th>prefix</th><th>synonym</th></tr></thead>
  <tbody>
  {% for r in view.relevance %}
    <tr>
      <th scope="row">{{ r.group|group }}</th>
      {% if r.na %}
      <td class="na" colspan="10">N/A: {{ r.na }}</td>
      {% else %}
      <td class="num">{{ r.n }}</td>
      {% for m in ["ndcg@10", "recall@10", "recall@100", "mrr@10", "success@1"] %}<td class="num">{{ r.metrics[m]|score }}</td>{% endfor %}
      {% for p in ["original", "typo", "prefix", "synonym"] %}<td class="num">{{ r.by_perturbation[p]|score }}</td>{% endfor %}
      {% endif %}
    </tr>
    {% if r.violation %}<tr><td></td><td class="violated" colspan="10">⚠ constraint violated: {{ r.violation }}</td></tr>{% endif %}
  {% endfor %}
  </tbody>
</table></div>
</section>
{% endif %}

{% if view.checks %}
<section>
<h2>Correctness checks</h2>
<div class="scroll"><table>
  <thead><tr><th>Mode</th><th>Tasks</th><th>Filter violations</th><th>Order violations</th><th>Facet accuracy</th><th>Facets verified</th><th>Facets unverifiable</th></tr></thead>
  <tbody>
  {% for c in view.checks %}
    <tr>
      <th scope="row">{{ (c.mode ~ "/" ~ c.base)|group }}</th><td class="num">{{ c.n }}</td>
      <td class="num">{{ c.filter_violations if c.filter_violations is not none else "—" }}</td>
      <td class="num">{{ c.order_violations if c.order_violations is not none else "—" }}</td>
      <td class="num">{{ c.facet_accuracy|score }}</td>
      <td class="num">{{ c.facet_verified }}</td><td class="num">{{ c.facet_unverifiable }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table></div>
</section>
{% endif %}

{# run performance #}
{% endblock %}
```

With `StrictUndefined`, `run.manifest.config.path|default(...)` works because `default` handles undefined attributes on a dict. Missing nested dicts (`engine_image`, `dataset`) are guarded with `is defined`. If a template raises `UndefinedError` on a fixture, add the guard for that key in the same `is defined` style rather than dropping `StrictUndefined`.

- [ ] **Step 6: Write the stylesheet**

`harness/irbench/report/static/style.css`. The series slots are the validated reference categorical palette (light and dark steps).

```css
:root {
  color-scheme: light;
  --surface: #fcfcfb;
  --surface-2: #f2f1ee;
  --border: #dddcd7;
  --text: #0b0b0b;
  --text-2: #52514e;
  --link: #1c5cab;
  --alert-bg: #fdf3e1;
  --alert-border: #e5b45c;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
  --series-5: #e87ba4; --series-6: #008300; --series-7: #6250d6; --series-8: #e34948;
  --series-other: #8a8984;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --surface-2: #242423; --border: #3a3a38;
    --text: #f0efec; --text-2: #c3c2b7; --link: #86b6ef;
    --alert-bg: #33290f; --alert-border: #c98500;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
    --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
    --series-other: #8f8e88;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --surface-2: #242423; --border: #3a3a38;
  --text: #f0efec; --text-2: #c3c2b7; --link: #86b6ef;
  --alert-bg: #33290f; --alert-border: #c98500;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
  --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
  --series-other: #8f8e88;
}

* { box-sizing: border-box; }
body {
  margin: 0; background: var(--surface); color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
a { color: var(--link); }
code { font: 0.9em ui-monospace, SFMono-Regular, Menlo, monospace; }
header.site, main, footer { max-width: 1120px; margin: 0 auto; padding: 0 16px; }
header.site { display: flex; justify-content: space-between; align-items: center; padding-top: 16px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }
.brand { font-weight: 700; color: var(--text); text-decoration: none; }
footer { color: var(--text-2); font-size: 13px; padding-top: 32px; padding-bottom: 32px; }
h1 { font-size: 28px; margin: 24px 0 4px; }
h2 { font-size: 20px; margin: 32px 0 8px; }
h3 { font-size: 16px; margin: 20px 0 8px; }
.lede, .meta, .note, .crumbs { color: var(--text-2); }
.note { font-size: 13px; max-width: 72ch; }
.alert { background: var(--alert-bg); border-left: 3px solid var(--alert-border); padding: 8px 12px; }
.empty { color: var(--text-2); font-style: italic; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; margin: 8px 0; font-size: 14px; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--border); text-align: left; white-space: nowrap; }
thead th { color: var(--text-2); font-weight: 600; border-bottom-color: var(--text-2); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.na, td.missing { color: var(--text-2); }
td.na[title] { text-decoration: underline dotted; cursor: help; }
td.violated { color: var(--text); background: var(--alert-bg); }
td.failed { color: var(--text); background: var(--alert-bg); white-space: normal; }
dl.facts { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; }
dl.facts dt { color: var(--text-2); }
dl.facts dd { margin: 0; overflow-wrap: anywhere; }
ul.notes li, ul.runs li { margin: 4px 0; }

/* Charts (Task 6): 2px lines, ≥8px markers with a 2px surface ring, hairline grid. */
.chart { width: 100%; max-width: 720px; height: auto; display: block; }
.chart .grid { stroke: var(--border); stroke-width: 1; }
.chart .tick, .chart .axis-label { fill: var(--text-2); font-size: 12px; }
.chart .direct { fill: var(--text); font-size: 12px; }
.chart .series { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.chart .series.dashed { stroke-dasharray: 6 4; }
.chart .marker { fill: var(--c); stroke: var(--surface); stroke-width: 2; }
.chart .marker.invalid { fill: var(--surface); stroke: var(--c); }
ul.legend { list-style: none; padding: 0; display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; }
ul.legend .key { display: inline-block; width: 16px; height: 2px; background: var(--c); vertical-align: middle; margin-right: 6px; }
ul.legend .key.dashed { background: repeating-linear-gradient(90deg, var(--c) 0 6px, transparent 6px 10px); }
details { margin: 8px 0 24px; }
```

- [ ] **Step 7: Add the CLI command**

In `harness/irbench/cli.py`, add `from irbench.report.render import DEFAULT_DOCS_URL, build_site` and:

```python
@app.command()
def report(
    results: Path = typer.Option(..., "--results", help="results/ or .irbench/runs for a preview."),
    out: Path = typer.Option(..., "--out", help="Output directory (replaced on every build)."),
    docs_url: str = typer.Option(DEFAULT_DOCS_URL, "--docs-url"),
) -> None:
    """Build the static HTML report from run bundles."""
    try:
        site = build_site(results, out, docs_url)
    except (FileNotFoundError, FileExistsError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e
    for u in site.unreadable:
        typer.echo(f"unreadable bundle {u.path}: {u.error}", err=True)
    typer.echo(f"report: {len(site.pages)} pages in {out}")
    if site.unreadable:
        raise typer.Exit(code=1)
```

- [ ] **Step 8: Run the tests**

Run: `uv run pytest harness/tests/test_report_render.py -v`
Expected: all pass. A failure in `test_text_from_results_is_escaped` means autoescape is off for `.j2` files. `autoescape=True` (not `select_autoescape`) covers every template; check it.

- [ ] **Step 9: Lint, full suite, commit**

```bash
uv run ruff check --fix . && uv run ruff format harness engines && uv run pytest
git add pyproject.toml uv.lock harness/irbench/report harness/irbench/cli.py \
  harness/tests/test_report_render.py
git commit -m "feat: static HTML report with relevance matrix, perturbations and run pages"
```

---

### Task 6: Performance section: latency, throughput charts and indexing cost

Spec §11 "Performance: latency percentiles, QPS curve, indexing time / RAM / disk". Spec §5.7 defines a valid step.

**Files:**
- Create: `harness/irbench/report/charts.py`
- Create: `harness/irbench/report/templates/perf.html.j2`
- Modify: `harness/irbench/report/render.py` (`perf_view`, wire into `scenario_view` and `run_view`)
- Modify: `harness/irbench/report/templates/scenario.html.j2` (`{# performance #}` → include), `run.html.j2` (`{# run performance #}`)
- Test: `harness/tests/test_report_charts.py`, `harness/tests/test_report_render.py`

**Interfaces:**
- Consumes: `RunResult.perf` (`limits.p99_ms`, `latency[group]` with `p50_ms`/`p99_ms`/`errors`/`count`, `throughput[group]` with `steps[]` of `concurrency`, `qps`, `p99_ms`, `error_rate`, `valid`, plus `max_qps`/`max_qps_concurrency`, and `skipped_groups`); `RunResult.index`; `engine_slots`; `Cell`; `fmt_ms`; `group_key`.
- Produces:
  - `Point(concurrency, qps, valid, note)`, `Series(label, slot, dashed, points)`.
  - `nice_max(value) -> float`, `qps_chart(title, series, steps) -> Markup | None`.
  - `perf_view(s, slots) -> dict` with keys `runs`, `latency`, `max_qps`, `charts`, `indexing`, `unmeasured`.

- [ ] **Step 1: Write the failing chart tests**

Create `harness/tests/test_report_charts.py`:

```python
import pytest

from irbench.report.charts import Point, Series, nice_max, qps_chart

STEPS = [1, 2, 4, 8]


def series(label, slot, qps, invalid_last=False, dashed=False):
    pts = [Point(c, q, not (invalid_last and i == len(qps) - 1), f"{c}: {q}")
           for i, (c, q) in enumerate(zip(STEPS, qps))]
    return Series(label, slot, dashed, pts)


@pytest.mark.parametrize("value,expected", [
    (0, 1), (0.4, 0.5), (85.6, 100), (170.7, 200), (2426, 2500), (2950, 5000), (5000, 5000),
])
def test_nice_max(value, expected):
    assert nice_max(value) == pytest.approx(expected)


def test_two_series():
    svg = str(qps_chart("keyword", [series("meilisearch", 2, [85, 170, 300, 310], True),
                                    series("qdrant", 3, [100, 200, 390, 700])], STEPS))
    assert svg.startswith("<svg") and 'role="img"' in svg and 'aria-label="keyword"' in svg
    assert svg.count("<polyline") == 2 and svg.count("<circle") == 8
    assert svg.count('class="marker invalid"') == 1
    assert "var(--series-2)" in svg and "var(--series-3)" in svg
    assert svg.count('class="direct"') == 2  # ≤ 4 series: direct labels
    for tick in ("1", "2", "4", "8"):
        assert f">{tick}</text>" in svg
    assert ">1,000</text>" in svg  # y axis tops at nice_max(700) = 1000


def test_many_series_drop_direct_labels():
    svg = str(qps_chart("x", [series(f"e{i}", i, [10, 20]) for i in range(1, 6)], STEPS))
    assert 'class="direct"' not in svg


def test_single_point_has_a_marker_and_no_line():
    svg = str(qps_chart("x", [series("solo", 1, [42.0])], STEPS))
    assert "<polyline" not in svg and svg.count("<circle") == 1


def test_folded_and_dashed_series():
    svg = str(qps_chart("x", [series("e9", 0, [1, 2], dashed=True)], STEPS))
    assert "var(--series-other)" in svg and "dashed" in svg


def test_labels_are_escaped():
    svg = str(qps_chart("<t>", [Series("<b>x</b>", 1, False, [Point(1, 1.0, True, "<i>")])], STEPS))
    assert "<b>" not in svg and "<i>" not in svg and "&lt;b&gt;" in svg


def test_nothing_to_plot():
    assert qps_chart("x", [], STEPS) is None
    assert qps_chart("x", [Series("a", 1, False, [])], STEPS) is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest harness/tests/test_report_charts.py -v`
Expected: `ModuleNotFoundError: No module named 'irbench.report.charts'`.

- [ ] **Step 3: Implement the chart**

Create `harness/irbench/report/charts.py`:

```python
"""Inline SVG line chart: QPS by concurrency step, one line per run. No JavaScript: hover
uses native <title> tooltips, and the step table under each chart carries the numbers."""

import math
from dataclasses import dataclass

from markupsafe import Markup, escape

WIDTH, HEIGHT = 640, 300
LEFT, RIGHT, TOP, BOTTOM = 64, 120, 16, 44


@dataclass(frozen=True)
class Point:
    concurrency: int
    qps: float
    valid: bool  # p99 and error rate within the scenario's limits (spec §5.7)
    note: str  # tooltip text


@dataclass(frozen=True)
class Series:
    label: str
    slot: int  # categorical slot 1..8, fixed per engine; 0 = folded "other"
    dashed: bool  # tuned track
    points: list[Point]


def nice_max(value: float) -> float:
    """Smallest 1, 2, 2.5 or 5 × 10^k at or above value."""
    if value <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(value))
    for m in (1, 2, 2.5, 5, 10):
        if value <= m * exp * (1 + 1e-9):
            return m * exp
    return 10 * exp


def qps_chart(title: str, series: list[Series], steps: list[int]) -> Markup | None:
    plotted = [(s, [p for p in s.points if p.concurrency in steps]) for s in series]
    plotted = [(s, pts) for s, pts in plotted if pts]
    if not plotted or not steps:
        return None
    ymax = nice_max(max(p.qps for _, pts in plotted for p in pts))
    pw, ph = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

    def x(c: int) -> float:
        return LEFT + steps.index(c) / max(len(steps) - 1, 1) * pw

    def y(v: float) -> float:
        return TOP + ph - v / ymax * ph

    out = [f'<svg class="chart" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
           f'aria-label="{escape(title)}">']
    for i in range(5):
        v = ymax * i / 4
        out.append(f'<line class="grid" x1="{LEFT}" x2="{LEFT + pw}" y1="{y(v):.1f}" '
                   f'y2="{y(v):.1f}"/>')
        out.append(f'<text class="tick" x="{LEFT - 8}" y="{y(v) + 4:.1f}" '
                   f'text-anchor="end">{v:,.0f}</text>')
    for c in steps:
        out.append(f'<text class="tick" x="{x(c):.1f}" y="{TOP + ph + 18}" '
                   f'text-anchor="middle">{c}</text>')
    mid = TOP + ph / 2
    out.append(f'<text class="axis-label" x="{LEFT + pw / 2:.1f}" y="{HEIGHT - 6}" '
               'text-anchor="middle">concurrent clients</text>')
    out.append(f'<text class="axis-label" x="14" y="{mid:.1f}" transform="rotate(-90 14 '
               f'{mid:.1f})" text-anchor="middle">queries per second</text>')

    direct = len(plotted) <= 4
    for s, pts in plotted:
        color = f"var(--series-{s.slot})" if s.slot else "var(--series-other)"
        if len(pts) > 1:
            coords = " ".join(f"{x(p.concurrency):.1f},{y(p.qps):.1f}" for p in pts)
            dashed = " dashed" if s.dashed else ""
            out.append(f'<polyline class="series{dashed}" points="{coords}" '
                       f'style="stroke:{color}"/>')
        for p in pts:
            cls = "marker" if p.valid else "marker invalid"
            out.append(f'<circle class="{cls}" cx="{x(p.concurrency):.1f}" cy="{y(p.qps):.1f}" '
                       f'r="4" style="--c:{color}"><title>{escape(s.label)}: '
                       f'{escape(p.note)}</title></circle>')
        if direct:
            last = pts[-1]
            out.append(f'<text class="direct" x="{x(last.concurrency) + 10:.1f}" '
                       f'y="{y(last.qps) + 4:.1f}">{escape(s.label)}</text>')
    out.append("</svg>")
    return Markup("".join(out))
```

- [ ] **Step 4: Run the chart tests**

Run: `uv run pytest harness/tests/test_report_charts.py -v`
Expected: all pass.

- [ ] **Step 5: Write the failing page tests**

Add `from report_fixtures import perf as perf_report` to the imports at the top of `harness/tests/test_report_render.py` (next to `write_bundle`), then append:

```python
def test_performance_section(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch", perf_report=perf_report(
        {"keyword/keyword": (85.0, 4), "semantic/semantic": (30.0, 2)},
        latency_groups=["keyword/keyword", "semantic/semantic", "filtered/keyword"]))
    write_bundle(results, "qdrant", groups={"semantic/semantic": 0.5},
                 na={"keyword:keyword": "no BM25"},
                 perf_report=perf_report({"semantic/semantic": (100.0, 8)},
                                         skipped=["keyword/keyword"]))
    write_bundle(results, "elasticsearch", perf_report=None)
    build_site(results, out)
    page = read(out, PAGE)

    assert page.count('<svg class="chart"') == 2          # keyword, semantic for one model
    assert "340" in page                                   # meilisearch keyword max QPS (85 × 4)
    assert "2.50 / 12.2" in page                           # p50 / p99 latency cell
    assert "no BM25" in page                               # skipped throughput is N/A with reason
    assert "elasticsearch" in page and "Performance was not measured" in page
    assert "727.3 MB" in page and "1.6 GB" in page and "42.5" in page  # indexing cost


def test_no_valid_step_is_explicit(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch", perf_report=perf_report({"keyword/keyword": (50.0, 0)}))
    build_site(results, out)
    page = read(out, PAGE)
    assert "none" in page and "no concurrency step met" in page


def test_engine_color_follows_the_engine_across_pages(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch", hardware="a", perf_report=perf_report({"keyword/keyword": (1.0, 1)}))
    write_bundle(results, "qdrant", hardware="b", perf_report=perf_report({"keyword/keyword": (1.0, 1)}))
    build_site(results, out)
    assert "var(--series-2)" in read(out, "ecommerce__product-search__b.html")  # qdrant is slot 2


def test_run_page_shows_its_perf(tmp_path):
    results, out = site(tmp_path)
    d = write_bundle(results, "meilisearch", perf_report=perf_report({"keyword/keyword": (85.0, 2)}))
    build_site(results, out)
    page = read(out, f"runs/{d.name}.html")
    assert "Performance" in page and "170" in page
```

- [ ] **Step 6: Run them to verify they fail**

Run: `uv run pytest harness/tests/test_report_render.py -k "performance or valid_step or color_follows or its_perf" -v`
Expected: FAIL (no chart, no performance section).

- [ ] **Step 7: Implement `perf_view` and wire it in**

In `harness/irbench/report/render.py`, add `from irbench.report.charts import Point, Series, qps_chart` and:

```python
NO_VALID_STEP = "no concurrency step met p99 ≤ {p99:g} ms and ≤ {err:.1%} errors"


def _latency_cell(run: RunResult, group: str) -> Cell:
    lat = run.perf["latency"].get(group)
    if lat is None:
        return Cell("N/A", "na", run.na_reason(group) or "not measured")
    note = f"{lat['count']} requests, {lat['errors']} errors" if lat.get("count") else ""
    return Cell(f"{fmt_ms(lat['p50_ms'])} / {fmt_ms(lat['p99_ms'])}", "score", note)


def _qps_cell(run: RunResult, group: str) -> Cell:
    if group in run.perf["skipped_groups"]:
        return Cell("N/A", "na", run.na_reason(group) or "the engine does not support this mode")
    t = run.perf["throughput"].get(group)
    if t is None:
        return Cell("—", "missing", "not measured")
    if t["max_qps"] is None:
        limits = run.perf["limits"]
        return Cell("none", "violated",
                    NO_VALID_STEP.format(p99=limits["p99_ms"], err=limits["max_error_rate"]))
    return Cell(f"{t['max_qps']:,.0f}", "score", f"at {t['max_qps_concurrency']} concurrent clients")


def _series(run: RunResult, group: str, slots: dict[str, int]) -> Series:
    points = [
        Point(
            st["concurrency"], st["qps"], st["valid"],
            f"{st['concurrency']} clients: {st['qps']:,.0f} QPS, p99 {fmt_ms(st['p99_ms'])} ms, "
            f"errors {st['error_rate']:.2%}" + ("" if st["valid"] else " (invalid step)"),
        )
        for st in run.perf["throughput"][group]["steps"]
    ]
    return Series(run.label, slots.get(run.engine, 0), run.track != "default", points)


def perf_view(s: ScenarioResults, slots: dict[str, int]) -> dict[str, Any]:
    measured = [r for r in s.runs if r.perf is not None and not r.failure]
    latency_groups = sorted(
        {g for r in measured for g in r.perf["latency"] if g != "all"}, key=group_key
    )
    throughput_groups = sorted(
        {g for r in measured for g in (*r.perf["throughput"], *r.perf["skipped_groups"])},
        key=group_key,
    )
    charts = []
    for model in s.models:
        runs = [r for r in measured if r.model == model]
        for group in throughput_groups:
            with_data = [r for r in runs if group in r.perf["throughput"]]
            series = [_series(r, group, slots) for r in with_data]
            steps = sorted({p.concurrency for se in series for p in se.points})
            svg = qps_chart(f"{fmt_group(group)}: queries per second by concurrency, {model}",
                            series, steps)
            if svg is None:
                continue
            table = [
                {"concurrency": c, "cells": [
                    next((f"{p.qps:,.0f}" + ("" if p.valid else " (invalid)")
                          for p in se.points if p.concurrency == c), "—")
                    for se in series]}
                for c in steps
            ]
            charts.append({"group": group, "model": model, "svg": svg, "series": series,
                           "table": table})
    return {
        "runs": measured,
        "latency": [{"group": g, "cells": [_latency_cell(r, g) for r in measured]}
                    for g in latency_groups],
        "max_qps": [{"group": g, "cells": [_qps_cell(r, g) for r in measured]}
                    for g in throughput_groups],
        "charts": charts,
        "indexing": [r for r in s.runs if r.index and not r.failure],
        "unmeasured": [r.label for r in s.runs if r.perf is None and not r.failure],
    }
```

In `scenario_view`, add `"perf": perf_view(s, slots),`. In `run_view`, add the run's own perf rows:

```python
        "perf": [
            {"group": g, "latency": _latency_cell(run, g), "max_qps": _qps_cell(run, g)}
            # "all" mixes modes and has no group_key; it stays out, as in summarize_perf.
            for g in sorted((k for k in run.perf["latency"] if k != "all"), key=group_key)
        ]
        if run.perf and not run.failure
        else [],
```

- [ ] **Step 8: Write the performance template and include it**

Create `harness/irbench/report/templates/perf.html.j2`:

```html
<section>
<h2>Performance</h2>
<p class="note">Measured by replaying the exact requests scored for relevance (warm-up pass
discarded). Latency is at one client. Throughput steps up the number of concurrent clients;
max QPS is the best step with p99 ≤ 100 ms and ≤ 0.1% errors. Indexing is wall-clock time
until the engine reports every document indexed.</p>
{% if view.perf.unmeasured %}
<p class="note">Performance was not measured for: {{ view.perf.unmeasured|join(", ") }}.</p>
{% endif %}
{% if view.perf.runs %}
<h3>Latency at one client: p50 / p99 (ms)</h3>
<div class="scroll"><table>
  <thead><tr><th>Mode</th>{% for r in view.perf.runs %}<th>{{ r.label }} · {{ r.model }}</th>{% endfor %}</tr></thead>
  <tbody>
  {% for row in view.perf.latency %}
    <tr><th scope="row">{{ row.group|group }}</th>{% for c in row.cells %}<td class="num {{ c.kind }}"{% if c.note %} title="{{ c.note }}"{% endif %}>{{ c.text }}</td>{% endfor %}</tr>
  {% endfor %}
  </tbody>
</table></div>

<h3>Throughput: max QPS</h3>
<div class="scroll"><table>
  <thead><tr><th>Mode</th>{% for r in view.perf.runs %}<th>{{ r.label }} · {{ r.model }}</th>{% endfor %}</tr></thead>
  <tbody>
  {% for row in view.perf.max_qps %}
    <tr><th scope="row">{{ row.group|group }}</th>{% for c in row.cells %}<td class="num {{ c.kind }}"{% if c.note %} title="{{ c.note }}"{% endif %}>{{ c.text }}</td>{% endfor %}</tr>
  {% endfor %}
  </tbody>
</table></div>

{% for chart in view.perf.charts %}
<h3>{{ chart.group|group }} · {{ chart.model }}: QPS by concurrent clients</h3>
<ul class="legend">
{% for se in chart.series %}
  <li><span class="key{% if se.dashed %} dashed{% endif %}" style="--c: var(--series-{{ se.slot if se.slot else 'other' }})"></span>{{ se.label }}</li>
{% endfor %}
</ul>
{{ chart.svg }}
<p class="note">Hollow markers are invalid steps (p99 or error rate over the limit); stepping stops there.</p>
<details><summary>Step values</summary>
<table>
  <thead><tr><th>Clients</th>{% for se in chart.series %}<th>{{ se.label }}</th>{% endfor %}</tr></thead>
  <tbody>
  {% for row in chart.table %}<tr><th scope="row">{{ row.concurrency }}</th>{% for c in row.cells %}<td class="num">{{ c }}</td>{% endfor %}</tr>{% endfor %}
  </tbody>
</table>
</details>
{% endfor %}
{% endif %}

{% if view.perf.indexing %}
<h3>Indexing cost</h3>
<div class="scroll"><table>
  <thead><tr><th>Engine</th><th>Model</th><th>Documents</th><th>Time (s)</th><th>Peak memory</th><th>Disk</th></tr></thead>
  <tbody>
  {% for r in view.perf.indexing %}
    <tr><th scope="row">{{ r.label }}</th><td>{{ r.model }}</td><td class="num">{{ "{:,}".format(r.index.n_docs) }}</td>
      <td class="num">{{ "%.1f"|format(r.index.seconds) }}</td><td class="num">{{ r.index.memory_peak_bytes|bytes }}</td><td class="num">{{ r.index.disk_bytes|bytes }}</td></tr>
  {% endfor %}
  </tbody>
</table></div>
{% endif %}
</section>
```

In `scenario.html.j2`, replace the line `{# performance #}` with `{% include "perf.html.j2" %}`. In `run.html.j2`, replace `{# run performance #}` with:

```html
{% if view.perf %}
<section>
<h2>Performance</h2>
<div class="scroll"><table>
  <thead><tr><th>Mode</th><th>Latency p50 / p99 (ms)</th><th>Max QPS</th></tr></thead>
  <tbody>
  {% for row in view.perf %}
    <tr><th scope="row">{{ row.group|group }}</th>
      <td class="num {{ row.latency.kind }}">{{ row.latency.text }}</td>
      <td class="num {{ row.max_qps.kind }}"{% if row.max_qps.note %} title="{{ row.max_qps.note }}"{% endif %}>{{ row.max_qps.text }}</td></tr>
  {% endfor %}
  </tbody>
</table></div>
</section>
{% endif %}
```

`memory_peak_bytes` or `disk_bytes` can be `null` in `index.json` when the runtime could not read them; `fmt_bytes(None)` prints `—`. `n_docs` and `seconds` are always set by the index stage.

- [ ] **Step 9: Run the tests**

Run: `uv run pytest harness/tests/test_report_render.py harness/tests/test_report_charts.py -v` then `uv run pytest`
Expected: all pass. If `"340"` fails in `test_performance_section`, print the max-QPS row: the fixture's `steps(85.0, 4)` gives 85 × 4 = 340 at c=4.

- [ ] **Step 10: Lint and commit**

```bash
uv run ruff check --fix . && uv run ruff format harness engines
git add harness/irbench/report harness/tests/test_report_charts.py harness/tests/test_report_render.py
git commit -m "feat: report performance section with latency, QPS charts and indexing cost"
```

---

### Task 7: Published vectors

Spec §2 and §5.2: vectors are "cached and published so reproduction needs no API keys"; "`embed` downloads published vectors when available instead of recomputing". Assets live on a GitHub release of `meilisearch/irbench`; `embeddings/published.yaml` (spec §4 `embeddings/`) records URL, checksum, counts and the dataset version and seed they were built from.

**Files:**
- Create: `harness/irbench/embed/publish.py`
- Create: `embeddings/published.yaml`
- Modify: `harness/irbench/stages/embed.py`, `harness/irbench/stages/run.py` (embed stage result), `harness/irbench/cli.py`, `.gitignore` (`dist/`)
- Test: `harness/tests/test_vector_publish.py`

**Interfaces:**
- Consumes: `Workspace.doc_vectors_path`, `.query_vectors_path`, `.scenario_dir`, `.dataset_dir`; `read_vectors`; `fetch_file(url, dest, sha256, client)`, `sha256_file`, `ChecksumError` (`irbench.datasets.fetch`); `prepare.manifest.json` (`dataset.id`, `dataset.version`, `seed`).
- Produces:
  - `REGISTRY`, `RELEASE_REPO = "meilisearch/irbench"`, `DEFAULT_RELEASE = "vectors"`.
  - `VectorAsset`, `PublishedVectors`, `load_registry`, `find_published`, `asset_name`.
  - `pack_vectors(root, scenario_id, model_id, out_dir, release) -> tuple[PublishedVectors, list[Path]]` and `download_published(root, entry, client=None) -> tuple[Path, Path]`.
  - `embed_scenario(..., *, compute=False, client=None)`; `EmbedResult.source` (`"published"` or `"computed"`).
  - CLI `irbench embed --compute` and `irbench vectors pack`.

- [ ] **Step 1: Write the failing tests**

Create `harness/tests/test_vector_publish.py`:

```python
import httpx
import pytest
import yaml
from typer.testing import CliRunner

from fake_engine import SCENARIO, fake_root
from irbench.cli import app
from irbench.datasets.fetch import ChecksumError, sha256_file
from irbench.embed.publish import (
    REGISTRY,
    _save_registry,
    find_published,
    load_registry,
    pack_vectors,
)
from irbench.embed.vectors import read_vectors, write_vectors
from irbench.paths import Workspace
from irbench.scenario import load_scenario
from irbench.stages.embed import embed_scenario
from test_embed_cache import FakeEmbedder


def serve(directory, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(str(request.url))
        path = directory / request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, content=path.read_bytes()) if path.exists() else httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def vector_paths(root):
    ws, scenario = Workspace(root), load_scenario(root, SCENARIO)
    return ws.doc_vectors_path(scenario.dataset, "fake"), ws.query_vectors_path(SCENARIO, "fake")


def test_pack_writes_assets_and_registry(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    entry, files = pack_vectors(root, SCENARIO, "fake", tmp_path / "dist")
    docs, queries = vector_paths(root)
    assert [f.name for f in files] == [
        "ecommerce__product-search__fake__docs.parquet",
        "ecommerce__product-search__fake__queries.parquet",
    ]
    assert entry.docs.url == (
        "https://github.com/meilisearch/irbench/releases/download/vectors/"
        "ecommerce__product-search__fake__docs.parquet"
    )
    assert entry.docs.sha256 == sha256_file(docs) and entry.queries.sha256 == sha256_file(queries)
    assert entry.docs.count == len(read_vectors(docs)[0]) and entry.docs.dim == 4
    assert entry.dataset == "wands" and entry.seed == 20261009
    assert load_registry(root) == [entry]

    pack_vectors(root, SCENARIO, "fake", tmp_path / "dist", release="vectors-2")
    registry = load_registry(root)
    assert len(registry) == 1 and "/download/vectors-2/" in registry[0].docs.url


def test_published_vectors_are_downloaded_instead_of_computed(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    pack_vectors(root, SCENARIO, "fake", tmp_path / "dist")
    docs, queries = vector_paths(root)
    before = sha256_file(docs), sha256_file(queries)
    docs.unlink()
    queries.unlink()
    seen = []

    # "fake" is not in the model registry: computing would raise, so success means download.
    result = embed_scenario(root, SCENARIO, "fake", client=serve(tmp_path / "dist", seen))

    assert result.source == "published" and result.n_computed == 0
    assert (sha256_file(docs), sha256_file(queries)) == before
    assert len(seen) == 2


def test_local_files_matching_the_checksum_are_not_downloaded_again(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    pack_vectors(root, SCENARIO, "fake", tmp_path / "dist")
    seen = []
    embed_scenario(root, SCENARIO, "fake", client=serve(tmp_path / "dist", seen))
    assert seen == []


def test_tampered_asset_is_refused(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    _, files = pack_vectors(root, SCENARIO, "fake", tmp_path / "dist")
    files[0].write_bytes(b"not parquet")
    docs, _ = vector_paths(root)
    docs.unlink()
    with pytest.raises(ChecksumError):
        embed_scenario(root, SCENARIO, "fake", client=serve(tmp_path / "dist"))
    assert not docs.exists()


def test_seed_mismatch_is_refused(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    entry, _ = pack_vectors(root, SCENARIO, "fake", tmp_path / "dist")
    _save_registry(root, [entry.model_copy(update={"seed": 1})])
    with pytest.raises(ValueError, match="--compute"):
        embed_scenario(root, SCENARIO, "fake", client=serve(tmp_path / "dist"))


def test_id_mismatch_removes_the_file(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    entry, files = pack_vectors(root, SCENARIO, "fake", tmp_path / "dist")
    ids, vectors = read_vectors(files[1])
    write_vectors(files[1], ids[::-1], vectors[::-1])  # same vectors, wrong order
    asset = entry.queries.model_copy(update={"sha256": sha256_file(files[1])})
    _save_registry(root, [entry.model_copy(update={"queries": asset})])
    _, queries = vector_paths(root)
    queries.unlink()
    with pytest.raises(ValueError, match="do not match"):
        embed_scenario(root, SCENARIO, "fake", client=serve(tmp_path / "dist"))
    assert not queries.exists()


def test_compute_flag_and_injected_embedder_ignore_the_registry(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    pack_vectors(root, SCENARIO, "fake", tmp_path / "dist")
    result = embed_scenario(root, SCENARIO, "fake", embedder=FakeEmbedder())
    assert result.source == "computed"
    with pytest.raises(ValueError, match="unknown model"):
        embed_scenario(root, SCENARIO, "fake", compute=True)  # no embedder for "fake"


def test_no_registry_entry_computes(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    assert find_published(root, SCENARIO, "fake") is None
    assert embed_scenario(root, SCENARIO, "fake", embedder=FakeEmbedder()).source == "computed"


def test_registry_file_is_documented(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    pack_vectors(root, SCENARIO, "fake", tmp_path / "dist")
    text = (root / REGISTRY).read_text()
    assert text.startswith("# Published vectors")
    assert yaml.safe_load(text)["vectors"][0]["model"] == "fake"


def test_cli_pack_prints_the_upload_command(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    result = CliRunner().invoke(app, ["vectors", "pack", "--scenario", SCENARIO, "--model", "fake",
                                      "--out", str(tmp_path / "dist"), "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "gh release upload vectors" in result.output and "--clobber" in result.output
```

In `test_compute_flag_and_injected_embedder_ignore_the_registry`, `embed_scenario(..., embedder=FakeEmbedder())` rewrites the local vector files with freshly computed ones. That is expected: an injected embedder means "compute".

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest harness/tests/test_vector_publish.py -v`
Expected: `ModuleNotFoundError: No module named 'irbench.embed.publish'`.

- [ ] **Step 3: Implement the registry, pack and download**

Create `harness/irbench/embed/publish.py`:

```python
"""Published vectors (spec §2, §5.2): computed once, uploaded as GitHub release assets and
downloaded by `irbench embed`, so reproducing a run needs no model download or API key."""

import shutil
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import BaseModel, ConfigDict

from irbench.datasets.fetch import fetch_file, sha256_file
from irbench.embed.vectors import read_vectors
from irbench.io import read_json, read_jsonl
from irbench.paths import Workspace, slug
from irbench.scenario import load_scenario

REGISTRY = Path("embeddings") / "published.yaml"
RELEASE_REPO = "meilisearch/irbench"
DEFAULT_RELEASE = "vectors"
HEADER = (
    "# Published vectors. `irbench embed` downloads these instead of computing them.\n"
    "# Written by `irbench vectors pack`; each url is an asset of a GitHub release.\n"
)


class VectorAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str
    sha256: str
    count: int
    dim: int


class PublishedVectors(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario: str
    model: str
    dataset: str
    dataset_version: str
    seed: int  # query variants depend on the perturbation seed
    docs: VectorAsset
    queries: VectorAsset


def load_registry(root: Path) -> list[PublishedVectors]:
    path = root / REGISTRY
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [PublishedVectors.model_validate(e) for e in data.get("vectors") or []]


def find_published(root: Path, scenario_id: str, model_id: str) -> PublishedVectors | None:
    return next(
        (e for e in load_registry(root) if (e.scenario, e.model) == (scenario_id, model_id)), None
    )


def _save_registry(root: Path, entries: list[PublishedVectors]) -> None:
    path = root / REGISTRY
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(entries, key=lambda e: (e.scenario, e.model))
    body = yaml.safe_dump({"vectors": [e.model_dump() for e in ordered]}, sort_keys=False)
    path.write_text(HEADER + body, encoding="utf-8")


def asset_name(scenario_id: str, model_id: str, kind: str) -> str:
    return f"{slug(scenario_id)}__{slug(model_id)}__{kind}.parquet"


def _prepared(root: Path, scenario_id: str) -> dict[str, Any]:
    path = Workspace(root).scenario_dir(scenario_id) / "prepare.manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `irbench prepare --scenario {scenario_id}`")
    return read_json(path)


def pack_vectors(
    root: Path, scenario_id: str, model_id: str, out_dir: Path, release: str = DEFAULT_RELEASE
) -> tuple[PublishedVectors, list[Path]]:
    ws, scenario, info = Workspace(root), load_scenario(root, scenario_id), _prepared(root, scenario_id)
    sources = {
        "docs": ws.doc_vectors_path(scenario.dataset, model_id),
        "queries": ws.query_vectors_path(scenario_id, model_id),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[str, VectorAsset] = {}
    files: list[Path] = []
    for kind, src in sources.items():
        if not src.exists():
            raise FileNotFoundError(
                f"{src} missing: run `irbench embed --scenario {scenario_id} --model {model_id} "
                "--compute` first"
            )
        name = asset_name(scenario_id, model_id, kind)
        dest = out_dir / name
        shutil.copyfile(src, dest)
        ids, vectors = read_vectors(dest)
        assets[kind] = VectorAsset(
            url=f"https://github.com/{RELEASE_REPO}/releases/download/{release}/{name}",
            sha256=sha256_file(dest),
            count=len(ids),
            dim=int(vectors.shape[1]),
        )
        files.append(dest)
    entry = PublishedVectors(
        scenario=scenario_id,
        model=model_id,
        dataset=info["dataset"]["id"],
        dataset_version=info["dataset"]["version"],
        seed=info["seed"],
        **assets,
    )
    others = [e for e in load_registry(root) if (e.scenario, e.model) != (scenario_id, model_id)]
    _save_registry(root, [*others, entry])
    return entry, files


def download_published(
    root: Path, entry: PublishedVectors, client: httpx.Client | None = None
) -> tuple[Path, Path]:
    ws, scenario, info = Workspace(root), load_scenario(root, entry.scenario), _prepared(root, entry.scenario)
    local = (info["dataset"]["id"], info["dataset"]["version"], info["seed"])
    published = (entry.dataset, entry.dataset_version, entry.seed)
    if local != published:
        raise ValueError(
            f"published {entry.model} vectors for {entry.scenario} were built from "
            f"{published[0]}@{published[1]} with seed {published[2]}, but the prepared data is "
            f"{local[0]}@{local[1]} with seed {local[2]}; run `irbench embed --compute`"
        )
    expected = {
        "docs": (
            ws.doc_vectors_path(scenario.dataset, entry.model),
            [d[scenario.id_field] for d in read_jsonl(ws.dataset_dir(scenario.dataset) / "docs.jsonl")],
        ),
        "queries": (
            ws.query_vectors_path(entry.scenario, entry.model),
            [v["qid"] for v in read_jsonl(ws.scenario_dir(entry.scenario) / "queries.variants.jsonl")],
        ),
    }
    for kind, (dest, ids) in expected.items():
        asset: VectorAsset = getattr(entry, kind)
        fetch_file(asset.url, dest, asset.sha256, client=client)
        got_ids, vectors = read_vectors(dest)
        if got_ids != ids or vectors.shape[1] != asset.dim:
            dest.unlink()
            raise ValueError(
                f"{asset.url}: ids or dimension do not match the prepared {kind} "
                f"({len(got_ids)} vs {len(ids)} ids, dim {vectors.shape[1]} vs {asset.dim}); "
                "run `irbench embed --compute`"
            )
    return expected["docs"][0], expected["queries"][0]
```

- [ ] **Step 4: Use the registry in `embed_scenario`**

In `harness/irbench/stages/embed.py`, add `import httpx` and `from irbench.embed.publish import download_published, find_published`. Add the field to `EmbedResult`:

```python
    source: str = "computed"  # or "published": downloaded from embeddings/published.yaml
```

Change the signature and insert the download path right after the `variants_path` check:

```python
def embed_scenario(
    root: Path,
    scenario_id: str,
    model_id: str,
    embedder: Embedder | None = None,
    *,
    compute: bool = False,
    client: httpx.Client | None = None,
) -> EmbedResult:
    ...
    # An injected embedder or --compute means "compute"; otherwise prefer published vectors.
    if embedder is None and not compute:
        entry = find_published(root, scenario_id, model_id)
        if entry is not None:
            docs_path, queries_path = download_published(root, entry, client)
            return EmbedResult(
                docs_path, queries_path, entry.docs.count, entry.queries.count, 0, "published"
            )
```

In `harness/irbench/stages/run.py`, make the embed stage record the source:

```python
            return {"computed": result.n_computed, "source": result.source}
```

Run `grep -rn '"computed"' harness/tests` and update any assertion on the exact embed stage result to include `"source": "computed"`.

- [ ] **Step 5: CLI**

In `harness/irbench/cli.py`, extend `embed` and add the `vectors` group:

```python
from irbench.embed.publish import DEFAULT_RELEASE, RELEASE_REPO, REGISTRY, pack_vectors

vectors_app = typer.Typer(no_args_is_help=True, help="Pack embedding vectors for publishing.")
app.add_typer(vectors_app, name="vectors")


@app.command()
def embed(
    scenario: str = typer.Option(..., "--scenario"),
    model: str = typer.Option(..., "--model", help="Model id from the embedding registry."),
    compute: bool = typer.Option(
        False, "--compute", help="Embed locally even when published vectors exist."
    ),
    root: Path = ROOT_OPTION,
) -> None:
    """Embed docs and query variants with one model, or download its published vectors."""
    r = embed_scenario(root, scenario, model, compute=compute)
    typer.echo(
        f"{model}: {r.n_docs} docs, {r.n_queries} queries, {r.n_computed} newly embedded "
        f"({r.source})"
    )
    typer.echo(f"docs: {r.docs_path}\nqueries: {r.queries_path}")


@vectors_app.command("pack")
def vectors_pack(
    scenario: str = typer.Option(..., "--scenario"),
    model: str = typer.Option(..., "--model"),
    release: str = typer.Option(DEFAULT_RELEASE, "--release", help="GitHub release tag."),
    out: Path = typer.Option(Path("dist/vectors"), "--out"),
    root: Path = ROOT_OPTION,
) -> None:
    """Copy computed vectors into release assets and record them in embeddings/published.yaml."""
    out_dir = out if out.is_absolute() else root / out
    entry, files = pack_vectors(root, scenario, model, out_dir, release)
    typer.echo(f"packed {entry.docs.count} doc and {entry.queries.count} query vectors "
               f"(dim {entry.docs.dim}); updated {REGISTRY}")
    typer.echo("upload them, then commit the registry:")
    typer.echo(f"  gh release view {release} -R {RELEASE_REPO} >/dev/null 2>&1 || "
               f"gh release create {release} -R {RELEASE_REPO} --title 'Published vectors' "
               "--notes 'Embedding vectors used by IRBench runs.'")
    typer.echo(f"  gh release upload {release} {' '.join(str(f) for f in files)} "
               f"-R {RELEASE_REPO} --clobber")
```

Replace the existing `embed` command; don't add a second one.

Create `embeddings/published.yaml`:

```yaml
# Published vectors. `irbench embed` downloads these instead of computing them.
# Written by `irbench vectors pack`; each url is an asset of a GitHub release.
vectors: []
```

Add `dist/` to `.gitignore`.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest harness/tests/test_vector_publish.py -v` then `uv run pytest`
Expected: all pass. The empty registry in the repo means every existing test still computes.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check --fix . && uv run ruff format harness engines
git add harness/irbench/embed/publish.py harness/irbench/stages/embed.py harness/irbench/stages/run.py \
  harness/irbench/cli.py harness/tests embeddings/published.yaml .gitignore
git commit -m "feat: publish vectors as release assets and download them in embed"
```

---

### Task 8: Mintlify docs and README

Spec §2: "Mintlify site in `docs/` (methodology, parity rules, contributing)"; §7: parity rules at `docs/methodology/parity.mdx`. Current Mintlify reads `docs.json`. The user's global note says `mint.json`, which is the legacy name that Mintlify now migrates automatically, so this plan uses `docs.json`; flag it to the user at handoff.

**Files:**
- Create: `docs/docs.json`, `docs/.mintignore`, `docs/index.mdx`, `docs/quickstart.mdx`, `docs/methodology/overview.mdx`, `docs/methodology/parity.mdx`, `docs/methodology/performance.mdx`, `docs/contributing/engine-config.mdx`, `docs/contributing/new-engine.mdx`, `docs/reference/cli.mdx`, `docs/reference/results.mdx`
- Create: `README.md`

**Interfaces:**
- Consumes: the facts in the spec, `scenario.yaml`, the engine READMEs, `loadgen/README.md`, and the CLI commands from Tasks 2, 3, 5 and 7. Every number and flag in the docs must match those sources. Where a page states a value (step limits, seed, metrics), copy it from the file named next to it.
- Produces: the docs site that the report links to (`DEFAULT_DOCS_URL` points at `docs/` on GitHub until a hosted URL exists), and the docs job in Task 9.

- [ ] **Step 1: Config and ignore file**

`docs/docs.json`:

```json
{
  "$schema": "https://mintlify.com/docs.json",
  "theme": "mint",
  "name": "IRBench",
  "colors": { "primary": "#2a78d6" },
  "navigation": {
    "groups": [
      { "group": "Get started", "pages": ["index", "quickstart"] },
      { "group": "Methodology", "pages": ["methodology/overview", "methodology/parity", "methodology/performance"] },
      { "group": "Contributing", "pages": ["contributing/engine-config", "contributing/new-engine"] },
      { "group": "Reference", "pages": ["reference/cli", "reference/results"] }
    ]
  },
  "footer": { "socials": { "github": "https://github.com/meilisearch/irbench" } }
}
```

`docs/.mintignore` (keeps the planning documents out of the site):

```
superpowers/
```

- [ ] **Step 2: Write the pages**

Write each page in MDX with `title` and `description` frontmatter. Required content per page (use the source in parentheses for every fact):

- `index.mdx`: what IRBench is and is not (spec §1, including non-goals); the v0.1 scope (spec §12: WANDS, three engines, six modes, two models); the four success criteria; neutrality (maintained by Meilisearch, configs editable by each engine's community, only reference-machine results published).
- `quickstart.mdx`: prerequisites (Python 3.12 with `uv`, Docker, a Rust toolchain for the loadgen); `uv sync --extra embed-local`; `uv run irbench prepare --scenario ecommerce/product-search`; `uv run irbench run --scenario ecommerce/product-search --engine meilisearch --model bge-small-en-v1.5`; what the printed lines mean; preview a report with `uv run irbench report --results .irbench/runs --out .irbench/site`; local numbers are not comparable.
- `methodology/overview.mdx`:
  - the nine pipeline stages and what each writes (spec §5; `harness/irbench/stages/run.py` `STAGES`, `STAGE_OUTPUTS`);
  - "requests as data" (rendered request files);
  - dataset and qrels grades (WANDS: Exact 2 / Partial 1 / Irrelevant 0; queries without a relevant judgment are skipped);
  - modes and structured bases (`scenario.yaml`);
  - the perturbation rules (spec §6) and seed (`scenario.yaml`, 20261009);
  - metrics (nDCG@10 with linear gain, Recall@10/100, MRR@10, Success@1; cross-checked against `trec_eval`);
  - filtered/sorted/faceted scoring and the violation rules (spec §5 "Scoring of filtered…");
  - error handling (spec §9, `max_error_rate` 0.001, 10 s timeout).
- `methodology/parity.mdx`:
  - every rule from spec §7, verbatim in meaning;
  - the two tracks;
  - the table of rules `irbench lint` enforces, from Task 2: `unjustified-setting`, `unsupported-key`, `query-leak`, `image-pin`, `engine-files`, `config-load`, `not-prepared`, with one example each;
  - how N/A is decided and shown.
- `methodology/performance.mdx`:
  - warm-up, latency at one client and stepped throughput (`scenario.yaml` `perf:` block: 1…64 clients, 20 s steps, p99 ≤ 100 ms, ≤ 0.1% errors);
  - closed-loop measurement and its known limits (copy from `loadgen/README.md`);
  - indexing time, peak memory (cgroup `memory.peak`) and disk;
  - the reference hardware and pinning (`hardware/aws-c7i-4xlarge.yaml`);
  - why results are only compared within a profile.
- `contributing/engine-config.mdx`:
  - the config file layout (`default`, `tuned`, `credits`; keys `index`, `search.<base>`, `unsupported`, `justification`, from the header comment of `engines/meilisearch/configs/ecommerce/product-search.yaml`);
  - a worked tuned example with a justification and credits;
  - the checks a PR must pass (`uv run irbench lint`, `uv run pytest -m docker -k <engine>`);
  - that maintainers re-run on the reference machine.
- `contributing/new-engine.mdx`:
  - the files every engine needs (`adapter.py`, `README.md`, `docker/compose.yaml`, `configs/<scenario>.yaml`);
  - the adapter methods (read `harness/irbench/engines/base.py` and list each abstract method with one line);
  - the contract tests (`harness/tests/test_contract.py`; add the engine to `ENGINES`).
- `reference/cli.mdx`: every command with its flags, generated from `uv run irbench --help` and `uv run irbench <command> --help` (`prepare`, `embed`, `run`, `score`, `lint`, `publish`, `report`, `vectors pack`, `version`).
- `reference/results.mdx`: the bundle layout (Task 3), and the fields of `manifest.json`, `scores.json`, `perf.json` and `index.json`, with the field names copied from a real run in `.irbench/runs/`.

- [ ] **Step 3: README**

Create `README.md` at the repository root: one paragraph on what IRBench measures and its neutrality; a "Quickstart" with the four commands from `quickstart.mdx`; links to `docs/` pages (methodology, parity, contributing) and `results/`; the license (MIT, `LICENSE`).

- [ ] **Step 4: Check the site builds with no broken links**

Run: `cd docs && npx --yes mint@latest broken-links`
Expected: no broken links. If Mintlify reports the `superpowers/` Markdown files, check that `.mintignore` is supported by the installed CLI version. If it isn't, move nothing; report the CLI version and the message.

Run: `cd docs && npx --yes mint@latest dev` and open `http://localhost:3000`
Expected: every page in the navigation renders. Stop the server afterwards.

- [ ] **Step 5: Commit**

```bash
git add docs/docs.json docs/.mintignore docs/*.mdx docs/methodology docs/contributing docs/reference README.md
git commit -m "docs: Mintlify site with methodology, parity rules, contributing and reference"
```

---

### Task 9: CI and report deployment

Spec §7: "PR edits … → CI runs adapter contract tests + smoke subset"; machine-checkable parity rules enforced in CI. Spec §10: contract tests "Run in CI on every PR".

**Files:**
- Create: `.github/workflows/ci.yml`, `.github/workflows/report.yml`

**Interfaces:**
- Consumes: `uv run ruff …`, `uv run pytest` (default excludes `slow` and `docker`), `uv run pytest -m docker`, `uv run irbench prepare`, `uv run irbench lint`, `uv run irbench report`, `cargo fmt/clippy/test` in `loadgen/`, `npx mint broken-links` in `docs/`.
- Produces: required checks for PRs; a GitHub Pages deployment of `results/` on `main`.

- [ ] **Step 1: Write `ci.yml`**

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: read

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  python:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --locked
      - run: uv run ruff check .
      - run: uv run ruff format --check harness engines
      - run: uv run pytest

  parity:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --locked
      - uses: actions/cache@v4
        with:
          path: .irbench/data
          key: datasets-${{ hashFiles('datasets/**/dataset.yaml') }}
      - run: uv run irbench prepare --scenario ecommerce/product-search
      - run: uv run irbench lint

  contract:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --locked
      - run: uv run pytest -m docker -v

  loadgen:
    runs-on: ubuntu-24.04
    defaults:
      run:
        working-directory: loadgen
    steps:
      - uses: actions/checkout@v5
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
      - uses: Swatinem/rust-cache@v2
        with:
          workspaces: loadgen
      - run: cargo fmt --check
      - run: cargo clippy --all-targets -- -D warnings
      - run: cargo test

  docs:
    runs-on: ubuntu-24.04
    defaults:
      run:
        working-directory: docs
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-node@v5
        with:
          node-version: 22
      - run: npx --yes mint@latest broken-links

  report:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --locked
      - run: uv run irbench report --results results --out site
```

- [ ] **Step 2: Write `report.yml`**

```yaml
name: Report

on:
  push:
    branches: [main]
    paths: ["results/**", "harness/irbench/report/**", ".github/workflows/report.yml"]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --locked
      - run: uv run irbench report --results results --out site
      - uses: actions/upload-pages-artifact@v4
        with:
          path: site

  deploy:
    needs: build
    runs-on: ubuntu-24.04
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

GitHub Pages must be enabled for the repository with "GitHub Actions" as the source. That is a repository setting; don't change it from here. Mention it in the final report.

- [ ] **Step 3: Validate locally**

Run, from the repo root:
- `uvx --from actionlint-py actionlint .github/workflows/*.yml`
- `uv run ruff check . && uv run ruff format --check harness engines && uv run pytest`
- `uv run irbench lint`
- `uv run irbench report --results results --out .irbench/site-ci`
- `(cd loadgen && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test)`

Expected: actionlint reports nothing, and every command succeeds. If `actionlint-py` isn't installable, run `docker run --rm -v "$PWD:/repo" -w /repo rhysd/actionlint:latest` instead.

Then run `uv run pytest -m docker -v` (Docker or OrbStack running).
Expected: the contract tests pass for all three engines. This is the slowest check; run it in the background.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml .github/workflows/report.yml
git commit -m "ci: python, parity lint, contract tests, loadgen, docs and report jobs"
```

---

### Task 10: Real end-to-end check and publishing

Like Plan 3's Task 6, this task works against reality. No code is expected. A bug found here gets fixed test-first in the task that owns the code, with a ruling recorded in the ledger. **Uploading to GitHub and pushing are outward-facing: ask the user before each one.**

**Files:** `embeddings/published.yaml` (registry entries), nothing else expected.

**Interfaces:**
- Consumes: everything above; the three local runs from Plan 3 in `.irbench/runs/`; the cached `bge-small-en-v1.5` vectors.
- Produces: published `bge-small-en-v1.5` vectors (and `text-embedding-3-small` if the user provides an API key); a verified local report; the branch pushed with a PR stacked on #2.

- [ ] **Step 1: Lint on real data**

Run: `uv run irbench lint`
Expected: `lint: 3 configs, 0 findings`. Report any finding with the offending value; don't weaken a rule to pass.

- [ ] **Step 2: Preview the report from the Plan 3 runs**

Run: `uv run irbench report --results .irbench/runs --out .irbench/site && open .irbench/site/index.html`
Expected: one scenario page for profile `local`, labelled not comparable. Check by eye, in light and dark mode (toggle the OS appearance):
- the relevance matrix, with Qdrant's N/A cells;
- the perturbation table;
- three latency columns;
- max QPS matching the Plan 3 PR table (Elasticsearch keyword 2,950, Meilisearch semantic 570, Qdrant semantic 2,019);
- three charts with readable direct labels and no label collisions;
- indexing cost;
- each run page.

Take a screenshot of the scenario page in each mode for the final message. Fix any layout problem (overlapping direct labels, overflow at 375 px wide) in Task 5 or 6's code with a test where one applies.

- [ ] **Step 3: Pack the bge vectors**

Run: `uv run irbench vectors pack --scenario ecommerce/product-search --model bge-small-en-v1.5`
Expected: 42,994 doc vectors and 1,543 query vectors, dim 384. The registry entry is written and the `gh` commands are printed.

- [ ] **Step 4: Upload (ask first)**

Show the user the printed `gh release create/upload` commands and the asset sizes (`ls -lh dist/vectors`). Run them only after explicit approval. Then verify the download path end to end:

```bash
mv .irbench/embeddings/wands/bge-small-en-v1.5 /tmp/irbench-bge-backup
mv .irbench/scenarios/ecommerce__product-search/embeddings/bge-small-en-v1.5 /tmp/irbench-bge-queries-backup
uv run irbench embed --scenario ecommerce/product-search --model bge-small-en-v1.5
```

Expected: `… 0 newly embedded (published)`. Compare checksums against the backups with `shasum -a 256`, then delete the backups.

The doc vectors directory also holds `cache.parquet`. Restore it from the backup after the check, so later `--compute` runs stay incremental.

- [ ] **Step 5: OpenAI vectors (ask first)**

`text-embedding-3-small` needs `OPENAI_API_KEY` and costs money (roughly 45k short texts). Ask the user whether to compute and publish it now. If yes:
1. `uv sync --extra embed-api`
2. `uv run irbench embed --scenario ecommerce/product-search --model text-embedding-3-small --compute`
3. Pack it, then upload with approval, as in Steps 3–4.

If no, record it as a follow-up in the final message.

- [ ] **Step 6: Commit the registry**

```bash
git add embeddings/published.yaml
git commit -m "feat: publish bge-small-en-v1.5 vectors for ecommerce/product-search"
```

- [ ] **Step 7: Push and open the PR (ask first)**

With approval, push `plan4-report` and open a PR stacked on `plan3-loadgen`: "Plan 4: report, docs, CI and published vectors". In the description, include:
- what each part does;
- the local report screenshots;
- the test counts;
- the Pages setting to enable;
- whether the OpenAI vectors were published.

Watch CI and fix any job that fails only on Linux (e.g. path or locale differences) with a test.

---

## Self-Review

**Spec coverage**

| Spec section | Where |
|---|---|
| §2 Report: static HTML from results JSON, no frontend framework | Tasks 5, 6 (Jinja2 → plain HTML + inline SVG, no JS) |
| §2 Docs: Mintlify in `docs/` | Task 8 |
| §2 Embeddings: vectors cached and published, no API keys to reproduce | Task 7, Task 10 Steps 3–5 |
| §1 relevance by imperfection and performance separately, never blended | Task 5 (perturbation table, no overall score), Task 6 (separate section) |
| §4 `results/<run-id>/`, `report/`, `embeddings/`, `docs/` | Task 3, Tasks 4–6 (`harness/irbench/report/`, see below), Task 7, Task 8 |
| §5.2 embed downloads published vectors when available | Task 7 |
| §5 CLI `irbench report --results results/ --out site/` | Task 5 |
| §7 parity rules documented in `docs/methodology/parity.mdx` | Task 8 |
| §7 machine-checkable rules enforced in CI; CI rejects query IDs/strings | Task 2, Task 9 `parity` job |
| §7 justification for non-default settings | Task 2 `unjustified-setting` |
| §7 tracks; credits for tuned configs (§11) | Task 2 credits, Task 5 run page |
| §7 contribution flow: CI runs adapter contract tests | Task 9 `contract` job |
| §7 only reference-machine results published | Task 3 |
| §8 results only compared within a profile | Task 4 grouping, Task 5 page per profile |
| §9 the report shows failures explicitly | Tasks 4, 5 |
| §10 metric correctness vs `trec_eval` | Task 1 |
| §10 adapter contract tests in CI on every PR | Task 9 |
| §10 loadgen mock-server tests | Task 9 `loadgen` job (tests from Plan 3) |
| §11 engines × modes nDCG@10, perturbation deltas | Task 5 |
| §11 latency percentiles, QPS curve, indexing time/RAM/disk | Task 6 |
| §11 model comparison per engine | Task 5 (`model_comparison`, shown with ≥ 2 models) |
| §11 per run: config diff, manifest, N/A reasons, credits | Task 5 run page |
| §11 links to Mintlify methodology docs | Task 5 (`--docs-url`, header and scenario page) |
| §12 v0.1: `text-embedding-3-small` cached and published | Task 10 Step 5 (needs the user's API key; follow-up if declined) |

**Decided here (ledger these when the owning task runs):**
- The report generator lives in `harness/irbench/report/`, not a top-level `report/` (§4). It ships inside the `irbench` package so `irbench report` works from an installed wheel and templates load with `PackageLoader`. The spec's intent (a separate static generator) holds.
- `docs.json`, not `mint.json`: Mintlify's current config name. Flag to the user, whose global instructions mention `mint.json`.
- Parity lint scope: the `query-leak` check covers `index` and `search` only. `justification` and `unsupported` are prose and never reach the engine. Purely numeric query ids are skipped because they collide with numeric settings; variant and task ids are checked.
- Tuned-config credits are a top-level `credits` list in the config file, shown for the tuned track only.
- Published vectors: one GitHub release (`vectors`), asset names keyed by scenario and model, and a registry recording dataset version and seed. A mismatch fails loudly instead of computing silently.
- `local` runs are never published but can be previewed: `irbench report --results .irbench/runs` accepts full run directories.

**Out of scope for this plan:** running the benchmark on the reference machine (AWS) and publishing those results; Pinecone; the deferred perf gaps listed in PR #2.

**Placeholder scan.** Code steps carry full code. Task 8's pages are specified by required content and named sources instead of full text, because their facts must be copied from files that exist at execution time (the CLI help output, the scenario values, a real run's field names).

**Type consistency.**
- `RunResult`, `ScenarioResults`, `Unreadable`, `load_results` and `group_key` (Task 4) are used with the same names in Tasks 5 and 6.
- `Cell(text, kind, note)`, `engine_slots` and `scenario_view(s, slots)` (Task 5) are extended by `perf_view(s, slots)` (Task 6).
- `Point`/`Series`/`qps_chart` keep their fields between the chart tests and `_series`.
- `PublishedVectors`/`VectorAsset` field names match between `pack_vectors`, `download_published`, the registry YAML and the tests.
- `EmbedResult.source` is set in `embed_scenario` and read by the CLI and the run stage.
