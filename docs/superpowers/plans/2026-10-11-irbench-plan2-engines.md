# IRBench Plan 2 — Docker Runtime and Engine Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the WANDS product-search scenario end to end against real engines: `irbench run` starts a pinned Meilisearch, Elasticsearch or Qdrant container, indexes the documents with the shared vectors, renders every task into an inspectable request file, executes those requests, and writes a scored, manifested run (relevance per mode and perturbation, filter/order/facet checks, N/A reasons).

**Architecture:** Engine-specific code lives in `engines/<engine>/` (adapter, config per scenario with `default`/`tuned` tracks, Docker compose file, parity README), so a competitor edits one folder. The harness loads `engines/<engine>/adapter.py` by path; adapters are pure renderers/parsers plus three index calls, all through one small `EngineAdapter` base class. Stages communicate through files in `.irbench/runs/<run-id>/` (`index.json`, `rendered/`, `na.json`, `run.trec`, `checks.jsonl`, `errors.jsonl`, `scores.json`, `manifest.json`), so Plan 3's Rust loadgen replays exactly the rendered requests used for relevance.

**Tech Stack:** Python 3.12, httpx, pydantic v2, PyYAML, typer, numpy/pyarrow (existing); Docker Engine + Docker Compose v2 (`docker compose up --wait`); images `getmeili/meilisearch:v1.54.3`, `elasticsearch:9.5.5`, `qdrant/qdrant:v1.19.2`.

**Spec:** `docs/superpowers/specs/2026-10-09-irbench-design.md`

**Plan series:** Plan 2 of 4 for v0.1. Plan 1 (merged) built prepare/embed/score. Plan 3: Rust loadgen + `perf` stage (reads `rendered/` and the hardware profile's `loadgen.cpuset`). Plan 4: static HTML report, Mintlify docs, CI (parity lint incl. `justification:` enforcement, contract tests in CI), publishing vectors, `trec_eval` cross-check.

## Global Constraints

- Python `>=3.12,<3.13`. All engine HTTP goes through `httpx`; no engine SDKs.
- Engines never embed: documents and queries carry the vectors from `irbench embed`; cosine similarity everywhere.
- Images pinned by tag in `engines/<engine>/docker/compose.yaml`: `getmeili/meilisearch:v1.54.3`, `elasticsearch:9.5.5`, `qdrant/qdrant:v1.19.2`. The resolved digest is recorded in every run manifest.
- Host ports are offset so a developer's own engines keep their default ports: Meilisearch `17700`, Elasticsearch `19200`, Qdrant `16333`.
- Engine config files: `engines/<engine>/configs/<scenario-id>.yaml` with top-level keys `default` (required) and `tuned` (optional, deep-merged over `default`). Track keys: `index`, `search` (per retrieval base: `keyword|semantic|hybrid`), `unsupported` (key → reason), `justification`.
- **default track** = engine as shipped plus the minimum to load the scenario schema (searchable/filterable/sortable/facet fields, synonyms, vector field, facet value limit). ANN parameters stay at each engine's defaults.
- Field priority comes from `index_fields.searchable` order: Meilisearch uses attribute order natively; Elasticsearch uses boosts `n - i` (name^5 … description^1).
- Every engine with keyword retrieval loads the scenario's `synonyms.txt`.
- Hybrid uses the engine's native fusion only (Meilisearch `hybrid`, Elasticsearch `query` + `knn` score combination). No harness-side fusion. Otherwise N/A.
- Unsupported (mode, base) pairs are declared with a reason in the engine config and reported as N/A — never scored as 0.
- `top_k` = 100. Relevance request timeout = `request_timeout_s` (10 s); an error or timeout → empty result (scored 0) and a line in `errors.jsonl`.
- Facet checks: facet value limit `max_facet_values` = 1000 (WANDS has 601 `product_class` values); the verification fetch uses `facet_verify_limit` = 1000 and is "unverifiable" when it returns ≥ that many hits.
- A failed stage marks the run `failed` with its reason in `manifest.json`; if the container was started it is always torn down.
- Engine API details in this plan were written from each engine's public docs; the Docker contract tests (Task 12) are the authority. When a contract test shows an API differs, fix the adapter (or declare the mode unsupported with a truthful reason) and ledger a ruling.
- Commits must NOT contain `Co-Authored-By` lines (user rule).

## Review Focus

1. **A stage fails after `up`** (index error, bad config) — the container must still be torn down and the manifest must say `failed` with the reason; later stages must not run. Test: Task 8 `test_failed_stage_records_reason_and_tears_down`.
2. **Engine errors or times out on some requests** — those tasks score 0, appear in `errors.jsonl`, and the run continues. Tests: Task 7 `test_relevance_writes_run_checks_and_errors`, `test_slow_engine_times_out_and_scores_empty`.
3. **A mode the engine cannot do** — reported as N/A with the config's reason, absent from relevance tables, never a 0. Test: Task 8 `test_pipeline_end_to_end_with_fake_engine` (asserts `not_applicable`).
4. **Faceted query whose result set exceeds the verification limit** — marked unverifiable, not counted as wrong; duplicate IDs in the verification fetch must not double counts. Tests: Task 5 `test_facet_check_at_limit_is_unverifiable`, `test_duplicate_check_ids_count_once`.
5. **Re-indexing into an engine that already holds the index** (second run) — the old index is replaced: no duplicates, no stale documents. Test: Task 12 `test_reindex_replaces_previous_index`.

---

## File Structure

```
pyproject.toml                               # + docker marker, first-party test helpers
hardware/local.yaml                          # no pinning (dev machines)
hardware/aws-c7i-4xlarge.yaml                # reference profile (engine 0-7 / 16g, loadgen 8-11)
scenarios/ecommerce/product-search/scenario.yaml   # + numeric fields, facet limits, timeout
engines/
  meilisearch/{adapter.py, README.md, docker/compose.yaml, configs/ecommerce/product-search.yaml}
  elasticsearch/{adapter.py, README.md, docker/compose.yaml, configs/ecommerce/product-search.yaml}
  qdrant/{adapter.py, README.md, docker/compose.yaml, configs/ecommerce/product-search.yaml}
harness/irbench/
  paths.py                                   # + slug(), doc/query vector paths, run_dir
  io.py                                      # + write_json/read_json
  scenario.py                                # + numeric, max_facet_values, facet_verify_limit, request_timeout_s
  runs.py                                    # STAGES, RunSpec (run_id)
  engines/__init__.py
  engines/config.py                          # tracks, deep_merge, config_diff, hash, unsupported
  engines/base.py                            # Request, SearchResult, IndexContext, EngineAdapter, call, poll, loader
  runtime/__init__.py
  runtime/hardware.py                        # HardwareProfile
  runtime/docker.py                          # DockerRuntime, Runtime protocol, wait_healthy
  score/checks.py                            # per-task filter/order/facet checks + summaries
  stages/context.py                          # RunContext: lazily loaded run inputs
  stages/render.py
  stages/index.py
  stages/relevance.py
  stages/score_run.py
  stages/run.py                              # pipeline orchestrator + manifest
  stages/embed.py                            # use Workspace vector paths
  cli.py                                     # + run command
harness/tests/
  fake_engine.py                             # FakeAdapter, FakeServer, FakeRuntime, fake_root
  adapter_helpers.py                         # make_adapter, make_task
  contract_fixture.py                        # 100-doc known-answer corpus
  test_paths.py, test_engine_config.py, test_engine_base.py, test_runtime.py, test_checks.py,
  test_render_stage.py, test_index_relevance.py, test_run_pipeline.py,
  test_adapter_meilisearch.py, test_adapter_elasticsearch.py, test_adapter_qdrant.py,
  test_contract.py                           # docker-marked, real engines
```

---

### Task 1: Scenario, workspace and JSON helpers for runs

**Files:**
- Modify: `harness/irbench/scenario.py`, `harness/irbench/paths.py`, `harness/irbench/io.py`, `harness/irbench/stages/embed.py`, `scenarios/ecommerce/product-search/scenario.yaml`
- Test: `harness/tests/test_scenario.py`, `harness/tests/test_paths.py`, `harness/tests/test_io.py`, `harness/tests/test_embed_stage.py`

**Interfaces:**
- Consumes: Plan 1 `Scenario`, `Workspace`, `embed_scenario`.
- Produces:
  - `IndexFields.numeric: list[str]`; `Scenario.max_facet_values: int = 1000`, `Scenario.facet_verify_limit: int = 1000`, `Scenario.request_timeout_s: float = 10.0`.
  - `irbench.paths.slug(value: str) -> str`; `Workspace.doc_vectors_path(dataset_id, model_id) -> Path`; `Workspace.query_vectors_path(scenario_id, model_id) -> Path`; `Workspace.run_dir(run_id) -> Path` (= `.irbench/runs/<run_id>`).
  - `irbench.io.write_json(path, data, indent: int | None = 2) -> None`, `irbench.io.read_json(path) -> Any`.

- [ ] **Step 1: Write the failing tests**

Append to `harness/tests/test_scenario.py`:

```python
def _scenario_with(tmp_path, mutate):
    raw = yaml.safe_load((REPO_ROOT / "scenarios" / SCENARIO / "scenario.yaml").read_text())
    mutate(raw)
    target = tmp_path / "scenarios" / SCENARIO
    target.mkdir(parents=True)
    (target / "scenario.yaml").write_text(yaml.safe_dump(raw))
    return lambda: load_scenario(tmp_path, SCENARIO)


def test_scenario_declares_numeric_fields_and_facet_limits():
    s = load_scenario(REPO_ROOT, SCENARIO)
    assert s.index_fields.numeric == ["average_rating", "review_count"]
    assert s.max_facet_values >= 601  # WANDS has 601 distinct product_class values
    assert s.facet_verify_limit == 1000
    assert s.request_timeout_s == 10


def test_range_filter_on_non_numeric_field_is_rejected(tmp_path):
    load = _scenario_with(
        tmp_path, lambda raw: raw["index_fields"].update(numeric=["review_count"])
    )
    with pytest.raises(ValueError, match="average_rating"):
        load()


def test_sortable_field_must_be_numeric(tmp_path):
    def mutate(raw):
        raw["index_fields"]["sortable"].append("product_class")

    with pytest.raises(ValueError, match="product_class"):
        _scenario_with(tmp_path, mutate)()
```

Create `harness/tests/test_paths.py`:

```python
from irbench.paths import Workspace, slug


def test_slug_and_run_dir(tmp_path):
    ws = Workspace(tmp_path)
    assert slug("ecommerce/product-search") == "ecommerce__product-search"
    assert ws.run_dir("abc") == tmp_path / ".irbench" / "runs" / "abc"


def test_vector_paths(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.doc_vectors_path("wands", "org/model") == (
        tmp_path / ".irbench" / "embeddings" / "wands" / "org__model" / "docs.parquet"
    )
    assert ws.query_vectors_path("ecommerce/product-search", "org/model") == (
        tmp_path
        / ".irbench"
        / "scenarios"
        / "ecommerce__product-search"
        / "embeddings"
        / "org__model"
        / "queries.parquet"
    )
```

Append to `harness/tests/test_io.py` (add `read_json, write_json` to its `irbench.io` import):

```python
def test_json_roundtrip_creates_parents(tmp_path):
    path = tmp_path / "a" / "b.json"
    write_json(path, {"b": 1, "a": [1.5]})
    assert read_json(path) == {"b": 1, "a": [1.5]}
    assert path.read_text().endswith("\n")
    write_json(path, {"x": [1, 2]}, indent=None)
    assert path.read_text() == '{"x": [1, 2]}\n'
```

Append to `harness/tests/test_embed_stage.py` (add `from irbench.paths import Workspace`):

```python
def test_vectors_land_at_workspace_paths(prepared_root):
    result = embed_scenario(prepared_root, SCENARIO, "fake", embedder=FakeEmbedder())
    ws = Workspace(prepared_root)
    assert result.docs_path == ws.doc_vectors_path("wands", "fake")
    assert result.queries_path == ws.query_vectors_path(SCENARIO, "fake")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest harness/tests/test_scenario.py harness/tests/test_paths.py harness/tests/test_io.py harness/tests/test_embed_stage.py -q`
Expected: FAIL — `AttributeError: 'IndexFields' object has no attribute 'numeric'`, `ImportError: cannot import name 'slug'`, `cannot import name 'read_json'`, `'Workspace' object has no attribute 'doc_vectors_path'`. The two "is rejected" tests fail with `DID NOT RAISE`.

- [ ] **Step 3: Implement**

`harness/irbench/scenario.py` — add `numeric` to `IndexFields`, the three scenario fields after `top_k`, and two checks in `_check_consistency`:

```python
class IndexFields(BaseModel):
    searchable: list[str]
    filterable: list[str]
    sortable: list[str]
    facets: list[str]
    numeric: list[str] = []
```

```python
    top_k: int = 100
    max_facet_values: int = 1000
    facet_verify_limit: int = 1000
    request_timeout_s: float = 10.0
```

```python
        for t in self.filter_templates:
            if t.field not in f.filterable:
                raise ValueError(f"filter template field {t.field!r} is not filterable")
            if (t.op == "eq") != (t.values is None):
                raise ValueError(
                    f"filter template {t.field!r}: eq takes no values, ranges need values"
                )
            if t.op != "eq" and t.field not in f.numeric:
                raise ValueError(
                    f"range filter template field {t.field!r} must be listed in index_fields.numeric"
                )
        for name in f.sortable:
            if name not in f.numeric:
                raise ValueError(f"sortable field {name!r} must be numeric (v0.1 sorts numbers only)")
```

`harness/irbench/paths.py` — make `slug` public and add the paths:

```python
from dataclasses import dataclass
from pathlib import Path


def slug(value: str) -> str:
    return value.replace("/", "__")


@dataclass(frozen=True)
class Workspace:
    root: Path

    @property
    def cache(self) -> Path:
        return self.root / ".irbench"

    def dataset_dir(self, dataset_id: str) -> Path:
        return self.cache / "data" / dataset_id

    def raw_dir(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "raw"

    def scenario_dir(self, scenario_id: str) -> Path:
        return self.cache / "scenarios" / slug(scenario_id)

    def embeddings_dir(self, dataset_id: str, model_id: str) -> Path:
        return self.cache / "embeddings" / dataset_id / slug(model_id)

    def doc_vectors_path(self, dataset_id: str, model_id: str) -> Path:
        return self.embeddings_dir(dataset_id, model_id) / "docs.parquet"

    def query_vectors_path(self, scenario_id: str, model_id: str) -> Path:
        return self.scenario_dir(scenario_id) / "embeddings" / slug(model_id) / "queries.parquet"

    def run_dir(self, run_id: str) -> Path:
        return self.cache / "runs" / run_id
```

`harness/irbench/io.py` — add:

```python
def write_json(path: Path, data: Any, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=indent, sort_keys=indent is not None, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
```

`harness/irbench/stages/embed.py` — replace the two path computations at the end:

```python
    docs_path = ws.doc_vectors_path(scenario.dataset, model_id)
    queries_path = ws.query_vectors_path(scenario_id, model_id)
```

`scenarios/ecommerce/product-search/scenario.yaml` — add `numeric` under `index_fields` and the limits after `top_k`:

```yaml
index_fields:
  searchable: [name, product_class, category, features, description]
  filterable: [product_class, category, average_rating, review_count]
  sortable: [average_rating, review_count]
  facets: [product_class]
  # Numeric fields: range filters and sorting apply to these only.
  numeric: [average_rating, review_count]
```

```yaml
top_k: 100
# Facet value limit every engine must return (WANDS has 601 product_class values).
max_facet_values: 1000
# Faceted checks re-fetch the same query up to this many hits; at or above it the
# result set is too large to verify and the task is reported "unverifiable".
facet_verify_limit: 1000
# Relevance-stage request timeout; a timeout scores the task 0 and is logged.
request_timeout_s: 10
index_timeout_s: 1800
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass (70 existing + 7 new = 77 passed, 1 deselected).

- [ ] **Step 5: Commit**

```bash
git add harness/irbench/scenario.py harness/irbench/paths.py harness/irbench/io.py harness/irbench/stages/embed.py scenarios/ecommerce/product-search/scenario.yaml harness/tests/test_scenario.py harness/tests/test_paths.py harness/tests/test_io.py harness/tests/test_embed_stage.py
git commit -m "feat: scenario numeric fields, facet limits and run workspace paths"
```

---

### Task 2: Engine configs with default and tuned tracks

**Files:**
- Create: `harness/irbench/engines/__init__.py` (empty), `harness/irbench/engines/config.py`
- Test: `harness/tests/test_engine_config.py`

**Interfaces:**
- Consumes: `QueryTask`, `BASE_MODES` (Plan 1 `irbench.models`).
- Produces:
  - `TRACKS = ("default", "tuned")`
  - `deep_merge(base: dict, override: dict) -> dict` — nested dicts merge, anything else replaces; inputs untouched.
  - `config_diff(default: dict, effective: dict) -> dict[str, dict[str, Any]]` — `{"dotted.path": {"default": a, "effective": b}}`.
  - `TrackConfig` (pydantic, `extra="forbid"`): `index: dict`, `search: dict[str, dict]` (keys ⊆ `keyword|semantic|hybrid`), `unsupported: dict[str, str]`, `justification: dict[str, str]`; `unsupported_reason(task) -> str | None` checks keys `"<mode>:<base>"`, `"<mode>"`, `"*:<base>"` in that order.
  - `EngineConfig(path, raw)`: `.track(name) -> TrackConfig`, `.diff(name) -> dict`, `.hash(name) -> str` (sha256 hex of the canonical JSON of the effective track).
  - `config_path(root, engine, scenario_id) -> Path`, `load_engine_config(root, engine, scenario_id) -> EngineConfig`.

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_engine_config.py`

```python
import pytest
import yaml

from irbench.engines.config import (
    config_diff,
    deep_merge,
    load_engine_config,
)
from irbench.models import QueryTask

SCENARIO = "ecommerce/product-search"


def _write(root, engine, raw):
    path = root / "engines" / engine / "configs" / "ecommerce" / "product-search.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(raw))
    return path


def _task(mode, base):
    return QueryTask(
        task_id=f"1:{mode}:{base}", qid="1", variant_of="1", perturbation="original",
        text="sofa", mode=mode, base=base,
    )


def test_deep_merge_is_recursive_and_pure():
    base = {"a": {"x": 1, "y": 2}, "b": [1]}
    merged = deep_merge(base, {"a": {"y": 3}, "b": [2]})
    assert merged == {"a": {"x": 1, "y": 3}, "b": [2]}
    assert base == {"a": {"x": 1, "y": 2}, "b": [1]}


def test_config_diff_lists_changed_paths():
    assert config_diff({"a": {"x": 1}, "b": 2}, {"a": {"x": 5}, "b": 2, "c": 3}) == {
        "a.x": {"default": 1, "effective": 5},
        "c": {"default": None, "effective": 3},
    }


def test_tuned_track_merges_over_default(tmp_path):
    _write(tmp_path, "e", {
        "default": {"index": {"k": 1, "keep": True}, "search": {"keyword": {"x": 1}}},
        "tuned": {"index": {"k": 2}},
    })
    cfg = load_engine_config(tmp_path, "e", SCENARIO)
    assert cfg.track("tuned").index == {"k": 2, "keep": True}
    assert cfg.track("default").search == {"keyword": {"x": 1}}
    assert cfg.diff("default") == {}
    assert cfg.diff("tuned") == {"index.k": {"default": 1, "effective": 2}}
    assert cfg.hash("default") != cfg.hash("tuned")
    assert len(cfg.hash("default")) == 64


def test_missing_tuned_track_is_an_error(tmp_path):
    _write(tmp_path, "e", {"default": {}})
    with pytest.raises(ValueError, match="tuned"):
        load_engine_config(tmp_path, "e", SCENARIO).track("tuned")


def test_unknown_keys_are_rejected(tmp_path):
    _write(tmp_path, "e", {"default": {"serch": {}}})
    with pytest.raises(ValueError, match="serch"):
        load_engine_config(tmp_path, "e", SCENARIO)


def test_search_params_keyed_by_base_mode_only(tmp_path):
    _write(tmp_path, "e", {"default": {"search": {"filtered": {}}}})
    with pytest.raises(ValueError, match="filtered"):
        load_engine_config(tmp_path, "e", SCENARIO)


def test_missing_config_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="product-search.yaml"):
        load_engine_config(tmp_path, "e", SCENARIO)


def test_unsupported_reason_precedence(tmp_path):
    _write(tmp_path, "e", {"default": {"unsupported": {
        "faceted:semantic": "specific",
        "hybrid": "whole mode",
        "*:keyword": "whole base",
    }}})
    track = load_engine_config(tmp_path, "e", SCENARIO).track("default")
    assert track.unsupported_reason(_task("faceted", "semantic")) == "specific"
    assert track.unsupported_reason(_task("hybrid", "hybrid")) == "whole mode"
    assert track.unsupported_reason(_task("sorted", "keyword")) == "whole base"
    assert track.unsupported_reason(_task("sorted", "semantic")) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest harness/tests/test_engine_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'irbench.engines'`.

- [ ] **Step 3: Implement** — `harness/irbench/engines/config.py` (and an empty `harness/irbench/engines/__init__.py`)

```python
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from irbench.models import BASE_MODES, QueryTask

TRACKS = ("default", "tuned")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def config_diff(
    default: dict[str, Any], effective: dict[str, Any], prefix: str = ""
) -> dict[str, dict[str, Any]]:
    diff: dict[str, dict[str, Any]] = {}
    for key in sorted(set(default) | set(effective)):
        path = f"{prefix}{key}"
        a, b = default.get(key), effective.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            diff.update(config_diff(a, b, path + "."))
        elif a != b:
            diff[path] = {"default": a, "effective": b}
    return diff


class TrackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: dict[str, Any] = {}
    search: dict[str, dict[str, Any]] = {}
    unsupported: dict[str, str] = {}
    justification: dict[str, str] = {}

    @field_validator("search")
    @classmethod
    def _search_keys_are_bases(cls, value: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        unknown = sorted(set(value) - set(BASE_MODES))
        if unknown:
            raise ValueError(f"search keys must be retrieval bases {BASE_MODES}, got {unknown}")
        return value

    def unsupported_reason(self, task: QueryTask) -> str | None:
        for key in (f"{task.mode}:{task.base}", task.mode, f"*:{task.base}"):
            if key in self.unsupported:
                return self.unsupported[key]
        return None


@dataclass(frozen=True)
class EngineConfig:
    path: Path
    raw: dict[str, Any]

    def _data(self, name: str) -> dict[str, Any]:
        if name not in TRACKS:
            raise ValueError(f"unknown track {name!r}; known: {', '.join(TRACKS)}")
        default = self.raw.get("default") or {}
        if name == "default":
            return default
        if "tuned" not in self.raw:
            raise ValueError(f"{self.path}: no tuned track defined")
        return deep_merge(default, self.raw["tuned"] or {})

    def track(self, name: str) -> TrackConfig:
        try:
            return TrackConfig.model_validate(self._data(name))
        except ValueError as e:
            raise ValueError(f"{self.path} [{name}]: {e}") from e

    def diff(self, name: str) -> dict[str, dict[str, Any]]:
        return config_diff(self.track("default").model_dump(), self.track(name).model_dump())

    def hash(self, name: str) -> str:
        canonical = json.dumps(self.track(name).model_dump(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


def config_path(root: Path, engine: str, scenario_id: str) -> Path:
    return root / "engines" / engine / "configs" / f"{scenario_id}.yaml"


def load_engine_config(root: Path, engine: str, scenario_id: str) -> EngineConfig:
    path = config_path(root, engine, scenario_id)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: engine {engine!r} has no config for {scenario_id}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = sorted(set(raw) - set(TRACKS))
    if unknown or "default" not in raw:
        raise ValueError(f"{path}: top-level keys must be 'default' and optional 'tuned', got {sorted(raw)}")
    config = EngineConfig(path, raw)
    for name in TRACKS:
        if name in raw:
            config.track(name)
    return config
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest harness/tests/test_engine_config.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add harness/irbench/engines harness/tests/test_engine_config.py
git commit -m "feat: engine configs with default and tuned tracks"
```

---

### Task 3: Adapter contract, rendered requests and adapter loader

**Files:**
- Create: `harness/irbench/engines/base.py`
- Test: `harness/tests/test_engine_base.py`

**Interfaces:**
- Consumes: `TrackConfig` (Task 2), `Scenario` (Task 1), `QueryTask`.
- Produces:
  - `Request(method, path, body=None, content=None, headers={}, protocol="http")` frozen dataclass; `.to_json() -> dict` (has `body` **or** `content`), `Request.from_json(dict)`, `.send(client, timeout=None) -> httpx.Response`. This JSON shape is the rendered-request file format Plan 3's loadgen reads.
  - `EngineError(RuntimeError)`; `call(client, request, timeout=None) -> Any` — sends, raises `EngineError("<METHOD> <path> -> <status>: <text>")` on status ≥ 400, returns parsed JSON or `None`.
  - `SearchResult(ids: list[str], facets: dict[str, dict[str, int]] | None = None)`.
  - `IndexContext(scenario, config: TrackConfig, index: str, dim: int, synonyms: list[list[str]])`.
  - `EngineAdapter` (ABC) class attributes `name`, `host_port`, `health_path`, `data_dir`; `__init__(ctx)`; abstract `create_index(client)`, `load_documents(client, docs, vectors)`, `wait_until_indexed(client, n_docs, timeout_s)`, `render(task, vector, limit, facets=True) -> Request`, `parse(task, payload) -> SearchResult`; concrete `unsupported_reason(task)`, `search_params(task) -> dict`.
  - `require_vector(task, vector) -> list[float]`, `index_name(scenario_id, model_id) -> str`, `poll(check, timeout_s, what, interval_s=0.5)`, `load_adapter_class(root, engine) -> type[EngineAdapter]`.

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_engine_base.py`

```python
import json

import httpx
import pytest

from irbench.engines.base import (
    EngineAdapter,
    EngineError,
    Request,
    call,
    index_name,
    load_adapter_class,
    poll,
    require_vector,
)
from irbench.models import QueryTask

TOY = '''
from irbench.engines.base import EngineAdapter


class Adapter(EngineAdapter):
    name = "toy"
    host_port = 1
    health_path = "/"
    data_dir = "/d"

    def create_index(self, client): pass
    def load_documents(self, client, docs, vectors): pass
    def wait_until_indexed(self, client, n_docs, timeout_s): pass
    def render(self, task, vector, limit, facets=True): pass
    def parse(self, task, payload): pass
'''


def _engine_file(root, engine, source):
    path = root / "engines" / engine / "adapter.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)


def test_request_json_roundtrip():
    r = Request("POST", "/i/search", body={"q": "sofa"})
    assert r.to_json() == {
        "protocol": "http", "method": "POST", "path": "/i/search", "headers": {},
        "body": {"q": "sofa"},
    }
    assert Request.from_json(json.loads(json.dumps(r.to_json()))) == r
    raw = Request("POST", "/_bulk", content="{}\n", headers={"Content-Type": "application/x-ndjson"})
    assert "body" not in raw.to_json()
    assert Request.from_json(raw.to_json()) == raw


def test_call_sends_json_or_raw_content_and_raises_on_http_error():
    seen = []

    def handler(req):
        seen.append((req.headers.get("content-type"), req.content))
        if req.url.path == "/bad":
            return httpx.Response(400, text="nope")
        if req.url.path == "/empty":
            return httpx.Response(204)
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(base_url="http://e", transport=httpx.MockTransport(handler))
    assert call(client, Request("POST", "/j", body={"a": 1})) == {"ok": True}
    call(client, Request("POST", "/n", content='{"x":1}\n', headers={"Content-Type": "application/x-ndjson"}))
    assert seen[0][0] == "application/json" and json.loads(seen[0][1]) == {"a": 1}
    assert seen[1] == ("application/x-ndjson", b'{"x":1}\n')
    assert call(client, Request("DELETE", "/empty")) is None
    with pytest.raises(EngineError, match="GET /bad -> 400: nope"):
        call(client, Request("GET", "/bad"))


def test_poll_returns_when_ready_and_times_out():
    state = iter([False, False, True])
    poll(lambda: next(state), timeout_s=5, what="x", interval_s=0)
    with pytest.raises(TimeoutError, match="waiting for thing"):
        poll(lambda: False, timeout_s=0, what="thing", interval_s=0)


def test_index_name_is_engine_safe():
    assert index_name("ecommerce/product-search", "bge-small-en-v1.5") == (
        "irbench_ecommerce_product_search_bge_small_en_v1_5"
    )


def test_require_vector():
    task = QueryTask(task_id="1:semantic:semantic", qid="1", variant_of="1",
                     perturbation="original", text="x", mode="semantic", base="semantic")
    assert require_vector(task, [0.5]) == [0.5]
    with pytest.raises(ValueError, match="1:semantic:semantic"):
        require_vector(task, None)


def test_load_adapter_class(tmp_path):
    _engine_file(tmp_path, "toy", TOY)
    cls = load_adapter_class(tmp_path, "toy")
    assert issubclass(cls, EngineAdapter) and cls.name == "toy"


def test_load_adapter_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope"):
        load_adapter_class(tmp_path, "nope")
    _engine_file(tmp_path, "bare", "x = 1\n")
    with pytest.raises(TypeError, match="Adapter"):
        load_adapter_class(tmp_path, "bare")
    _engine_file(tmp_path, "other", TOY)
    with pytest.raises(ValueError, match="toy"):
        load_adapter_class(tmp_path, "other")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest harness/tests/test_engine_base.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'irbench.engines.base'`.

- [ ] **Step 3: Implement** — `harness/irbench/engines/base.py`

```python
import importlib.util
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import httpx
import numpy as np

from irbench.engines.config import TrackConfig
from irbench.models import QueryTask
from irbench.scenario import Scenario


@dataclass(frozen=True)
class Request:
    """One HTTP request, exactly as written to a rendered-request file."""

    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    path: str
    body: Any = None
    content: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    protocol: Literal["http"] = "http"

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "protocol": self.protocol,
            "method": self.method,
            "path": self.path,
            "headers": dict(self.headers),
        }
        if self.content is not None:
            out["content"] = self.content
        else:
            out["body"] = self.body
        return out

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Request":
        return cls(
            method=data["method"],
            path=data["path"],
            body=data.get("body"),
            content=data.get("content"),
            headers=dict(data.get("headers") or {}),
            protocol=data.get("protocol", "http"),
        )

    def send(self, client: httpx.Client, timeout: float | None = None) -> httpx.Response:
        kwargs: dict[str, Any] = {"headers": self.headers}
        if self.content is not None:
            kwargs["content"] = self.content
        elif self.body is not None:
            kwargs["json"] = self.body
        if timeout is not None:
            kwargs["timeout"] = timeout
        return client.request(self.method, self.path, **kwargs)


class EngineError(RuntimeError):
    pass


def call(client: httpx.Client, request: Request, timeout: float | None = None) -> Any:
    response = request.send(client, timeout)
    if response.status_code >= 400:
        raise EngineError(
            f"{request.method} {request.path} -> {response.status_code}: {response.text[:500]}"
        )
    return response.json() if response.content else None


@dataclass(frozen=True)
class SearchResult:
    ids: list[str]
    facets: dict[str, dict[str, int]] | None = None


@dataclass(frozen=True)
class IndexContext:
    scenario: Scenario
    config: TrackConfig
    index: str
    dim: int
    synonyms: list[list[str]]


class EngineAdapter(ABC):
    """Everything engine-specific: index setup, request rendering, response parsing."""

    name: ClassVar[str]
    host_port: ClassVar[int]
    health_path: ClassVar[str]
    data_dir: ClassVar[str]

    def __init__(self, ctx: IndexContext):
        self.ctx = ctx

    @abstractmethod
    def create_index(self, client: httpx.Client) -> None:
        """Drop any existing index with this name, then create it from the scenario schema."""

    @abstractmethod
    def load_documents(
        self, client: httpx.Client, docs: list[dict[str, Any]], vectors: np.ndarray
    ) -> None:
        """Send every document with its vector (row i of `vectors` belongs to docs[i])."""

    @abstractmethod
    def wait_until_indexed(self, client: httpx.Client, n_docs: int, timeout_s: float) -> None:
        """Return once all n_docs are searchable; raise TimeoutError or EngineError otherwise."""

    @abstractmethod
    def render(
        self, task: QueryTask, vector: list[float] | None, limit: int, facets: bool = True
    ) -> Request:
        """The search request for one task. `facets=False` renders the facet verification fetch."""

    @abstractmethod
    def parse(self, task: QueryTask, payload: Any) -> SearchResult:
        """Doc IDs in rank order (and facet counts when requested) from a response body."""

    def unsupported_reason(self, task: QueryTask) -> str | None:
        return self.ctx.config.unsupported_reason(task)

    def search_params(self, task: QueryTask) -> dict[str, Any]:
        return self.ctx.config.search.get(task.base, {})


def require_vector(task: QueryTask, vector: list[float] | None) -> list[float]:
    if vector is None:
        raise ValueError(f"task {task.task_id} needs a query vector")
    return vector


def index_name(scenario_id: str, model_id: str) -> str:
    return "irbench_" + re.sub(r"[^a-z0-9]+", "_", f"{scenario_id}__{model_id}".lower()).strip("_")


def poll(
    check: Callable[[], bool], timeout_s: float, what: str, interval_s: float = 0.5
) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        if check():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out after {timeout_s:g}s waiting for {what}")
        time.sleep(interval_s)


def load_adapter_class(root: Path, engine: str) -> type[EngineAdapter]:
    path = root / "engines" / engine / "adapter.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: unknown engine {engine!r}")
    spec = importlib.util.spec_from_file_location(f"irbench_engine_{engine}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, "Adapter", None)
    if not (isinstance(cls, type) and issubclass(cls, EngineAdapter)):
        raise TypeError(f"{path} must define `Adapter`, a subclass of EngineAdapter")
    if cls.name != engine:
        raise ValueError(f"{path}: Adapter.name is {cls.name!r}, expected {engine!r}")
    return cls
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest harness/tests/test_engine_base.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add harness/irbench/engines/base.py harness/tests/test_engine_base.py
git commit -m "feat: engine adapter contract, rendered requests and adapter loader"
```

---

### Task 4: Hardware profiles and the Docker runtime

**Files:**
- Create: `hardware/local.yaml`, `hardware/aws-c7i-4xlarge.yaml`, `harness/irbench/runtime/__init__.py` (empty), `harness/irbench/runtime/hardware.py`, `harness/irbench/runtime/docker.py`
- Test: `harness/tests/test_runtime.py`

**Interfaces:**
- Consumes: `poll` (Task 3), `Workspace` (Task 1).
- Produces:
  - `HardwareProfile(id, description, engine: EngineLimits(cpuset, memory), loadgen: LoadgenLimits(cpuset))`; `load_hardware(root, profile_id) -> HardwareProfile`.
  - `Runtime` protocol: `up()`, `down()`, `image_info() -> dict[str, str | None]` (`{"image", "digest"}`), `memory_peak_bytes() -> int | None`, `disk_bytes(path) -> int | None`.
  - `DockerRuntime(root, engine, profile, runner=run_command, state_dir=None)` — compose project and container both named `irbench-<engine>`, compose service named `engine`, file `engines/<engine>/docker/compose.yaml`; writes `.irbench/runtime/<engine>.override.yaml` only when the profile pins CPU/memory.
  - `run_command(cmd: list[str]) -> str` (raises `RuntimeError` with stderr); `wait_healthy(client, path, timeout_s, interval_s=1.0)`.

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_runtime.py`

```python
import httpx
import pytest
import yaml

from conftest import REPO_ROOT
from irbench.runtime.docker import DockerRuntime, wait_healthy
from irbench.runtime.hardware import load_hardware


def _root(tmp_path):
    compose = tmp_path / "engines" / "meilisearch" / "docker" / "compose.yaml"
    compose.parent.mkdir(parents=True)
    compose.write_text("services: {}\n")
    return tmp_path, compose


def _recorder(responses=None):
    calls = []

    def run(cmd):
        calls.append(cmd)
        line = " ".join(cmd)
        for prefix, out in (responses or {}).items():
            if line.startswith(prefix):
                if isinstance(out, Exception):
                    raise out
                return out
        return ""

    return run, calls


def test_hardware_profiles_load():
    assert load_hardware(REPO_ROOT, "local").engine.cpuset is None
    ref = load_hardware(REPO_ROOT, "aws-c7i-4xlarge")
    assert (ref.engine.cpuset, ref.engine.memory, ref.loadgen.cpuset) == ("0-7", "16g", "8-11")
    with pytest.raises(FileNotFoundError, match="nope"):
        load_hardware(REPO_ROOT, "nope")


def test_up_without_pinning_uses_only_the_engine_compose_file(tmp_path):
    root, compose = _root(tmp_path)
    run, calls = _recorder()
    rt = DockerRuntime(root, "meilisearch", load_hardware(REPO_ROOT, "local"), runner=run)
    rt.up()
    assert calls == [[
        "docker", "compose", "-p", "irbench-meilisearch", "-f", str(compose),
        "up", "-d", "--wait", "--wait-timeout", "300",
    ]]
    rt.down()
    assert calls[-1][-3:] == ["down", "-v", "--remove-orphans"]


def test_up_with_reference_profile_pins_cpu_and_memory(tmp_path):
    root, _ = _root(tmp_path)
    run, calls = _recorder()
    rt = DockerRuntime(root, "meilisearch", load_hardware(REPO_ROOT, "aws-c7i-4xlarge"), runner=run)
    rt.up()
    override = root / ".irbench" / "runtime" / "meilisearch.override.yaml"
    assert yaml.safe_load(override.read_text()) == {
        "services": {"engine": {"cpuset": "0-7", "mem_limit": "16g", "memswap_limit": "16g"}}
    }
    assert ["-f", str(override)] == calls[0][6:8]


def test_missing_compose_file_is_reported(tmp_path):
    rt = DockerRuntime(tmp_path, "nope", load_hardware(REPO_ROOT, "local"), runner=_recorder()[0])
    with pytest.raises(FileNotFoundError, match="nope"):
        rt.up()


def test_inspection_helpers(tmp_path):
    root, _ = _root(tmp_path)
    run, _ = _recorder({
        "docker inspect": "getmeili/meilisearch:v1.54.3\n",
        "docker image inspect": '["getmeili/meilisearch@sha256:abc"]\n',
        "docker exec irbench-meilisearch cat": "123456\n",
        "docker exec irbench-meilisearch du": "2048\t/meili_data\n",
    })
    rt = DockerRuntime(root, "meilisearch", load_hardware(REPO_ROOT, "local"), runner=run)
    assert rt.image_info() == {
        "image": "getmeili/meilisearch:v1.54.3", "digest": "getmeili/meilisearch@sha256:abc",
    }
    assert rt.memory_peak_bytes() == 123456
    assert rt.disk_bytes("/meili_data") == 2048 * 1024


def test_inspection_failures_return_none(tmp_path):
    root, _ = _root(tmp_path)
    run, _ = _recorder({"docker": RuntimeError("no such container")})
    rt = DockerRuntime(root, "meilisearch", load_hardware(REPO_ROOT, "local"), runner=run)
    assert rt.image_info() == {"image": None, "digest": None}
    assert rt.memory_peak_bytes() is None
    assert rt.disk_bytes("/x") is None


def test_wait_healthy_retries_until_200():
    statuses = iter([503, 503, 200])
    client = httpx.Client(
        base_url="http://e",
        transport=httpx.MockTransport(lambda r: httpx.Response(next(statuses))),
    )
    wait_healthy(client, "/health", timeout_s=5, interval_s=0)
    down = httpx.Client(
        base_url="http://e", transport=httpx.MockTransport(lambda r: httpx.Response(503))
    )
    with pytest.raises(TimeoutError, match="/health"):
        wait_healthy(down, "/health", timeout_s=0, interval_s=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest harness/tests/test_runtime.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'irbench.runtime'`.

- [ ] **Step 3: Implement**

`hardware/local.yaml`:

```yaml
id: local
description: >-
  Developer machine. No CPU or memory pinning; results are not comparable
  with the reference profile and are never published.
engine: {cpuset: null, memory: null}
loadgen: {cpuset: null}
```

`hardware/aws-c7i-4xlarge.yaml`:

```yaml
id: aws-c7i-4xlarge
description: >-
  AWS c7i.4xlarge (16 vCPU, 32 GB), gp3 SSD. Engine container pinned to
  8 vCPU / 16 GB; loadgen on a disjoint 4-vCPU set.
engine: {cpuset: "0-7", memory: "16g"}
loadgen: {cpuset: "8-11"}
```

`harness/irbench/runtime/hardware.py`:

```python
from pathlib import Path

import yaml
from pydantic import BaseModel


class EngineLimits(BaseModel):
    cpuset: str | None = None
    memory: str | None = None


class LoadgenLimits(BaseModel):
    cpuset: str | None = None


class HardwareProfile(BaseModel):
    id: str
    description: str = ""
    engine: EngineLimits = EngineLimits()
    loadgen: LoadgenLimits = LoadgenLimits()


def load_hardware(root: Path, profile_id: str) -> HardwareProfile:
    path = root / "hardware" / f"{profile_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: unknown hardware profile {profile_id!r}")
    profile = HardwareProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if profile.id != profile_id:
        raise ValueError(f"{path}: id {profile.id!r} does not match {profile_id!r}")
    return profile
```

`harness/irbench/runtime/docker.py`:

```python
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import httpx
import yaml

from irbench.engines.base import poll
from irbench.paths import Workspace
from irbench.runtime.hardware import HardwareProfile

Runner = Callable[[list[str]], str]


class Runtime(Protocol):
    def up(self) -> None: ...
    def down(self) -> None: ...
    def image_info(self) -> dict[str, str | None]: ...
    def memory_peak_bytes(self) -> int | None: ...
    def disk_bytes(self, path: str) -> int | None: ...


def run_command(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"{cmd[0]} not found: is Docker installed and on PATH?") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"{' '.join(cmd)} failed ({e.returncode}): {e.stderr.strip()[-2000:]}"
        ) from e
    return out.stdout


class DockerRuntime:
    """One engine container, managed through its compose file."""

    def __init__(
        self,
        root: Path,
        engine: str,
        profile: HardwareProfile,
        runner: Runner = run_command,
        state_dir: Path | None = None,
    ):
        self.engine = engine
        self.profile = profile
        self.runner = runner
        self.compose_file = root / "engines" / engine / "docker" / "compose.yaml"
        self.project = f"irbench-{engine}"
        self.container = f"irbench-{engine}"
        self.override_file = (state_dir or Workspace(root).cache / "runtime") / (
            f"{engine}.override.yaml"
        )

    def _compose(self, *args: str) -> list[str]:
        cmd = ["docker", "compose", "-p", self.project, "-f", str(self.compose_file)]
        if self.override_file.exists():
            cmd += ["-f", str(self.override_file)]
        return cmd + list(args)

    def _write_override(self) -> None:
        limits = self.profile.engine
        service: dict[str, str] = {}
        if limits.cpuset:
            service["cpuset"] = limits.cpuset
        if limits.memory:
            service["mem_limit"] = limits.memory
            service["memswap_limit"] = limits.memory  # no swap: memory limit is a hard limit
        if service:
            self.override_file.parent.mkdir(parents=True, exist_ok=True)
            self.override_file.write_text(yaml.safe_dump({"services": {"engine": service}}))
        else:
            self.override_file.unlink(missing_ok=True)

    def up(self, timeout_s: int = 300) -> None:
        if not self.compose_file.exists():
            raise FileNotFoundError(f"{self.compose_file} missing: unknown engine {self.engine!r}")
        self._write_override()
        self.runner(self._compose("up", "-d", "--wait", "--wait-timeout", str(timeout_s)))

    def down(self) -> None:
        self.runner(self._compose("down", "-v", "--remove-orphans"))

    def image_info(self) -> dict[str, str | None]:
        try:
            image = self.runner(
                ["docker", "inspect", "--format", "{{.Config.Image}}", self.container]
            ).strip()
            digests = json.loads(
                self.runner(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", image])
            )
        except (RuntimeError, ValueError):
            return {"image": None, "digest": None}
        return {"image": image, "digest": digests[0] if digests else None}

    def memory_peak_bytes(self) -> int | None:
        # cgroup v2: peak memory of the container since it started.
        try:
            out = self.runner(["docker", "exec", self.container, "cat", "/sys/fs/cgroup/memory.peak"])
            return int(out.strip())
        except (RuntimeError, ValueError):
            return None

    def disk_bytes(self, path: str) -> int | None:
        try:
            out = self.runner(["docker", "exec", self.container, "du", "-sk", path])
            return int(out.split()[0]) * 1024
        except (RuntimeError, ValueError, IndexError):
            return None


def wait_healthy(
    client: httpx.Client, path: str, timeout_s: float, interval_s: float = 1.0
) -> None:
    def ok() -> bool:
        try:
            return client.get(path).status_code == 200
        except httpx.HTTPError:
            return False

    poll(ok, timeout_s, f"{client.base_url}{path} to answer 200", interval_s)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest harness/tests/test_runtime.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add hardware harness/irbench/runtime harness/tests/test_runtime.py
git commit -m "feat: hardware profiles and Docker compose runtime"
```

---

### Task 5: Per-task filter, order and facet checks

**Files:**
- Create: `harness/irbench/score/checks.py`
- Test: `harness/tests/test_checks.py`

**Interfaces:**
- Consumes: `filter_violations`, `order_violations`, `expected_facets`, `facet_accuracy` (Plan 1 `score/constraints.py`); `SearchResult` (Task 3).
- Produces:
  - `TaskCheck` (pydantic): `task_id, mode, base, error: str | None, filter_violations: int | None, order_violations: int | None, facet_accuracy: float | None, facet_verifiable: bool | None`.
  - `check_task(task, result: SearchResult, check_ids: list[str] | None, docs_by_id, verify_limit: int) -> TaskCheck` — facets verifiable iff `check_ids is not None and len(unique check_ids) < verify_limit`; accuracy is the mean over the task's facet fields.
  - `CheckSummary` (pydantic): `mode, base, n, errors, filter_violations, order_violations, facet_accuracy, facet_verified, facet_unverifiable` (the last five `None` when the mode has no such check).
  - `summarize_checks(checks: list[TaskCheck]) -> list[CheckSummary]` sorted by (mode, base); errored tasks count in `errors` only.

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_checks.py`

```python
import pytest

from irbench.engines.base import SearchResult
from irbench.models import Filter, QueryTask, Sort
from irbench.score.checks import TaskCheck, check_task, summarize_checks

DOCS = {
    "a": {"product_class": ["Beds"], "average_rating": 4.5},
    "b": {"product_class": ["Sofas"], "average_rating": 3.0},
    "c": {"product_class": ["Beds"], "average_rating": None},
}


def _task(mode, **extra):
    return QueryTask(task_id=f"1:{mode}:keyword", qid="1", variant_of="1",
                     perturbation="original", text="x", mode=mode, base="keyword", **extra)


def test_plain_task_has_no_checks():
    c = check_task(_task("keyword"), SearchResult(["a"]), None, DOCS, 10)
    assert (c.filter_violations, c.order_violations, c.facet_accuracy, c.facet_verifiable) == (
        None, None, None, None,
    )


def test_filter_and_order_violations():
    f = _task("filtered", filter=Filter(field="product_class", op="eq", value="Beds"))
    assert check_task(f, SearchResult(["a", "b", "zzz"]), None, DOCS, 10).filter_violations == 2
    s = _task("sorted", sort=Sort(field="average_rating", direction="desc"))
    assert check_task(s, SearchResult(["a", "b", "c"]), None, DOCS, 10).order_violations == 0
    assert check_task(s, SearchResult(["c", "a"]), None, DOCS, 10).order_violations == 1


def test_facet_accuracy_against_verification_fetch():
    t = _task("faceted", facets=["product_class"])
    good = SearchResult(["a"], facets={"product_class": {"Beds": 2, "Sofas": 1}})
    c = check_task(t, good, ["a", "b", "c"], DOCS, 10)
    assert (c.facet_verifiable, c.facet_accuracy) == (True, 1.0)
    wrong = SearchResult(["a"], facets={"product_class": {"Beds": 5, "Sofas": 1}})
    assert check_task(t, wrong, ["a", "b", "c"], DOCS, 10).facet_accuracy == pytest.approx(0.5)
    missing = SearchResult(["a"], facets=None)
    assert check_task(t, missing, ["a", "b", "c"], DOCS, 10).facet_accuracy == 0.0


def test_facet_check_at_limit_is_unverifiable():
    t = _task("faceted", facets=["product_class"])
    c = check_task(t, SearchResult(["a"], facets={}), ["a", "b", "c"], DOCS, verify_limit=3)
    assert (c.facet_verifiable, c.facet_accuracy) == (False, None)
    failed_fetch = check_task(t, SearchResult(["a"], facets={}), None, DOCS, 10)
    assert failed_fetch.facet_verifiable is False


def test_duplicate_check_ids_count_once():
    t = _task("faceted", facets=["product_class"])
    result = SearchResult(["a"], facets={"product_class": {"Beds": 1}})
    assert check_task(t, result, ["a", "a", "a"], DOCS, 10).facet_accuracy == 1.0


def test_summaries_by_mode_and_base():
    checks = [
        TaskCheck(task_id="1", mode="filtered", base="keyword", filter_violations=0),
        TaskCheck(task_id="2", mode="filtered", base="keyword", filter_violations=3),
        TaskCheck(task_id="3", mode="filtered", base="keyword", error="boom"),
        TaskCheck(task_id="4", mode="faceted", base="keyword", facet_verifiable=True, facet_accuracy=1.0),
        TaskCheck(task_id="5", mode="faceted", base="keyword", facet_verifiable=True, facet_accuracy=0.5),
        TaskCheck(task_id="6", mode="faceted", base="keyword", facet_verifiable=False),
        TaskCheck(task_id="7", mode="keyword", base="keyword"),
    ]
    by_key = {(s.mode, s.base): s for s in summarize_checks(checks)}
    filtered = by_key[("filtered", "keyword")]
    assert (filtered.n, filtered.errors, filtered.filter_violations) == (3, 1, 3)
    assert filtered.order_violations is None
    faceted = by_key[("faceted", "keyword")]
    assert (faceted.facet_verified, faceted.facet_unverifiable) == (2, 1)
    assert faceted.facet_accuracy == pytest.approx(0.75)
    plain = by_key[("keyword", "keyword")]
    assert (plain.filter_violations, plain.facet_accuracy) == (None, None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest harness/tests/test_checks.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'irbench.score.checks'`.

- [ ] **Step 3: Implement** — `harness/irbench/score/checks.py`

```python
from collections import defaultdict

from pydantic import BaseModel

from irbench.engines.base import SearchResult
from irbench.models import QueryTask
from irbench.score.constraints import (
    Docs,
    expected_facets,
    facet_accuracy,
    filter_violations,
    order_violations,
)


class TaskCheck(BaseModel):
    task_id: str
    mode: str
    base: str
    error: str | None = None
    filter_violations: int | None = None
    order_violations: int | None = None
    facet_accuracy: float | None = None
    facet_verifiable: bool | None = None


class CheckSummary(BaseModel):
    mode: str
    base: str
    n: int
    errors: int
    filter_violations: int | None = None
    order_violations: int | None = None
    facet_accuracy: float | None = None
    facet_verified: int | None = None
    facet_unverifiable: int | None = None


def check_task(
    task: QueryTask,
    result: SearchResult,
    check_ids: list[str] | None,
    docs_by_id: Docs,
    verify_limit: int,
) -> TaskCheck:
    check = TaskCheck(task_id=task.task_id, mode=task.mode, base=task.base)
    if task.filter is not None:
        check.filter_violations = filter_violations(result.ids, task.filter, docs_by_id)
    if task.sort is not None:
        check.order_violations = order_violations(result.ids, task.sort, docs_by_id)
    if task.facets:
        unique = list(dict.fromkeys(check_ids)) if check_ids is not None else None
        check.facet_verifiable = unique is not None and len(unique) < verify_limit
        if check.facet_verifiable:
            reported = result.facets or {}
            scores = [
                facet_accuracy(
                    {str(k): int(v) for k, v in reported.get(name, {}).items()},
                    expected_facets(unique, name, docs_by_id),
                )
                for name in task.facets
            ]
            check.facet_accuracy = sum(scores) / len(scores)
    return check


def summarize_checks(checks: list[TaskCheck]) -> list[CheckSummary]:
    groups: dict[tuple[str, str], list[TaskCheck]] = defaultdict(list)
    for c in checks:
        groups[(c.mode, c.base)].append(c)
    summaries = []
    for (mode, base), items in sorted(groups.items()):
        ok = [c for c in items if c.error is None]
        s = CheckSummary(mode=mode, base=base, n=len(items), errors=len(items) - len(ok))
        fv = [c.filter_violations for c in ok if c.filter_violations is not None]
        if fv:
            s.filter_violations = sum(fv)
        ov = [c.order_violations for c in ok if c.order_violations is not None]
        if ov:
            s.order_violations = sum(ov)
        faceted = [c for c in ok if c.facet_verifiable is not None]
        if faceted:
            verified = [c.facet_accuracy for c in faceted if c.facet_verifiable]
            s.facet_verified = len(verified)
            s.facet_unverifiable = len(faceted) - len(verified)
            s.facet_accuracy = sum(verified) / len(verified) if verified else None
        summaries.append(s)
    return summaries
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest harness/tests/test_checks.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add harness/irbench/score/checks.py harness/tests/test_checks.py
git commit -m "feat: per-task filter, order and facet checks with summaries"
```

---

### Task 6: Run context, fake engine for tests, and the render stage

**Files:**
- Create: `harness/irbench/runs.py`, `harness/irbench/stages/context.py`, `harness/irbench/stages/render.py`, `harness/tests/fake_engine.py`
- Modify: `pyproject.toml` (first-party test helpers, docker marker)
- Test: `harness/tests/test_render_stage.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3; Plan 1 `read_vectors`, `read_jsonl`, `read_qrels`, `load_term_groups`, `prepare`, `embed_scenario`.
- Produces:
  - `irbench.runs.STAGES = ("prepare", "embed", "up", "index", "render", "relevance", "score", "down")`; `RunSpec(scenario, engine, model, track="default", hardware="local", host="localhost")` frozen pydantic model with `.run_id` = `"<slug(scenario)>__<engine>__<slug(model)>__<track>__<hardware>"`.
  - `irbench.stages.context.ClientFactory = Callable[[str], httpx.Client]`; `default_client(base_url) -> httpx.Client` (timeout 60 s); `RunContext(root, spec, client_factory=None)` with lazily loaded `scenario`, `config`, `track`, `adapter_cls`, `docs`, `docs_by_id`, `doc_vectors` (ids checked against `docs.jsonl`), `query_vectors`, `tasks`, `qrels`, `adapter`; methods `base_url`, `client()`, `query_vector(task) -> list[float] | None`, `rendered_path(task, check=False) -> Path`; attributes `run_dir`, `rendered_dir`.
  - Run-dir layout: `rendered/<mode>/<base>/<qid>.json` (+ `<qid>.check.json` for faceted tasks), `na.json` = `{task_id: reason}`.
  - `render_stage(ctx) -> RenderResult(n_rendered, n_unsupported)`.
  - Test helpers in `fake_engine.py`: `FakeAdapter`, `FakeServer`, `FakeRuntime`, `fake_root(tmp_path, wands_raw, config=None)`, `set_scenario(root, **changes)`, `FAKE_SPEC`, `perfect_answers(ctx)`.

- [ ] **Step 1: Register the test helpers and the docker marker in `pyproject.toml`**

```toml
[tool.pytest.ini_options]
testpaths = ["harness/tests"]
markers = [
  "slow: needs network or large models",
  "docker: starts real engine containers (run with -m docker)",
]
addopts = "-m 'not slow and not docker'"
```

```toml
[tool.ruff.lint.isort]
# Test helpers are imported as top-level modules (pytest rootdir import mode).
known-first-party = [
  "irbench", "conftest", "test_prepare", "test_embed_cache",
  "fake_engine", "adapter_helpers", "contract_fixture",
]
```

- [ ] **Step 2: Write the fake engine** — `harness/tests/fake_engine.py`

```python
"""An in-memory engine behind httpx.MockTransport, for stage and pipeline tests."""

import json
import shutil
from typing import Any

import httpx
import yaml

from conftest import REPO_ROOT
from irbench.engines.base import EngineAdapter, Request, SearchResult, call, poll
from irbench.runs import RunSpec
from irbench.score.constraints import expected_facets
from irbench.stages.embed import embed_scenario
from irbench.stages.prepare import prepare
from test_embed_cache import FakeEmbedder
from test_prepare import fixture_client, make_root

SCENARIO = "ecommerce/product-search"
FAKE_SPEC = RunSpec(scenario=SCENARIO, engine="fake", model="fake")
FAKE_CONFIG = {"default": {"unsupported": {"hybrid": "fake engine has no hybrid"}}}


class FakeAdapter(EngineAdapter):
    name = "fake"
    host_port = 9999
    health_path = "/health"
    data_dir = "/data"

    def create_index(self, client):
        call(client, Request("PUT", f"/indexes/{self.ctx.index}"))

    def load_documents(self, client, docs, vectors):
        for start in range(0, len(docs), 2):
            ids = [d["id"] for d in docs[start : start + 2]]
            call(client, Request("POST", f"/indexes/{self.ctx.index}/docs", body={"ids": ids}))

    def wait_until_indexed(self, client, n_docs, timeout_s):
        def ready():
            return call(client, Request("GET", f"/indexes/{self.ctx.index}/count"))["count"] == n_docs

        poll(ready, timeout_s, "fake indexing", interval_s=0)

    def render(self, task, vector, limit, facets=True):
        body = {
            "task_id": task.task_id,
            "limit": limit,
            "facets": facets and bool(task.facets),
            "check": bool(task.facets) and not facets,
            "has_vector": vector is not None,
            **self.search_params(task),
        }
        return Request("POST", f"/indexes/{self.ctx.index}/search", body=body)

    def parse(self, task, payload):
        return SearchResult(ids=payload["ids"], facets=payload.get("facets"))


class FakeServer:
    """Answers searches from a {task_id: [doc ids]} table."""

    def __init__(self, answers=None, facets=None, fail=(), slow=(), fail_checks=False,
                 fail_index=False, drop_docs=False):
        self.answers: dict[str, list[str]] = answers or {}
        self.facets: dict[str, dict[str, dict[str, int]]] = facets or {}
        self.fail, self.slow = set(fail), set(slow)
        self.fail_checks, self.fail_index, self.drop_docs = fail_checks, fail_index, drop_docs
        self.docs: set[str] = set()

    def client_factory(self, base_url: str) -> httpx.Client:
        return httpx.Client(base_url=base_url, transport=httpx.MockTransport(self))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body: Any = json.loads(request.content) if request.content else None
        if path == "/health":
            return httpx.Response(200)
        if request.method == "PUT":
            if self.fail_index:
                return httpx.Response(500, text="index creation failed")
            self.docs = set()
            return httpx.Response(200, json={})
        if path.endswith("/docs"):
            if not self.drop_docs:
                self.docs.update(body["ids"])
            return httpx.Response(200, json={})
        if path.endswith("/count"):
            return httpx.Response(200, json={"count": len(self.docs)})
        if path.endswith("/search"):
            tid = body["task_id"]
            if tid in self.slow:
                raise httpx.ReadTimeout("engine too slow", request=request)
            if tid in self.fail or (body["check"] and self.fail_checks):
                return httpx.Response(500, text="search failed")
            out: dict[str, Any] = {"ids": self.answers.get(tid, [])[: body["limit"]]}
            if body["facets"]:
                out["facets"] = self.facets.get(tid, {})
            return httpx.Response(200, json=out)
        return httpx.Response(404)


class FakeRuntime:
    def __init__(self):
        self.calls: list[str] = []

    def up(self):
        self.calls.append("up")

    def down(self):
        self.calls.append("down")

    def image_info(self):
        return {"image": "fake:1", "digest": "fake@sha256:0"}

    def memory_peak_bytes(self):
        return 1024

    def disk_bytes(self, path):
        return 2048


def fake_root(tmp_path, wands_raw, config=None):
    """Mini WANDS prepared and embedded with the fake model, plus an `engines/fake` folder."""
    root = make_root(tmp_path, wands_raw)
    prepare(root, SCENARIO, client=fixture_client(wands_raw))
    embed_scenario(root, SCENARIO, "fake", embedder=FakeEmbedder())
    engine_dir = root / "engines" / "fake"
    (engine_dir / "configs" / "ecommerce").mkdir(parents=True)
    (engine_dir / "adapter.py").write_text("from fake_engine import FakeAdapter as Adapter  # noqa: F401\n")
    (engine_dir / "configs" / "ecommerce" / "product-search.yaml").write_text(
        yaml.safe_dump(config or FAKE_CONFIG)
    )
    shutil.copytree(REPO_ROOT / "hardware", root / "hardware")
    return root


def set_scenario(root, **changes):
    path = root / "scenarios" / SCENARIO / "scenario.yaml"
    raw = yaml.safe_load(path.read_text())
    raw.update(changes)
    path.write_text(yaml.safe_dump(raw))


def perfect_answers(ctx):
    """Every supported task answered with its relevant docs, best grade first."""
    answers, facets = {}, {}
    for task in ctx.tasks:
        grades = ctx.qrels[task.task_id]
        ids = [d for d, g in sorted(grades.items(), key=lambda kv: (-kv[1], kv[0])) if g > 0]
        answers[task.task_id] = ids
        if task.facets:
            facets[task.task_id] = {
                name: expected_facets(ids, name, ctx.docs_by_id) for name in task.facets
            }
    return answers, facets
```

- [ ] **Step 3: Write the failing tests** — `harness/tests/test_render_stage.py`

```python
import numpy as np
import pytest

from fake_engine import FAKE_SPEC, fake_root
from irbench.embed.vectors import read_vectors, write_vectors
from irbench.engines.base import Request
from irbench.io import read_json
from irbench.paths import Workspace
from irbench.runs import RunSpec
from irbench.stages.context import RunContext
from irbench.stages.render import render_stage


def test_run_id_names_every_axis():
    spec = RunSpec(scenario="ecommerce/product-search", engine="meilisearch",
                   model="org/bge", track="tuned", hardware="aws-c7i-4xlarge")
    assert spec.run_id == "ecommerce__product-search__meilisearch__org__bge__tuned__aws-c7i-4xlarge"


def test_render_writes_one_request_per_supported_task(tmp_path, wands_raw):
    ctx = RunContext(fake_root(tmp_path, wands_raw), FAKE_SPEC)
    result = render_stage(ctx)

    supported = [t for t in ctx.tasks if t.mode != "hybrid"]
    assert result.n_rendered == len(supported)
    assert result.n_unsupported == len(ctx.tasks) - len(supported) > 0
    na = read_json(ctx.run_dir / "na.json")
    assert set(na) == {t.task_id for t in ctx.tasks if t.mode == "hybrid"}
    assert set(na.values()) == {"fake engine has no hybrid"}

    keyword = next(t for t in supported if t.mode == "keyword")
    semantic = next(t for t in supported if t.mode == "semantic")
    assert Request.from_json(read_json(ctx.rendered_path(keyword))).body["has_vector"] is False
    assert Request.from_json(read_json(ctx.rendered_path(semantic))).body["has_vector"] is True

    faceted = next(t for t in supported if t.mode == "faceted")
    main = Request.from_json(read_json(ctx.rendered_path(faceted))).body
    check = Request.from_json(read_json(ctx.rendered_path(faceted, check=True))).body
    assert (main["facets"], main["limit"]) == (True, 100)
    assert (check["facets"], check["check"], check["limit"]) == (False, True, 1000)
    assert not ctx.rendered_path(keyword, check=True).exists()


def test_render_clears_previous_requests(tmp_path, wands_raw):
    ctx = RunContext(fake_root(tmp_path, wands_raw), FAKE_SPEC)
    stale = ctx.rendered_dir / "keyword" / "keyword" / "stale.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    render_stage(ctx)
    assert not stale.exists()


def test_stale_doc_vectors_are_rejected(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    path = Workspace(root).doc_vectors_path("wands", "fake")
    ids, vectors = read_vectors(path)
    write_vectors(path, ids[::-1], np.asarray(vectors[::-1]))
    with pytest.raises(ValueError, match="does not match docs.jsonl"):
        RunContext(root, FAKE_SPEC).doc_vectors
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest harness/tests/test_render_stage.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'irbench.runs'`.

- [ ] **Step 5: Implement**

`harness/irbench/runs.py`:

```python
from pydantic import BaseModel, ConfigDict

from irbench.paths import slug

STAGES = ("prepare", "embed", "up", "index", "render", "relevance", "score", "down")


class RunSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenario: str
    engine: str
    model: str
    track: str = "default"
    hardware: str = "local"
    host: str = "localhost"

    @property
    def run_id(self) -> str:
        return "__".join(
            [slug(self.scenario), self.engine, slug(self.model), self.track, self.hardware]
        )
```

`harness/irbench/stages/context.py`:

```python
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from irbench.embed.vectors import read_vectors
from irbench.engines.base import EngineAdapter, IndexContext, index_name, load_adapter_class
from irbench.engines.config import EngineConfig, TrackConfig, load_engine_config
from irbench.io import read_jsonl, read_qrels
from irbench.models import Qrels, QueryTask
from irbench.paths import Workspace
from irbench.runs import RunSpec
from irbench.scenario import Scenario, load_scenario
from irbench.terms import load_term_groups

ClientFactory = Callable[[str], httpx.Client]


def default_client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=60.0)


class RunContext:
    """Inputs of one run, loaded on first use so early stages can create them."""

    def __init__(self, root: Path, spec: RunSpec, client_factory: ClientFactory | None = None):
        self.root = root
        self.spec = spec
        self.ws = Workspace(root)
        self.run_dir = self.ws.run_dir(spec.run_id)
        self.rendered_dir = self.run_dir / "rendered"
        self.client_factory = client_factory or default_client

    @cached_property
    def scenario(self) -> Scenario:
        return load_scenario(self.root, self.spec.scenario)

    @cached_property
    def config(self) -> EngineConfig:
        return load_engine_config(self.root, self.spec.engine, self.spec.scenario)

    @cached_property
    def track(self) -> TrackConfig:
        return self.config.track(self.spec.track)

    @cached_property
    def adapter_cls(self) -> type[EngineAdapter]:
        return load_adapter_class(self.root, self.spec.engine)

    @cached_property
    def docs(self) -> list[dict[str, Any]]:
        return list(read_jsonl(self.ws.dataset_dir(self.scenario.dataset) / "docs.jsonl"))

    @cached_property
    def docs_by_id(self) -> dict[str, dict[str, Any]]:
        return {d[self.scenario.id_field]: d for d in self.docs}

    def _embed_hint(self) -> str:
        return f"run `irbench embed --scenario {self.spec.scenario} --model {self.spec.model}`"

    @cached_property
    def doc_vectors(self) -> np.ndarray:
        path = self.ws.doc_vectors_path(self.scenario.dataset, self.spec.model)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: {self._embed_hint()}")
        ids, vectors = read_vectors(path)
        doc_ids = [d[self.scenario.id_field] for d in self.docs]
        if ids != doc_ids:
            raise ValueError(
                f"{path} does not match docs.jsonl ({len(ids)} vs {len(doc_ids)} ids): "
                f"{self._embed_hint()}"
            )
        return vectors

    @cached_property
    def query_vectors(self) -> dict[str, np.ndarray]:
        path = self.ws.query_vectors_path(self.spec.scenario, self.spec.model)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: {self._embed_hint()}")
        ids, vectors = read_vectors(path)
        return dict(zip(ids, vectors))

    @cached_property
    def tasks(self) -> list[QueryTask]:
        path = self.ws.scenario_dir(self.spec.scenario) / "tasks.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing: run `irbench prepare --scenario {self.spec.scenario}`"
            )
        return [QueryTask.model_validate(r) for r in read_jsonl(path)]

    @cached_property
    def qrels(self) -> Qrels:
        return read_qrels(self.ws.scenario_dir(self.spec.scenario) / "qrels.tasks.tsv")

    @cached_property
    def adapter(self) -> EngineAdapter:
        return self.adapter_cls(
            IndexContext(
                scenario=self.scenario,
                config=self.track,
                index=index_name(self.spec.scenario, self.spec.model),
                dim=int(self.doc_vectors.shape[1]),
                synonyms=load_term_groups(self.scenario.synonyms_path),
            )
        )

    @property
    def base_url(self) -> str:
        return f"http://{self.spec.host}:{self.adapter_cls.host_port}"

    def client(self) -> httpx.Client:
        return self.client_factory(self.base_url)

    def query_vector(self, task: QueryTask) -> list[float] | None:
        if task.base == "keyword":
            return None
        if task.qid not in self.query_vectors:
            raise ValueError(f"no query vector for {task.qid!r}: {self._embed_hint()}")
        return self.query_vectors[task.qid].tolist()

    def rendered_path(self, task: QueryTask, check: bool = False) -> Path:
        name = f"{task.qid}.check.json" if check else f"{task.qid}.json"
        return self.rendered_dir / task.mode / task.base / name
```

`harness/irbench/stages/render.py`:

```python
import shutil
from dataclasses import dataclass

from irbench.io import write_json
from irbench.stages.context import RunContext


@dataclass(frozen=True)
class RenderResult:
    n_rendered: int
    n_unsupported: int


def render_stage(ctx: RunContext) -> RenderResult:
    """Write every supported task as a request file; unsupported ones go to na.json."""
    if ctx.rendered_dir.exists():
        shutil.rmtree(ctx.rendered_dir)
    adapter, scenario = ctx.adapter, ctx.scenario
    unsupported: dict[str, str] = {}
    rendered = 0
    for task in ctx.tasks:
        reason = adapter.unsupported_reason(task)
        if reason:
            unsupported[task.task_id] = reason
            continue
        vector = ctx.query_vector(task)
        request = adapter.render(task, vector, scenario.top_k)
        write_json(ctx.rendered_path(task), request.to_json(), indent=None)
        if task.facets:
            check = adapter.render(task, vector, scenario.facet_verify_limit, facets=False)
            write_json(ctx.rendered_path(task, check=True), check.to_json(), indent=None)
        rendered += 1
    write_json(ctx.run_dir / "na.json", unsupported)
    return RenderResult(n_rendered=rendered, n_unsupported=len(unsupported))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest harness/tests/test_render_stage.py -q && uv run ruff check .`
Expected: 4 passed; ruff clean.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml harness/irbench/runs.py harness/irbench/stages/context.py harness/irbench/stages/render.py harness/tests/fake_engine.py harness/tests/test_render_stage.py
git commit -m "feat: run context and render stage writing inspectable request files"
```

---

### Task 7: Index and relevance stages

**Files:**
- Create: `harness/irbench/stages/index.py`, `harness/irbench/stages/relevance.py`
- Test: `harness/tests/test_index_relevance.py`

**Interfaces:**
- Consumes: `RunContext` (Task 6), `Runtime`, `wait_healthy` (Task 4), `Request`, `call`, `EngineError`, `SearchResult` (Task 3), `check_task`, `TaskCheck` (Task 5), Plan 1 `write_run`, `write_jsonl`.
- Produces:
  - `index_stage(ctx, runtime) -> IndexStats(seconds, n_docs, memory_peak_bytes, disk_bytes)`; writes `index.json`. Indexing time = create + load + wait.
  - `run_request(client, adapter, task, request, timeout) -> SearchResult` (shared with the contract tests).
  - `relevance_stage(ctx) -> RelevanceResult(n_tasks, n_errors)`; writes `run.trec` (top_k per task, tag = run id), `checks.jsonl` (one `TaskCheck` per executed task), `errors.jsonl` (`{"task_id", "request": "search"|"check", "error"}`).
  - Errors caught per request: `httpx.HTTPError`, `EngineError`, `ValueError`, `KeyError`, `TypeError`. A failed check fetch keeps the search result and marks facets unverifiable.

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_index_relevance.py`

```python
import pytest

from fake_engine import FAKE_SPEC, FakeRuntime, FakeServer, fake_root, perfect_answers, set_scenario
from irbench.io import read_json, read_jsonl, read_run
from irbench.stages.context import RunContext
from irbench.stages.index import index_stage
from irbench.stages.relevance import relevance_stage
from irbench.stages.render import render_stage


def _ctx(root, server):
    return RunContext(root, FAKE_SPEC, client_factory=server.client_factory)


def test_index_loads_all_docs_and_records_stats(tmp_path, wands_raw):
    server = FakeServer()
    ctx = _ctx(fake_root(tmp_path, wands_raw), server)
    stats = index_stage(ctx, FakeRuntime())
    assert stats.n_docs == 4 and server.docs == {"0", "1", "2", "3"}
    saved = read_json(ctx.run_dir / "index.json")
    assert (saved["memory_peak_bytes"], saved["disk_bytes"], saved["n_docs"]) == (1024, 2048, 4)


def test_index_times_out_when_engine_never_catches_up(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    set_scenario(root, index_timeout_s=0)
    with pytest.raises(TimeoutError, match="fake indexing"):
        index_stage(_ctx(root, FakeServer(drop_docs=True)), FakeRuntime())


def test_relevance_writes_run_checks_and_errors(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    ctx = _ctx(root, FakeServer())
    answers, facets = perfect_answers(ctx)
    failing = next(t.task_id for t in ctx.tasks if t.mode == "keyword")
    server = FakeServer(answers=answers, facets=facets, fail={failing})
    ctx = _ctx(root, server)
    render_stage(ctx)

    result = relevance_stage(ctx)

    supported = [t for t in ctx.tasks if t.mode != "hybrid"]
    assert (result.n_tasks, result.n_errors) == (len(supported), 1)
    run = read_run(ctx.run_dir / "run.trec")
    assert failing not in run
    ok = next(t.task_id for t in supported if t.task_id != failing and answers[t.task_id])
    assert run[ok] == answers[ok]
    errors = list(read_jsonl(ctx.run_dir / "errors.jsonl"))
    assert [(e["task_id"], e["request"]) for e in errors] == [(failing, "search")]
    assert "500" in errors[0]["error"]
    checks = {c["task_id"]: c for c in read_jsonl(ctx.run_dir / "checks.jsonl")}
    assert set(checks) == {t.task_id for t in supported}
    assert checks[failing]["error"]
    filtered = [c for c in checks.values() if c["mode"] == "filtered"]
    assert filtered and all(c["filter_violations"] == 0 for c in filtered)
    faceted = [c for c in checks.values() if c["mode"] == "faceted"]
    assert faceted and all(c["facet_accuracy"] == 1.0 for c in faceted)


def test_slow_engine_times_out_and_scores_empty(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    ctx = _ctx(root, FakeServer())
    slow = next(t.task_id for t in ctx.tasks if t.mode == "semantic")
    ctx = _ctx(root, FakeServer(answers=perfect_answers(ctx)[0], slow={slow}))
    render_stage(ctx)
    relevance_stage(ctx)
    assert slow not in read_run(ctx.run_dir / "run.trec")
    errors = list(read_jsonl(ctx.run_dir / "errors.jsonl"))
    assert errors[0]["task_id"] == slow and "ReadTimeout" in errors[0]["error"]


def test_failed_check_fetch_keeps_result_and_marks_facets_unverifiable(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    ctx = _ctx(root, FakeServer())
    answers, facets = perfect_answers(ctx)
    ctx = _ctx(root, FakeServer(answers=answers, facets=facets, fail_checks=True))
    render_stage(ctx)
    relevance_stage(ctx)
    run = read_run(ctx.run_dir / "run.trec")
    checks = [c for c in read_jsonl(ctx.run_dir / "checks.jsonl") if c["mode"] == "faceted"]
    assert checks and all(c["facet_verifiable"] is False and c["error"] is None for c in checks)
    assert all(run[c["task_id"]] == answers[c["task_id"]] for c in checks if answers[c["task_id"]])
    errors = list(read_jsonl(ctx.run_dir / "errors.jsonl"))
    assert errors and {e["request"] for e in errors} == {"check"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest harness/tests/test_index_relevance.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'irbench.stages.index'`.

- [ ] **Step 3: Implement**

`harness/irbench/stages/index.py`:

```python
import time
from dataclasses import asdict, dataclass

from irbench.io import write_json
from irbench.runtime.docker import Runtime, wait_healthy
from irbench.stages.context import RunContext

HEALTH_TIMEOUT_S = 180


@dataclass(frozen=True)
class IndexStats:
    seconds: float
    n_docs: int
    memory_peak_bytes: int | None
    disk_bytes: int | None


def index_stage(ctx: RunContext, runtime: Runtime) -> IndexStats:
    adapter, docs, vectors = ctx.adapter, ctx.docs, ctx.doc_vectors
    with ctx.client() as client:
        wait_healthy(client, adapter.health_path, HEALTH_TIMEOUT_S)
        start = time.perf_counter()
        adapter.create_index(client)
        adapter.load_documents(client, docs, vectors)
        adapter.wait_until_indexed(client, len(docs), ctx.scenario.index_timeout_s)
        seconds = time.perf_counter() - start
    stats = IndexStats(
        seconds=round(seconds, 3),
        n_docs=len(docs),
        memory_peak_bytes=runtime.memory_peak_bytes(),
        disk_bytes=runtime.disk_bytes(adapter.data_dir),
    )
    write_json(ctx.run_dir / "index.json", asdict(stats))
    return stats
```

`harness/irbench/stages/relevance.py`:

```python
import json
from dataclasses import dataclass
from pathlib import Path

import httpx

from irbench.engines.base import EngineAdapter, EngineError, Request, SearchResult, call
from irbench.io import write_jsonl, write_run
from irbench.models import QueryTask, RunResults
from irbench.score.checks import TaskCheck, check_task
from irbench.stages.context import RunContext

REQUEST_ERRORS = (httpx.HTTPError, EngineError, ValueError, KeyError, TypeError)


@dataclass(frozen=True)
class RelevanceResult:
    n_tasks: int
    n_errors: int


def run_request(
    client: httpx.Client, adapter: EngineAdapter, task: QueryTask, request: Request, timeout: float
) -> SearchResult:
    return adapter.parse(task, call(client, request, timeout))


def _load(path: Path) -> Request:
    return Request.from_json(json.loads(path.read_text(encoding="utf-8")))


def _error(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


def relevance_stage(ctx: RunContext) -> RelevanceResult:
    """Execute each rendered request once, sequentially; score inputs go to the run dir."""
    scenario, adapter = ctx.scenario, ctx.adapter
    timeout = scenario.request_timeout_s
    results: RunResults = {}
    checks: list[TaskCheck] = []
    errors: list[dict[str, str]] = []

    with ctx.client() as client:
        for task in ctx.tasks:
            path = ctx.rendered_path(task)
            if not path.exists():  # unsupported: listed in na.json
                continue
            try:
                found = run_request(client, adapter, task, _load(path), timeout)
            except REQUEST_ERRORS as e:
                errors.append({"task_id": task.task_id, "request": "search", "error": _error(e)})
                checks.append(
                    TaskCheck(task_id=task.task_id, mode=task.mode, base=task.base, error=_error(e))
                )
                continue
            found = SearchResult(ids=found.ids[: scenario.top_k], facets=found.facets)
            check_ids = None
            if task.facets:
                try:
                    check_request = _load(ctx.rendered_path(task, check=True))
                    check_ids = run_request(client, adapter, task, check_request, timeout).ids
                except REQUEST_ERRORS as e:
                    errors.append({"task_id": task.task_id, "request": "check", "error": _error(e)})
            results[task.task_id] = found.ids
            checks.append(
                check_task(task, found, check_ids, ctx.docs_by_id, scenario.facet_verify_limit)
            )

    write_run(ctx.run_dir / "run.trec", {t: ids for t, ids in results.items() if ids}, ctx.spec.run_id)
    write_jsonl(ctx.run_dir / "checks.jsonl", (c.model_dump() for c in checks))
    write_jsonl(ctx.run_dir / "errors.jsonl", errors)
    n_failed = sum(1 for e in errors if e["request"] == "search")
    return RelevanceResult(n_tasks=len(checks), n_errors=n_failed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest harness/tests/test_index_relevance.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add harness/irbench/stages/index.py harness/irbench/stages/relevance.py harness/tests/test_index_relevance.py
git commit -m "feat: index and relevance stages with per-request error capture"
```

---

### Task 8: Run scoring, pipeline orchestrator, manifest and `irbench run`

**Files:**
- Create: `harness/irbench/stages/score_run.py`, `harness/irbench/stages/run.py`
- Modify: `harness/irbench/cli.py`
- Test: `harness/tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: Tasks 4–7; Plan 1 `score_relevance`, `write_manifest`, `prepare`, `embed_scenario`.
- Produces:
  - `score_stage(ctx) -> dict` writing `scores.json` = `{run_id, scenario, engine, model, track, relevance: [GroupScore…], checks: [CheckSummary…], not_applicable: {"<mode>:<base>": reason}, errors: int}`. Only supported tasks enter relevance scoring.
  - `run_pipeline(root, spec, stages, *, runtime_factory=None, client_factory=None, prepare_client=None, embedder=None) -> dict` — returns and writes `manifest.json` = `{run_id, scenario, engine, model, track, status, hardware, stages: {name: {status, started_at, seconds, result|error}}, engine_image, config: {path, hash, diff_vs_default}, dataset, seed}` plus `irbench_version`, `git_sha`, `created_at`. Re-running a subset of stages updates the existing manifest.
  - CLI: `irbench run --scenario --engine --model [--track default] [--hardware local] [--host localhost] [--stages prepare,embed,up,index,render,relevance,score,down]`; exit 1 when the run failed or a stage name is unknown.

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_run_pipeline.py`

```python
import pytest
from typer.testing import CliRunner

from fake_engine import (
    FAKE_CONFIG,
    FAKE_SPEC,
    SCENARIO,
    FakeRuntime,
    FakeServer,
    fake_root,
    perfect_answers,
)
from irbench.cli import app
from irbench.io import read_json
from irbench.runs import STAGES, RunSpec
from irbench.stages.context import RunContext
from irbench.stages.run import run_pipeline
from test_embed_cache import FakeEmbedder
from test_prepare import fixture_client


def _server(root, **kwargs):
    answers, facets = perfect_answers(RunContext(root, FAKE_SPEC))
    return FakeServer(answers=answers, facets=facets, **kwargs)


def test_pipeline_end_to_end_with_fake_engine(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    server, runtime = _server(root), FakeRuntime()

    manifest = run_pipeline(
        root, FAKE_SPEC, list(STAGES),
        runtime_factory=lambda *a: runtime, client_factory=server.client_factory,
        prepare_client=fixture_client(wands_raw), embedder=FakeEmbedder(),
    )

    assert manifest["status"] == "succeeded", manifest["stages"]
    assert list(manifest["stages"]) == list(STAGES)
    assert all(s["status"] == "succeeded" for s in manifest["stages"].values())
    assert runtime.calls == ["up", "down"]
    assert manifest["engine_image"] == {"image": "fake:1", "digest": "fake@sha256:0"}
    assert len(manifest["config"]["hash"]) == 64 and manifest["config"]["diff_vs_default"] == {}
    assert manifest["dataset"]["version"] == "fixture" and manifest["seed"] == 20261009
    assert manifest["hardware"]["id"] == "local"

    run_dir = RunContext(root, FAKE_SPEC).run_dir
    assert read_json(run_dir / "manifest.json")["status"] == "succeeded"
    scores = read_json(run_dir / "scores.json")
    assert scores["not_applicable"] == {"hybrid:hybrid": "fake engine has no hybrid"}
    assert all(g["mode"] != "hybrid" for g in scores["relevance"])
    overall = [g for g in scores["relevance"] if g["perturbation"] == "all"]
    assert overall and all(g["metrics"]["ndcg@10"] == 1.0 for g in overall)
    checks = {(c["mode"], c["base"]): c for c in scores["checks"]}
    assert checks[("filtered", "keyword")]["filter_violations"] == 0
    assert checks[("faceted", "keyword")]["facet_accuracy"] == 1.0
    assert scores["errors"] == 0


def test_failed_stage_records_reason_and_tears_down(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    runtime = FakeRuntime()
    manifest = run_pipeline(
        root, FAKE_SPEC, ["up", "index", "render"],
        runtime_factory=lambda *a: runtime,
        client_factory=FakeServer(fail_index=True).client_factory,
    )
    assert manifest["status"] == "failed"
    assert manifest["stages"]["index"]["status"] == "failed"
    assert "500" in manifest["stages"]["index"]["error"]
    assert "render" not in manifest["stages"]
    assert runtime.calls == ["up", "down"]
    assert manifest["stages"]["down"]["status"] == "succeeded"


def test_unknown_stage_is_rejected_before_anything_runs(tmp_path, wands_raw):
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="nope"):
        run_pipeline(fake_root(tmp_path, wands_raw), FAKE_SPEC, ["up", "nope"],
                     runtime_factory=lambda *a: runtime)
    assert runtime.calls == []


def test_tuned_track_records_its_diff(tmp_path, wands_raw):
    config = {**FAKE_CONFIG, "tuned": {"search": {"keyword": {"boost": 2}}}}
    root = fake_root(tmp_path, wands_raw, config=config)
    spec = RunSpec(scenario=SCENARIO, engine="fake", model="fake", track="tuned")
    manifest = run_pipeline(root, spec, ["render"], runtime_factory=lambda *a: FakeRuntime())
    assert manifest["status"] == "succeeded"
    assert manifest["run_id"].endswith("__tuned__local")
    assert manifest["config"]["diff_vs_default"] == {
        "search.keyword": {"default": None, "effective": {"boost": 2}}
    }


def test_cli_run_prints_scores_and_na(tmp_path, wands_raw, monkeypatch):
    root = fake_root(tmp_path, wands_raw)
    server = _server(root)
    monkeypatch.setattr("irbench.stages.run.DockerRuntime", lambda *a: FakeRuntime())
    monkeypatch.setattr("irbench.stages.run.default_client", server.client_factory)
    result = CliRunner().invoke(app, [
        "run", "--scenario", SCENARIO, "--engine", "fake", "--model", "fake",
        "--stages", "up,index,render,relevance,score,down", "--root", str(root),
    ])
    assert result.exit_code == 0, result.output
    assert "keyword/keyword: nDCG@10=1.0000" in result.stdout
    assert "hybrid/hybrid: N/A (fake engine has no hybrid)" in result.stdout


def test_cli_run_rejects_unknown_stage(tmp_path, wands_raw):
    result = CliRunner().invoke(app, [
        "run", "--scenario", SCENARIO, "--engine", "fake", "--model", "fake",
        "--stages", "bogus", "--root", str(fake_root(tmp_path, wands_raw)),
    ])
    assert result.exit_code == 1
    assert "bogus" in result.output
```

The `diff_vs_default` assertion follows `config_diff`: `search` is `{}` in default and `{"keyword": {...}}` when tuned, so the differing path is `search.keyword` (its default is absent, hence `None`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest harness/tests/test_run_pipeline.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'irbench.stages.run'`.

- [ ] **Step 3: Implement**

`harness/irbench/stages/score_run.py`:

```python
from typing import Any

from irbench.io import read_json, read_jsonl, read_run, write_json
from irbench.score.checks import TaskCheck, summarize_checks
from irbench.score.relevance import score_relevance
from irbench.stages.context import RunContext


def score_stage(ctx: RunContext) -> dict[str, Any]:
    run_dir = ctx.run_dir
    na: dict[str, str] = read_json(run_dir / "na.json")
    supported = [t for t in ctx.tasks if t.task_id not in na]
    run = read_run(run_dir / "run.trec")
    unknown = sorted(set(run) - {t.task_id for t in supported})
    if unknown:
        raise ValueError(f"run.trec has results for unknown or unsupported tasks: {unknown[:5]}")

    not_applicable: dict[str, str] = {}
    for task in ctx.tasks:
        if task.task_id in na:
            not_applicable.setdefault(f"{task.mode}:{task.base}", na[task.task_id])

    checks = [TaskCheck.model_validate(r) for r in read_jsonl(run_dir / "checks.jsonl")]
    report = {
        "run_id": ctx.spec.run_id,
        "scenario": ctx.spec.scenario,
        "engine": ctx.spec.engine,
        "model": ctx.spec.model,
        "track": ctx.spec.track,
        "relevance": [g.model_dump() for g in score_relevance(supported, ctx.qrels, run)],
        "checks": [s.model_dump() for s in summarize_checks(checks)],
        "not_applicable": not_applicable,
        "errors": sum(1 for e in read_jsonl(run_dir / "errors.jsonl") if e["request"] == "search"),
    }
    write_json(run_dir / "scores.json", report)
    return report
```

`harness/irbench/stages/run.py`:

```python
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from irbench.embed.base import Embedder
from irbench.io import read_json
from irbench.manifest import write_manifest
from irbench.runs import STAGES, RunSpec
from irbench.runtime.docker import DockerRuntime, Runtime
from irbench.runtime.hardware import HardwareProfile, load_hardware
from irbench.stages.context import ClientFactory, RunContext, default_client
from irbench.stages.embed import embed_scenario
from irbench.stages.index import index_stage
from irbench.stages.prepare import prepare
from irbench.stages.relevance import relevance_stage
from irbench.stages.render import render_stage
from irbench.stages.score_run import score_stage

RuntimeFactory = Callable[[Path, str, HardwareProfile], Runtime]
GENERATED_KEYS = ("irbench_version", "git_sha", "created_at")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def run_pipeline(
    root: Path,
    spec: RunSpec,
    stages: list[str],
    *,
    runtime_factory: RuntimeFactory | None = None,
    client_factory: ClientFactory | None = None,
    prepare_client: httpx.Client | None = None,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; known: {', '.join(STAGES)}")
    ordered = [s for s in STAGES if s in stages]

    ctx = RunContext(root, spec, client_factory or default_client)
    profile = load_hardware(root, spec.hardware)
    runtime = (runtime_factory or DockerRuntime)(root, spec.engine, profile)

    manifest_path = ctx.run_dir / "manifest.json"
    manifest: dict[str, Any] = read_json(manifest_path) if manifest_path.exists() else {}
    for key in GENERATED_KEYS:
        manifest.pop(key, None)
    manifest.update(
        run_id=spec.run_id, scenario=spec.scenario, engine=spec.engine, model=spec.model,
        track=spec.track, hardware=profile.model_dump(),
    )
    log: dict[str, Any] = manifest.setdefault("stages", {})

    def execute(stage: str) -> Any:
        if stage == "prepare":
            return {"tasks": prepare(root, spec.scenario, client=prepare_client).n_tasks}
        if stage == "embed":
            return {"computed": embed_scenario(root, spec.scenario, spec.model, embedder).n_computed}
        if stage == "up":
            runtime.up()
            manifest["engine_image"] = runtime.image_info()
            return None
        if stage == "index":
            return asdict(index_stage(ctx, runtime))
        if stage == "render":
            return asdict(render_stage(ctx))
        if stage == "relevance":
            return asdict(relevance_stage(ctx))
        if stage == "score":
            report = score_stage(ctx)
            return {"errors": report["errors"], "not_applicable": len(report["not_applicable"])}
        raise AssertionError(stage)

    def record(stage: str, fn: Callable[[], Any]) -> bool:
        started, t0 = _now(), time.perf_counter()
        entry: dict[str, Any] = {"started_at": started}
        try:
            result = fn()
        except Exception as e:  # spec §9: a failed stage is recorded, never a crash
            entry.update(status="failed", error=f"{type(e).__name__}: {e}")
            ok = False
        else:
            entry["status"] = "succeeded"
            if result is not None:
                entry["result"] = result
            ok = True
        entry["seconds"] = round(time.perf_counter() - t0, 3)
        log[stage] = entry
        return ok

    status, started_up, clean = "succeeded", False, False
    try:
        for stage in ordered:
            if stage == "down":
                continue
            log.pop(stage, None)
            if not record(stage, lambda s=stage: execute(s)):
                status = "failed"
                break
            started_up = started_up or stage == "up"
        clean = status == "succeeded"
    finally:
        if "down" in ordered or (started_up and not clean):
            if not record("down", runtime.down):
                status = "failed"

    manifest["status"] = status
    _describe_inputs(ctx, manifest)
    write_manifest(manifest_path, manifest)
    return manifest


def _describe_inputs(ctx: RunContext, manifest: dict[str, Any]) -> None:
    try:
        manifest["config"] = {
            "path": str(ctx.config.path.relative_to(ctx.root)),
            "hash": ctx.config.hash(ctx.spec.track),
            "diff_vs_default": ctx.config.diff(ctx.spec.track),
        }
    except (FileNotFoundError, ValueError) as e:
        manifest["config"] = {"error": str(e)}
    prepared = ctx.ws.scenario_dir(ctx.spec.scenario) / "prepare.manifest.json"
    if prepared.exists():
        info = read_json(prepared)
        manifest["dataset"] = info["dataset"]
        manifest["seed"] = info["seed"]
```

`harness/irbench/cli.py` — add imports and the `run` command:

```python
from irbench.io import read_json, read_jsonl, read_qrels, read_run
from irbench.runs import STAGES, RunSpec
from irbench.stages.run import run_pipeline
```

```python
@app.command()
def run(
    scenario: str = typer.Option(..., "--scenario"),
    engine: str = typer.Option(..., "--engine", help="Folder name under engines/."),
    model: str = typer.Option(..., "--model"),
    track: str = typer.Option("default", "--track", help="default or tuned."),
    hardware: str = typer.Option("local", "--hardware", help="Profile under hardware/."),
    host: str = typer.Option("localhost", "--host", help="Host where the engine port is published."),
    stages: str = typer.Option(",".join(STAGES), "--stages", help="Comma-separated stage names."),
    root: Path = ROOT_OPTION,
) -> None:
    """Run the pipeline for one scenario × engine × model × track."""
    spec = RunSpec(
        scenario=scenario, engine=engine, model=model, track=track, hardware=hardware, host=host
    )
    names = [s.strip() for s in stages.split(",") if s.strip()]
    try:
        manifest = run_pipeline(root, spec, names)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e

    for name, info in manifest["stages"].items():
        line = f"{name}: {info['status']}"
        if info["status"] == "failed":
            line += f" — {info['error']}"
        typer.echo(line)
    run_dir = Workspace(root).run_dir(spec.run_id)
    if manifest["stages"].get("score", {}).get("status") == "succeeded":
        scores = read_json(run_dir / "scores.json")
        for g in scores["relevance"]:
            if g["perturbation"] == "all":
                typer.echo(f"{g['mode']}/{g['base']}: nDCG@10={g['metrics']['ndcg@10']:.4f} n={g['n']}")
        for key, reason in scores["not_applicable"].items():
            typer.echo(f"{key.replace(':', '/')}: N/A ({reason})")
    typer.echo(f"run dir: {run_dir}")
    if manifest["status"] != "succeeded":
        raise typer.Exit(code=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: all pass; ruff clean (run `uv run ruff format .` first if format reports files).

- [ ] **Step 5: Commit**

```bash
git add harness/irbench/stages/score_run.py harness/irbench/stages/run.py harness/irbench/cli.py harness/tests/test_run_pipeline.py
git commit -m "feat: irbench run pipeline with per-stage manifest and run scoring"
```

---

### Task 9: Meilisearch adapter

**Files:**
- Create: `engines/meilisearch/adapter.py`, `engines/meilisearch/docker/compose.yaml`, `engines/meilisearch/configs/ecommerce/product-search.yaml`, `engines/meilisearch/README.md`, `harness/tests/adapter_helpers.py`
- Test: `harness/tests/test_adapter_meilisearch.py`

**Interfaces:**
- Consumes: `EngineAdapter`, `Request`, `SearchResult`, `call`, `poll`, `require_vector`, `EngineError` (Task 3); `deep_merge`, `TrackConfig`, `load_engine_config` (Task 2).
- Produces: `engines/meilisearch/adapter.py` with `Adapter` (`name="meilisearch"`, `host_port=17700`, `health_path="/health"`, `data_dir="/meili_data"`), module functions `filter_expression(f) -> str`, `synonyms_setting(groups) -> dict[str, list[str]]`, method `index_settings() -> dict`. Test helpers `make_adapter(engine, track=None, dim=3)`, `make_task(mode, base=None, text="grey sofa", **extra)`.
- Index settings (default track): `searchableAttributes` = scenario order, `filterableAttributes` = filterable ∪ facets, `sortableAttributes`, `synonyms` (each term → the other terms of its group), `faceting.maxValuesPerFacet` = `max_facet_values`, `embedders.default` = `{"source": "userProvided", "dimensions": dim}`; then `config.index` deep-merged.
- Requests: `POST /indexes/<index>/search`. keyword `{q}`; semantic `{vector, hybrid: {embedder: "default", semanticRatio: 1.0}}`; hybrid `{q, vector, hybrid: {embedder: "default"}}` (engine-default ratio); plus `filter`, `sort: ["field:dir"]`, `facets`, `limit`, `attributesToRetrieve: [id]`; then `config.search[base]` deep-merged.

- [ ] **Step 1: Write the test helpers** — `harness/tests/adapter_helpers.py`

```python
from conftest import REPO_ROOT
from irbench.engines.base import IndexContext, load_adapter_class
from irbench.engines.config import TrackConfig
from irbench.models import QueryTask
from irbench.scenario import load_scenario
from irbench.terms import load_term_groups

SCENARIO = "ecommerce/product-search"


def make_adapter(engine, track=None, dim=3):
    cls = load_adapter_class(REPO_ROOT, engine)
    scenario = load_scenario(REPO_ROOT, SCENARIO)
    return cls(IndexContext(
        scenario=scenario,
        config=TrackConfig.model_validate(track or {}),
        index="idx",
        dim=dim,
        synonyms=load_term_groups(scenario.synonyms_path),
    ))


def make_task(mode, base=None, text="grey sofa", qid="1", **extra):
    base = base or mode
    return QueryTask(task_id=f"{qid}:{mode}:{base}", qid=qid, variant_of=qid,
                     perturbation="original", text=text, mode=mode, base=base, **extra)
```

- [ ] **Step 2: Write the failing tests** — `harness/tests/test_adapter_meilisearch.py`

```python
import pytest

from adapter_helpers import SCENARIO, make_adapter, make_task
from conftest import REPO_ROOT
from irbench.engines.config import load_engine_config
from irbench.models import Filter, Sort

V = [0.1, 0.2, 0.3]


def _body(task, vector=None, limit=100, facets=True, track=None):
    return make_adapter("meilisearch", track).render(task, vector, limit, facets).body


def test_config_loads_and_declares_nothing_unsupported():
    config = load_engine_config(REPO_ROOT, "meilisearch", SCENARIO)
    assert config.track("default").unsupported == {}


def test_index_settings_follow_the_scenario():
    s = make_adapter("meilisearch").index_settings()
    assert s["searchableAttributes"] == ["name", "product_class", "category", "features", "description"]
    assert s["filterableAttributes"] == ["average_rating", "category", "product_class", "review_count"]
    assert s["sortableAttributes"] == ["average_rating", "review_count"]
    assert s["synonyms"]["grey"] == ["gray"] and s["synonyms"]["3 seater"] == ["three seater"]
    assert s["faceting"] == {"maxValuesPerFacet": 1000}
    assert s["embedders"] == {"default": {"source": "userProvided", "dimensions": 3}}


def test_index_settings_take_config_overrides():
    s = make_adapter("meilisearch", {"index": {"typoTolerance": {"enabled": False}}}).index_settings()
    assert s["typoTolerance"] == {"enabled": False}
    assert s["searchableAttributes"][0] == "name"


def test_keyword_semantic_hybrid_requests():
    kw = _body(make_task("keyword"))
    assert kw == {"q": "grey sofa", "limit": 100, "attributesToRetrieve": ["id"]}
    sem = _body(make_task("semantic"), V)
    assert "q" not in sem
    assert sem["vector"] == V and sem["hybrid"] == {"embedder": "default", "semanticRatio": 1.0}
    hyb = _body(make_task("hybrid"), V)
    assert hyb["q"] == "grey sofa" and hyb["vector"] == V and hyb["hybrid"] == {"embedder": "default"}
    with pytest.raises(ValueError, match="query vector"):
        _body(make_task("semantic"), None)


def test_filter_expressions_quote_strings_and_format_numbers():
    eq = make_task("filtered", "keyword", filter=Filter(field="product_class", op="eq", value='Kid\'s "Beds"'))
    assert _body(eq)["filter"] == 'product_class = "Kid\'s \\"Beds\\""'
    gte = make_task("filtered", "semantic", filter=Filter(field="average_rating", op="gte", value=4))
    assert _body(gte, V)["filter"] == "average_rating >= 4.0"


def test_sort_and_facets():
    s = make_task("sorted", "keyword", sort=Sort(field="review_count", direction="desc"))
    assert _body(s)["sort"] == ["review_count:desc"]
    f = make_task("faceted", "keyword", facets=["product_class"])
    assert _body(f)["facets"] == ["product_class"]
    check = _body(f, limit=1000, facets=False)
    assert "facets" not in check and check["limit"] == 1000


def test_search_overrides_apply_to_their_base_only():
    track = {"search": {"hybrid": {"hybrid": {"semanticRatio": 0.7}}}}
    assert _body(make_task("hybrid"), V, track=track)["hybrid"] == {"embedder": "default", "semanticRatio": 0.7}
    assert "hybrid" not in _body(make_task("keyword"), track=track)


def test_parse_ids_and_facets():
    adapter = make_adapter("meilisearch")
    payload = {"hits": [{"id": "7"}, {"id": "3"}], "facetDistribution": {"product_class": {"Beds": 2}}}
    result = adapter.parse(make_task("faceted", "keyword", facets=["product_class"]), payload)
    assert result.ids == ["7", "3"] and result.facets == {"product_class": {"Beds": 2}}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest harness/tests/test_adapter_meilisearch.py -q`
Expected: FAIL — `FileNotFoundError: .../engines/meilisearch/adapter.py missing`.

- [ ] **Step 4: Implement**

`engines/meilisearch/adapter.py`:

```python
"""Meilisearch adapter. Parity notes: engines/meilisearch/README.md."""

import time
from typing import Any

import httpx
import numpy as np

from irbench.engines.base import (
    EngineAdapter,
    EngineError,
    IndexContext,
    Request,
    SearchResult,
    call,
    poll,
    require_vector,
)
from irbench.engines.config import deep_merge
from irbench.models import Filter, QueryTask

EMBEDDER = "default"
BATCH_SIZE = 1000
SETUP_TIMEOUT_S = 600


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def filter_expression(f: Filter) -> str:
    if f.op == "eq":
        return f"{f.field} = {_quote(str(f.value))}"
    return f"{f.field} {'>=' if f.op == 'gte' else '<='} {float(f.value)}"


def synonyms_setting(groups: list[list[str]]) -> dict[str, list[str]]:
    return {term: [t for t in group if t != term] for group in groups for term in group}


class Adapter(EngineAdapter):
    name = "meilisearch"
    host_port = 17700
    health_path = "/health"
    data_dir = "/meili_data"

    def __init__(self, ctx: IndexContext):
        super().__init__(ctx)
        self._pending: list[int] = []

    def index_settings(self) -> dict[str, Any]:
        s = self.ctx.scenario
        f = s.index_fields
        settings = {
            "searchableAttributes": list(f.searchable),
            "filterableAttributes": sorted(set(f.filterable) | set(f.facets)),
            "sortableAttributes": list(f.sortable),
            "synonyms": synonyms_setting(self.ctx.synonyms),
            "faceting": {"maxValuesPerFacet": s.max_facet_values},
            "embedders": {EMBEDDER: {"source": "userProvided", "dimensions": self.ctx.dim}},
        }
        return deep_merge(settings, self.ctx.config.index)

    def _wait(self, client: httpx.Client, uid: int, timeout_s: float) -> None:
        def done() -> bool:
            task = call(client, Request("GET", f"/tasks/{uid}"))
            if task["status"] in ("failed", "canceled"):
                raise EngineError(f"meilisearch task {uid} {task['status']}: {task.get('error')}")
            return task["status"] == "succeeded"

        poll(done, timeout_s, f"meilisearch task {uid}", interval_s=0.2)

    def create_index(self, client: httpx.Client) -> None:
        idx = self.ctx.index
        self._pending = []
        existing = Request("GET", f"/indexes/{idx}").send(client)
        if existing.status_code == 200:
            self._wait(client, call(client, Request("DELETE", f"/indexes/{idx}"))["taskUid"], SETUP_TIMEOUT_S)
        elif existing.status_code != 404:
            raise EngineError(f"GET /indexes/{idx} -> {existing.status_code}: {existing.text[:500]}")
        created = call(client, Request("POST", "/indexes", body={"uid": idx, "primaryKey": self.ctx.scenario.id_field}))
        self._wait(client, created["taskUid"], SETUP_TIMEOUT_S)
        settings = call(client, Request("PATCH", f"/indexes/{idx}/settings", body=self.index_settings()))
        self._wait(client, settings["taskUid"], SETUP_TIMEOUT_S)

    def load_documents(self, client: httpx.Client, docs: list[dict[str, Any]], vectors: np.ndarray) -> None:
        for start in range(0, len(docs), BATCH_SIZE):
            batch = [
                {**doc, "_vectors": {EMBEDDER: vectors[i].tolist()}}
                for i, doc in enumerate(docs[start : start + BATCH_SIZE], start=start)
            ]
            task = call(client, Request("POST", f"/indexes/{self.ctx.index}/documents", body=batch))
            self._pending.append(task["taskUid"])

    def wait_until_indexed(self, client: httpx.Client, n_docs: int, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        for uid in self._pending:
            self._wait(client, uid, max(0.0, deadline - time.monotonic()))
        stats = call(client, Request("GET", f"/indexes/{self.ctx.index}/stats"))
        if stats["numberOfDocuments"] != n_docs:
            raise EngineError(f"meilisearch holds {stats['numberOfDocuments']} documents, expected {n_docs}")

    def render(self, task: QueryTask, vector: list[float] | None, limit: int, facets: bool = True) -> Request:
        body: dict[str, Any] = {}
        if task.base in ("keyword", "hybrid"):
            body["q"] = task.text
        if task.base in ("semantic", "hybrid"):
            body["vector"] = require_vector(task, vector)
            body["hybrid"] = {"embedder": EMBEDDER}
            if task.base == "semantic":
                body["hybrid"]["semanticRatio"] = 1.0
        body["limit"] = limit
        body["attributesToRetrieve"] = [self.ctx.scenario.id_field]
        if task.filter is not None:
            body["filter"] = filter_expression(task.filter)
        if task.sort is not None:
            body["sort"] = [f"{task.sort.field}:{task.sort.direction}"]
        if facets and task.facets:
            body["facets"] = list(task.facets)
        body = deep_merge(body, self.search_params(task))
        return Request("POST", f"/indexes/{self.ctx.index}/search", body=body)

    def parse(self, task: QueryTask, payload: Any) -> SearchResult:
        id_field = self.ctx.scenario.id_field
        return SearchResult(
            ids=[str(hit[id_field]) for hit in payload["hits"]],
            facets=payload.get("facetDistribution"),
        )
```

`engines/meilisearch/docker/compose.yaml`:

```yaml
# Pinned engine image. The resolved digest is recorded in every run manifest.
services:
  engine:
    image: getmeili/meilisearch:v1.54.3
    container_name: irbench-meilisearch
    ports:
      - "17700:7700"
    environment:
      MEILI_ENV: development      # no master key: local benchmark only
      MEILI_NO_ANALYTICS: "true"
    volumes:
      - data:/meili_data
    healthcheck:
      test: ["CMD-SHELL", "curl -fsS http://127.0.0.1:7700/health >/dev/null || wget -qO- http://127.0.0.1:7700/health >/dev/null"]
      interval: 2s
      timeout: 3s
      retries: 60

volumes:
  data:
```

`engines/meilisearch/configs/ecommerce/product-search.yaml`:

```yaml
# Meilisearch on ecommerce/product-search.
#
# default: Meilisearch as shipped. adapter.py generates only what the scenario
#   schema requires (searchable order, filterable/sortable/facet fields,
#   synonyms.txt, userProvided embedder, facet value limit).
# tuned: changes within the parity rules (docs/methodology/parity). Add a
#   `tuned:` block here; it is deep-merged over `default`.
#
# Keys: index (merged into index settings), search.<keyword|semantic|hybrid>
# (merged into search bodies of that base), unsupported (key -> reason),
# justification (key -> why a non-default ANN or ranking parameter is set).
default:
  index: {}
  search: {}
  unsupported: {}
```

`engines/meilisearch/README.md`:

````markdown
# Meilisearch

Image: `getmeili/meilisearch:v1.54.3` (host port 17700). Adapter: `adapter.py`.

## What the default track sends

| Scenario requirement | Meilisearch setting |
|---|---|
| Searchable fields, in priority order | `searchableAttributes` in scenario order (Meilisearch ranks earlier attributes higher) |
| Filterable / facet fields | `filterableAttributes` |
| Sortable fields | `sortableAttributes` |
| `synonyms.txt` | `synonyms`, every term mapped to the other terms of its group |
| Facet value limit (1000) | `faceting.maxValuesPerFacet` |
| Shared vectors | embedder `default`, `source: userProvided`, documents carry `_vectors.default` |

Everything else is Meilisearch's default: ranking rules, typo tolerance, prefix search, matching strategy, `pagination.maxTotalHits`.

## Modes

- keyword: `q`.
- semantic: `vector` with `hybrid.semanticRatio: 1.0`, no `q`.
- hybrid: `q` + `vector` with Meilisearch's default `semanticRatio` (native fusion).
- filtered / sorted / faceted: `filter`, `sort`, `facets` on top of keyword or semantic.

## Changing this configuration

Edit `configs/ecommerce/product-search.yaml`. Changes to `default` must stay "as shipped plus the schema". Put anything else in a `tuned:` block. Run the contract tests (`uv run pytest -m docker -k meilisearch`) before opening a PR.
````

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest harness/tests/test_adapter_meilisearch.py -q && uv run ruff check .`
Expected: 8 passed; ruff clean. If ruff reports line length or format issues in `engines/`, run `uv run ruff format engines harness` and re-run.

- [ ] **Step 6: Commit**

```bash
git add engines/meilisearch harness/tests/adapter_helpers.py harness/tests/test_adapter_meilisearch.py
git commit -m "feat: Meilisearch adapter, pinned compose file and default config"
```

---

### Task 10: Elasticsearch adapter

**Files:**
- Create: `engines/elasticsearch/adapter.py`, `engines/elasticsearch/docker/compose.yaml`, `engines/elasticsearch/configs/ecommerce/product-search.yaml`, `engines/elasticsearch/README.md`
- Test: `harness/tests/test_adapter_elasticsearch.py`

**Interfaces:**
- Consumes: as Task 9; `make_adapter`, `make_task`.
- Produces: `Adapter` (`name="elasticsearch"`, `host_port=19200`, `health_path="/_cluster/health?wait_for_status=yellow&timeout=1s"`, `data_dir="/usr/share/elasticsearch/data"`); methods `mappings()`, `index_body()`, `exact_field(name)`.
- Mapping (default track): searchable → `text` with `search_analyzer: irbench_search` (standard tokenizer + lowercase + `synonym_graph` from `synonyms.txt`); searchable fields that are also filterable/facets get a `raw` keyword sub-field; other non-numeric filterable/facet fields → `keyword`; numeric → `float`; `vector` → `dense_vector` (dims, `index: true`, `similarity: cosine`, other options at the version default); `dynamic: false`; `number_of_replicas: 0` (single node).
- Requests: `POST /<index>/_search` with `size`, `_source: false`. keyword `multi_match` over `name^5 … description^1`; semantic top-level `knn {field: vector, query_vector, k: top_k}`; hybrid = `query` + `knn` (Elasticsearch sums the scores: native linear combination); filters go in `bool.filter` (keyword) and `knn.filter` (vector); `sort: [{field: {order}}]`; facets via `aggs.<field>.terms {field: exact_field, size: max_facet_values}`.

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_adapter_elasticsearch.py`

```python
from adapter_helpers import SCENARIO, make_adapter, make_task
from conftest import REPO_ROOT
from irbench.engines.config import load_engine_config
from irbench.models import Filter, Sort

V = [0.1, 0.2, 0.3]
FIELDS = ["name^5", "product_class^4", "category^3", "features^2", "description^1"]


def _body(task, vector=None, limit=100, facets=True, track=None):
    return make_adapter("elasticsearch", track).render(task, vector, limit, facets).body


def test_config_loads_and_declares_nothing_unsupported():
    assert load_engine_config(REPO_ROOT, "elasticsearch", SCENARIO).track("default").unsupported == {}


def test_mappings_follow_the_scenario():
    props = make_adapter("elasticsearch").mappings()["properties"]
    assert props["name"] == {"type": "text", "search_analyzer": "irbench_search"}
    assert props["product_class"]["fields"] == {"raw": {"type": "keyword"}}
    assert props["category"]["fields"] == {"raw": {"type": "keyword"}}
    assert props["average_rating"] == {"type": "float"}
    assert props["vector"] == {"type": "dense_vector", "dims": 3, "index": True, "similarity": "cosine"}
    assert make_adapter("elasticsearch").mappings()["dynamic"] is False


def test_index_body_loads_synonyms_and_takes_overrides():
    body = make_adapter("elasticsearch", {"index": {"settings": {"refresh_interval": "30s"}}}).index_body()
    settings = body["settings"]
    assert settings["number_of_replicas"] == 0 and settings["refresh_interval"] == "30s"
    syn = settings["analysis"]["filter"]["irbench_synonyms"]
    assert syn["type"] == "synonym_graph" and "grey, gray" in syn["synonyms"]
    assert settings["analysis"]["analyzer"]["irbench_search"]["filter"] == ["lowercase", "irbench_synonyms"]


def test_keyword_semantic_hybrid_requests():
    kw = _body(make_task("keyword"))
    assert kw == {"size": 100, "_source": False,
                  "query": {"multi_match": {"query": "grey sofa", "fields": FIELDS}}}
    sem = _body(make_task("semantic"), V)
    assert "query" not in sem
    assert sem["knn"] == {"field": "vector", "query_vector": V, "k": 100}
    hyb = _body(make_task("hybrid"), V)
    assert hyb["query"]["multi_match"]["query"] == "grey sofa" and hyb["knn"]["k"] == 100


def test_filters_on_keyword_and_vector_bases():
    eq = Filter(field="product_class", op="eq", value="Beds")
    kw = _body(make_task("filtered", "keyword", filter=eq))
    assert kw["query"]["bool"]["filter"] == [{"term": {"product_class.raw": "Beds"}}]
    gte = Filter(field="average_rating", op="gte", value=4.5)
    sem = _body(make_task("filtered", "semantic", filter=gte), V)
    assert sem["knn"]["filter"] == [{"range": {"average_rating": {"gte": 4.5}}}]


def test_sort_and_facets():
    s = _body(make_task("sorted", "keyword", sort=Sort(field="average_rating", direction="desc")))
    assert s["sort"] == [{"average_rating": {"order": "desc"}}]
    f = make_task("faceted", "semantic", facets=["product_class"])
    assert _body(f, V)["aggs"] == {
        "product_class": {"terms": {"field": "product_class.raw", "size": 1000}}
    }
    check = _body(f, V, limit=1000, facets=False)
    assert "aggs" not in check and check["size"] == 1000 and check["knn"]["k"] == 100


def test_parse_ids_and_aggregations():
    payload = {
        "hits": {"hits": [{"_id": "9"}, {"_id": "2"}]},
        "aggregations": {"product_class": {"buckets": [{"key": "Beds", "doc_count": 3}]}},
    }
    result = make_adapter("elasticsearch").parse(make_task("faceted", "keyword", facets=["product_class"]), payload)
    assert result.ids == ["9", "2"] and result.facets == {"product_class": {"Beds": 3}}
    assert make_adapter("elasticsearch").parse(make_task("keyword"), {"hits": {"hits": []}}).facets is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest harness/tests/test_adapter_elasticsearch.py -q`
Expected: FAIL — `FileNotFoundError: .../engines/elasticsearch/adapter.py missing`.

- [ ] **Step 3: Implement**

`engines/elasticsearch/adapter.py`:

```python
"""Elasticsearch adapter. Parity notes: engines/elasticsearch/README.md."""

import json
from typing import Any

import httpx
import numpy as np

from irbench.engines.base import (
    EngineAdapter,
    EngineError,
    Request,
    SearchResult,
    call,
    poll,
    require_vector,
)
from irbench.engines.config import deep_merge
from irbench.models import Filter, QueryTask

VECTOR_FIELD = "vector"
SYNONYMS_FILTER = "irbench_synonyms"
SEARCH_ANALYZER = "irbench_search"
BATCH_SIZE = 1000


class Adapter(EngineAdapter):
    name = "elasticsearch"
    host_port = 19200
    health_path = "/_cluster/health?wait_for_status=yellow&timeout=1s"
    data_dir = "/usr/share/elasticsearch/data"

    def _exact_fields(self) -> set[str]:
        f = self.ctx.scenario.index_fields
        return (set(f.filterable) | set(f.facets)) - set(f.numeric)

    def exact_field(self, name: str) -> str:
        return f"{name}.raw" if name in self.ctx.scenario.index_fields.searchable else name

    def mappings(self) -> dict[str, Any]:
        f = self.ctx.scenario.index_fields
        exact = self._exact_fields()
        props: dict[str, Any] = {}
        for name in f.searchable:
            props[name] = {"type": "text", "search_analyzer": SEARCH_ANALYZER}
            if name in exact:
                props[name]["fields"] = {"raw": {"type": "keyword"}}
        for name in sorted(exact - set(f.searchable)):
            props[name] = {"type": "keyword"}
        for name in f.numeric:
            props[name] = {"type": "float"}
        props[VECTOR_FIELD] = {
            "type": "dense_vector",
            "dims": self.ctx.dim,
            "index": True,
            "similarity": "cosine",
        }
        return {"dynamic": False, "properties": props}

    def index_body(self) -> dict[str, Any]:
        filters = ["lowercase"]
        analysis: dict[str, Any] = {}
        if self.ctx.synonyms:
            analysis["filter"] = {
                SYNONYMS_FILTER: {
                    "type": "synonym_graph",
                    "synonyms": [", ".join(group) for group in self.ctx.synonyms],
                }
            }
            filters.append(SYNONYMS_FILTER)
        analysis["analyzer"] = {
            SEARCH_ANALYZER: {"type": "custom", "tokenizer": "standard", "filter": filters}
        }
        body = {
            "settings": {"number_of_replicas": 0, "analysis": analysis},
            "mappings": self.mappings(),
        }
        return deep_merge(body, self.ctx.config.index)

    def create_index(self, client: httpx.Client) -> None:
        idx = self.ctx.index
        response = Request("DELETE", f"/{idx}").send(client)
        if response.status_code not in (200, 404):
            raise EngineError(f"DELETE /{idx} -> {response.status_code}: {response.text[:500]}")
        call(client, Request("PUT", f"/{idx}", body=self.index_body()))

    def load_documents(self, client: httpx.Client, docs: list[dict[str, Any]], vectors: np.ndarray) -> None:
        idx, id_field = self.ctx.index, self.ctx.scenario.id_field
        for start in range(0, len(docs), BATCH_SIZE):
            lines = []
            for i, doc in enumerate(docs[start : start + BATCH_SIZE], start=start):
                lines.append(json.dumps({"index": {"_index": idx, "_id": str(doc[id_field])}}))
                lines.append(json.dumps({**doc, VECTOR_FIELD: vectors[i].tolist()}))
            result = call(client, Request(
                "POST", "/_bulk", content="\n".join(lines) + "\n",
                headers={"Content-Type": "application/x-ndjson"},
            ))
            if result.get("errors"):
                first = next(item["index"]["error"] for item in result["items"] if item["index"].get("error"))
                raise EngineError(f"bulk indexing failed: {first}")

    def wait_until_indexed(self, client: httpx.Client, n_docs: int, timeout_s: float) -> None:
        idx = self.ctx.index
        call(client, Request("POST", f"/{idx}/_refresh"))

        def ready() -> bool:
            return call(client, Request("GET", f"/{idx}/_count"))["count"] == n_docs

        poll(ready, timeout_s, f"{idx} to hold {n_docs} documents")

    def _text_query(self, text: str) -> dict[str, Any]:
        fields = self.ctx.scenario.index_fields.searchable
        n = len(fields)
        return {"multi_match": {"query": text, "fields": [f"{name}^{n - i}" for i, name in enumerate(fields)]}}

    def _filter_clause(self, f: Filter) -> dict[str, Any]:
        if f.op == "eq":
            return {"term": {self.exact_field(f.field): f.value}}
        return {"range": {f.field: {f.op: float(f.value)}}}

    def render(self, task: QueryTask, vector: list[float] | None, limit: int, facets: bool = True) -> Request:
        scenario = self.ctx.scenario
        body: dict[str, Any] = {"size": limit, "_source": False}
        filters = [self._filter_clause(task.filter)] if task.filter is not None else []
        if task.base in ("keyword", "hybrid"):
            query = self._text_query(task.text)
            body["query"] = {"bool": {"must": [query], "filter": filters}} if filters else query
        if task.base in ("semantic", "hybrid"):
            knn: dict[str, Any] = {
                "field": VECTOR_FIELD,
                "query_vector": require_vector(task, vector),
                "k": scenario.top_k,
            }
            if filters:
                knn["filter"] = filters
            body["knn"] = knn
        if task.sort is not None:
            body["sort"] = [{task.sort.field: {"order": task.sort.direction}}]
        if facets and task.facets:
            body["aggs"] = {
                name: {"terms": {"field": self.exact_field(name), "size": scenario.max_facet_values}}
                for name in task.facets
            }
        body = deep_merge(body, self.search_params(task))
        return Request("POST", f"/{self.ctx.index}/_search", body=body)

    def parse(self, task: QueryTask, payload: Any) -> SearchResult:
        aggs = payload.get("aggregations")
        facets = None
        if aggs is not None:
            facets = {
                name: {str(b["key"]): int(b["doc_count"]) for b in agg["buckets"]}
                for name, agg in aggs.items()
            }
        return SearchResult(ids=[hit["_id"] for hit in payload["hits"]["hits"]], facets=facets)
```

`engines/elasticsearch/docker/compose.yaml`:

```yaml
# Pinned engine image. The resolved digest is recorded in every run manifest.
services:
  engine:
    image: elasticsearch:9.5.5
    container_name: irbench-elasticsearch
    ports:
      - "19200:9200"
    environment:
      discovery.type: single-node
      xpack.security.enabled: "false"   # local benchmark only
      # No heap override: Elasticsearch sizes its heap from the container memory limit.
    volumes:
      - data:/usr/share/elasticsearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -fsS 'http://127.0.0.1:9200/_cluster/health?wait_for_status=yellow&timeout=1s' >/dev/null"]
      interval: 2s
      timeout: 3s
      retries: 90

volumes:
  data:
```

`engines/elasticsearch/configs/ecommerce/product-search.yaml`:

```yaml
# Elasticsearch on ecommerce/product-search.
#
# default: Elasticsearch as shipped. adapter.py generates only what the
#   scenario schema requires (mapping, synonyms.txt as a synonym_graph search
#   analyzer, field boosts from the scenario's priority order, dense_vector).
#   Not enabled by default: fuzziness, prefix matching, stemming, RRF.
# tuned: changes within the parity rules. Add a `tuned:` block here; it is
#   deep-merged over `default`. Example: fuzziness on keyword queries:
#     tuned:
#       search:
#         keyword: {query: {multi_match: {fuzziness: AUTO}}}
#
# Keys: index (merged into the create-index body), search.<keyword|semantic|hybrid>
# (merged into search bodies of that base), unsupported, justification.
default:
  index: {}
  search: {}
  unsupported: {}
```

The example in that comment would merge `fuzziness` into the plain `multi_match` query only; for filtered tasks the query is wrapped in `bool`. That is a known limitation of `search` overrides documented in the README below — do not change code for it.

`engines/elasticsearch/README.md`:

````markdown
# Elasticsearch

Image: `elasticsearch:9.5.5`, single node, security disabled (host port 19200). Adapter: `adapter.py`.

## What the default track sends

| Scenario requirement | Elasticsearch setting |
|---|---|
| Searchable fields, in priority order | `text` fields; `multi_match` (`best_fields`) with boosts `name^5, product_class^4, category^3, features^2, description^1` — Elasticsearch has no attribute order, so boosts `n - i` encode the scenario order |
| Filterable / facet text fields | `<field>.raw` keyword sub-field |
| Numeric filterable / sortable fields | `float` |
| `synonyms.txt` | `synonym_graph` filter in the search analyzer `irbench_search` (standard tokenizer + lowercase) |
| Facet value limit (1000) | `terms` aggregation `size` |
| Shared vectors | `dense_vector`, `similarity: cosine`, other `index_options` at this version's default |
| Single node | `number_of_replicas: 0` |

Not enabled by default (allowed in `tuned` with a visible setting): `fuzziness`, prefix matching, stemming analyzers, the `rrf`/`linear` retrievers.

## Modes

- keyword: `multi_match`.
- semantic: top-level `knn` with `k = top_k` (100) and the default `num_candidates`.
- hybrid: `query` + `knn` in one request; Elasticsearch adds the two scores (its documented native combination). No harness-side fusion.
- filtered: `bool.filter` (keyword) or `knn.filter` (semantic, pre-filtering).
- sorted: `sort` on the numeric field.
- faceted: `terms` aggregations; for semantic, the aggregation covers the `k` nearest documents.

## Config override caveat

`search.<base>` is deep-merged into the request body, so an override that targets `query.multi_match` applies only to unfiltered keyword requests (filtered ones wrap the query in `bool`). Use `adapter.py` for changes that must reach every request shape.

## Changing this configuration

Edit `configs/ecommerce/product-search.yaml` and run `uv run pytest -m docker -k elasticsearch` before opening a PR.
````

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest harness/tests/test_adapter_elasticsearch.py -q && uv run ruff check .`
Expected: 7 passed; ruff clean (format with `uv run ruff format engines harness` if needed).

- [ ] **Step 5: Commit**

```bash
git add engines/elasticsearch harness/tests/test_adapter_elasticsearch.py
git commit -m "feat: Elasticsearch adapter, pinned compose file and default config"
```

---

### Task 11: Qdrant adapter

**Files:**
- Create: `engines/qdrant/adapter.py`, `engines/qdrant/docker/compose.yaml`, `engines/qdrant/configs/ecommerce/product-search.yaml`, `engines/qdrant/README.md`
- Test: `harness/tests/test_adapter_qdrant.py`

**Interfaces:**
- Consumes: as Task 9.
- Produces: `Adapter` (`name="qdrant"`, `host_port=16333`, `health_path="/readyz"`, `data_dir="/qdrant/storage"`); `point_id(doc_id) -> int | str` (digit strings → int, else UUIDv5); methods `collection_body()`, `payload_indexes() -> dict[str, str]`.
- Collection: `vectors {size: dim, distance: Cosine}` deep-merged with `config.index`; payload indexes `float` for numeric fields, `keyword` for other filterable/facet/sortable fields; the original ID is kept in the payload and returned via `with_payload: [id]`.
- Requests: `POST /collections/<c>/points/query`: semantic `{query: vector, limit, with_payload}`; filtered adds `filter.must[match|range]`; sorted uses `prefetch {query: vector, limit: top_k, filter?}` + `query {order_by {key, direction}}`.
- Default config declares unsupported: `"*:keyword"`, `hybrid`, `"faceted:semantic"` (reasons in the file).

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_adapter_qdrant.py`

```python
import uuid

import pytest

from adapter_helpers import SCENARIO, make_adapter, make_task
from conftest import REPO_ROOT
from irbench.engines.config import load_engine_config
from irbench.models import Filter, Sort

V = [0.1, 0.2, 0.3]


def _load_point_id():
    import importlib.util

    spec = importlib.util.spec_from_file_location("q", REPO_ROOT / "engines" / "qdrant" / "adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.point_id


def _body(task, vector=V, limit=100, facets=True):
    return make_adapter("qdrant").render(task, vector, limit, facets).body


def test_point_ids():
    point_id = _load_point_id()
    assert point_id("42") == 42
    assert point_id("sku-1") == str(uuid.uuid5(uuid.NAMESPACE_URL, "sku-1"))


def test_config_declares_what_the_adapter_cannot_do():
    track = load_engine_config(REPO_ROOT, "qdrant", SCENARIO).track("default")
    assert track.unsupported_reason(make_task("keyword"))
    assert track.unsupported_reason(make_task("filtered", "keyword"))
    assert track.unsupported_reason(make_task("hybrid"))
    assert track.unsupported_reason(make_task("faceted", "semantic", facets=["product_class"]))
    assert track.unsupported_reason(make_task("semantic")) is None
    assert track.unsupported_reason(make_task("filtered", "semantic")) is None
    assert track.unsupported_reason(make_task("sorted", "semantic")) is None


def test_collection_and_payload_indexes():
    adapter = make_adapter("qdrant")
    assert adapter.collection_body() == {"vectors": {"size": 3, "distance": "Cosine"}}
    assert adapter.payload_indexes() == {
        "average_rating": "float", "category": "keyword",
        "product_class": "keyword", "review_count": "float",
    }


def test_semantic_and_filtered_requests():
    assert _body(make_task("semantic")) == {"query": V, "limit": 100, "with_payload": ["id"]}
    eq = make_task("filtered", "semantic", filter=Filter(field="product_class", op="eq", value="Beds"))
    assert _body(eq)["filter"] == {"must": [{"key": "product_class", "match": {"value": "Beds"}}]}
    gte = make_task("filtered", "semantic", filter=Filter(field="average_rating", op="gte", value=4))
    assert _body(gte)["filter"] == {"must": [{"key": "average_rating", "range": {"gte": 4.0}}]}


def test_sorted_request_reorders_the_nearest_neighbours():
    s = make_task("sorted", "semantic", sort=Sort(field="review_count", direction="desc"))
    assert _body(s) == {
        "prefetch": {"query": V, "limit": 100},
        "query": {"order_by": {"key": "review_count", "direction": "desc"}},
        "limit": 100,
        "with_payload": ["id"],
    }


def test_keyword_tasks_are_refused():
    with pytest.raises(ValueError, match="semantic"):
        _body(make_task("keyword"), None)


def test_parse_returns_original_ids_from_payload():
    payload = {"result": {"points": [{"id": 7, "score": 0.9, "payload": {"id": "7"}}]}}
    assert make_adapter("qdrant").parse(make_task("semantic"), payload).ids == ["7"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest harness/tests/test_adapter_qdrant.py -q`
Expected: FAIL — `FileNotFoundError: .../engines/qdrant/adapter.py missing`.

- [ ] **Step 3: Implement**

`engines/qdrant/adapter.py`:

```python
"""Qdrant adapter. Parity notes: engines/qdrant/README.md."""

import uuid
from typing import Any

import httpx
import numpy as np

from irbench.engines.base import (
    EngineAdapter,
    EngineError,
    Request,
    SearchResult,
    call,
    poll,
    require_vector,
)
from irbench.engines.config import deep_merge
from irbench.models import Filter, QueryTask

BATCH_SIZE = 256


def point_id(doc_id: str) -> int | str:
    return int(doc_id) if doc_id.isdigit() else str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


class Adapter(EngineAdapter):
    name = "qdrant"
    host_port = 16333
    health_path = "/readyz"
    data_dir = "/qdrant/storage"

    def collection_body(self) -> dict[str, Any]:
        body = {"vectors": {"size": self.ctx.dim, "distance": "Cosine"}}
        return deep_merge(body, self.ctx.config.index)

    def payload_indexes(self) -> dict[str, str]:
        f = self.ctx.scenario.index_fields
        names = sorted(set(f.filterable) | set(f.facets) | set(f.sortable))
        return {name: "float" if name in f.numeric else "keyword" for name in names}

    def create_index(self, client: httpx.Client) -> None:
        c = self.ctx.index
        response = Request("DELETE", f"/collections/{c}").send(client)
        if response.status_code not in (200, 404):
            raise EngineError(f"DELETE /collections/{c} -> {response.status_code}: {response.text[:500]}")
        call(client, Request("PUT", f"/collections/{c}", body=self.collection_body()))
        for name, schema in self.payload_indexes().items():
            call(client, Request(
                "PUT", f"/collections/{c}/index?wait=true",
                body={"field_name": name, "field_schema": schema},
            ))

    def load_documents(self, client: httpx.Client, docs: list[dict[str, Any]], vectors: np.ndarray) -> None:
        c, id_field = self.ctx.index, self.ctx.scenario.id_field
        for start in range(0, len(docs), BATCH_SIZE):
            points = [
                {"id": point_id(str(doc[id_field])), "vector": vectors[i].tolist(), "payload": doc}
                for i, doc in enumerate(docs[start : start + BATCH_SIZE], start=start)
            ]
            call(client, Request("PUT", f"/collections/{c}/points?wait=true", body={"points": points}))

    def wait_until_indexed(self, client: httpx.Client, n_docs: int, timeout_s: float) -> None:
        c = self.ctx.index

        def ready() -> bool:
            info = call(client, Request("GET", f"/collections/{c}"))["result"]
            if info["status"] != "green":
                return False
            count = call(client, Request("POST", f"/collections/{c}/points/count", body={"exact": True}))
            return count["result"]["count"] == n_docs

        poll(ready, timeout_s, f"qdrant collection {c} to be green with {n_docs} points")

    def _filter(self, f: Filter) -> dict[str, Any]:
        if f.op == "eq":
            return {"must": [{"key": f.field, "match": {"value": f.value}}]}
        return {"must": [{"key": f.field, "range": {f.op: float(f.value)}}]}

    def render(self, task: QueryTask, vector: list[float] | None, limit: int, facets: bool = True) -> Request:
        if task.base != "semantic" or (facets and task.facets):
            raise ValueError(
                f"qdrant adapter renders semantic, filtered and sorted tasks on the semantic base "
                f"only; {task.task_id} must be declared unsupported in the config"
            )
        vector = require_vector(task, vector)
        query_filter = self._filter(task.filter) if task.filter is not None else None
        body: dict[str, Any]
        if task.sort is not None:
            prefetch: dict[str, Any] = {"query": vector, "limit": self.ctx.scenario.top_k}
            if query_filter:
                prefetch["filter"] = query_filter
            body = {
                "prefetch": prefetch,
                "query": {"order_by": {"key": task.sort.field, "direction": task.sort.direction}},
            }
        else:
            body = {"query": vector}
            if query_filter:
                body["filter"] = query_filter
        body["limit"] = limit
        body["with_payload"] = [self.ctx.scenario.id_field]
        body = deep_merge(body, self.search_params(task))
        return Request("POST", f"/collections/{self.ctx.index}/points/query", body=body)

    def parse(self, task: QueryTask, payload: Any) -> SearchResult:
        id_field = self.ctx.scenario.id_field
        return SearchResult(ids=[str(p["payload"][id_field]) for p in payload["result"]["points"]])
```

`engines/qdrant/docker/compose.yaml`:

```yaml
# Pinned engine image. The resolved digest is recorded in every run manifest.
services:
  engine:
    image: qdrant/qdrant:v1.19.2
    container_name: irbench-qdrant
    ports:
      - "16333:6333"
    environment:
      QDRANT__TELEMETRY_DISABLED: "true"
    volumes:
      - data:/qdrant/storage
    healthcheck:
      # The image ships without curl/wget; bash's /dev/tcp checks the port is open.
      test: ["CMD-SHELL", "bash -c ':> /dev/tcp/127.0.0.1/6333' || exit 1"]
      interval: 2s
      timeout: 3s
      retries: 60

volumes:
  data:
```

`engines/qdrant/configs/ecommerce/product-search.yaml`:

```yaml
# Qdrant on ecommerce/product-search.
#
# default: Qdrant as shipped. adapter.py generates only what the scenario
#   schema requires (Cosine vectors, payload indexes for filterable, sortable
#   and facet fields). HNSW and search parameters are Qdrant's defaults.
# tuned: changes within the parity rules; add a `tuned:` block (deep-merged).
#
# Keys: index (merged into the create-collection body),
# search.<keyword|semantic|hybrid> (merged into query bodies of that base),
# unsupported (key -> reason, shown as N/A in results), justification.
default:
  index: {}
  search: {}
  unsupported:
    "*:keyword": >-
      Not implemented by the IRBench v0.1 Qdrant adapter. Qdrant's ranked keyword
      search uses sparse BM25 vectors; contributions adding it are welcome.
    hybrid: >-
      Needs keyword retrieval, which the IRBench v0.1 Qdrant adapter does not
      implement (see *:keyword).
    "faceted:semantic": >-
      Qdrant's facet API counts the points matching a filter, not the points
      returned by a vector query, so counts cannot describe the query's results.
```

`engines/qdrant/README.md`:

````markdown
# Qdrant

Image: `qdrant/qdrant:v1.19.2` (host port 16333). Adapter: `adapter.py`.

## What the default track sends

| Scenario requirement | Qdrant setting |
|---|---|
| Shared vectors | collection `vectors: {size, distance: Cosine}`; HNSW at Qdrant's defaults |
| Filterable / facet text fields | `keyword` payload indexes |
| Numeric filterable / sortable fields | `float` payload indexes |
| Document IDs | numeric IDs become integer point IDs, others UUIDv5; the original ID stays in the payload and is what results report |

## Modes

- semantic: Query API `query: <vector>`.
- filtered (semantic base): `filter.must` with `match` / `range`.
- sorted (semantic base): `prefetch` the `top_k` nearest points, then `order_by` the sort field. Qdrant's `order_by` leaves out points that lack the field.
- N/A, with the reasons in the config: keyword retrieval and everything built on it (keyword-based filtered/sorted/faceted, hybrid), and faceted on the semantic base.

## Changing this configuration

Edit `configs/ecommerce/product-search.yaml`. Removing an `unsupported` entry requires implementing that mode in `adapter.py`. Run `uv run pytest -m docker -k qdrant` before opening a PR.
````

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest harness/tests/test_adapter_qdrant.py -q && uv run ruff check .`
Expected: 7 passed; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add engines/qdrant harness/tests/test_adapter_qdrant.py
git commit -m "feat: Qdrant adapter, pinned compose file and default config"
```

---

### Task 12: Docker contract tests against the real engines

**Files:**
- Create: `harness/tests/contract_fixture.py`, `harness/tests/test_contract.py`
- Modify (only if a contract fails): `engines/<engine>/adapter.py`, `engines/<engine>/configs/ecommerce/product-search.yaml`, `engines/<engine>/README.md`

**Interfaces:**
- Consumes: `DockerRuntime`, `wait_healthy`, `load_hardware` (Task 4); `load_adapter_class`, `IndexContext` (Task 3); `load_engine_config` (Task 2); `run_request` (Task 7); `check_task` (Task 5).
- Produces: `contract_docs() -> tuple[list[dict], np.ndarray]` — 100 WANDS-shaped docs and 8-dim unit vectors; doc `i` is `f"{COLORS[i // 10]} {ITEMS[i % 10]}"`. Known answers: doc `42` = "white table", doc `50` = "grey sofa", doc `58` = "grey stool". `uv run pytest -m docker` runs the contract for every engine.

- [ ] **Step 1: Write the fixture** — `harness/tests/contract_fixture.py`

```python
"""100 product docs with known answers for the adapter contract tests (spec §10)."""

import numpy as np

ITEMS = ["sofa", "bed", "table", "chair", "lamp", "desk", "rug", "shelf", "stool", "mirror"]
COLORS = ["red", "blue", "green", "black", "white", "grey", "brown", "yellow", "pink", "orange"]
DIM = 8


def rating(i: int) -> float | None:
    return None if i % 7 == 0 else round(1.0 + (i * 37 % 41) / 10, 1)


def review_count(i: int) -> float | None:
    return None if i % 11 == 0 else float(i * 13 % 97)


def product_class(i: int) -> list[str]:
    classes = [ITEMS[i % 10].title() + "s"]
    if i % 20 == 0:
        classes.append("Sectionals")  # multi-valued facet field
    return classes


def contract_docs() -> tuple[list[dict], np.ndarray]:
    docs = []
    for i in range(100):
        color, item = COLORS[i // 10], ITEMS[i % 10]
        docs.append({
            "id": str(i),
            "name": f"{color} {item}",
            "product_class": product_class(i),
            "category": ["Furniture", item.title()],
            "description": f"a {color} {item} for the home",
            "features": [f"color : {color}"],
            "rating_count": review_count(i),
            "average_rating": rating(i),
            "review_count": review_count(i),
        })
    vectors = np.random.default_rng(7).normal(size=(100, DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return docs, vectors
```

- [ ] **Step 2: Write the contract tests** — `harness/tests/test_contract.py`

```python
"""Adapter contract: every engine indexes the fixture and passes known-answer
queries for every mode it claims. Needs Docker: `uv run pytest -m docker`."""

from dataclasses import dataclass

import httpx
import numpy as np
import pytest

from adapter_helpers import SCENARIO, make_task
from conftest import REPO_ROOT
from contract_fixture import COLORS, ITEMS, contract_docs, rating
from irbench.engines.base import EngineAdapter, IndexContext, load_adapter_class
from irbench.engines.config import load_engine_config
from irbench.models import Filter, Sort
from irbench.runtime.docker import DockerRuntime, wait_healthy
from irbench.runtime.hardware import load_hardware
from irbench.scenario import load_scenario
from irbench.score.checks import check_task
from irbench.stages.relevance import run_request
from irbench.terms import load_term_groups

pytestmark = pytest.mark.docker

ENGINES = ["meilisearch", "elasticsearch", "qdrant"]
INDEX = "irbench_contract"
TOP_K = 100
VERIFY_LIMIT = 1000


@dataclass
class Engine:
    adapter: EngineAdapter
    client: httpx.Client
    docs: list[dict]
    docs_by_id: dict[str, dict]
    vectors: np.ndarray

    def index(self):
        self.adapter.create_index(self.client)
        self.adapter.load_documents(self.client, self.docs, self.vectors)
        self.adapter.wait_until_indexed(self.client, len(self.docs), 300)

    def search(self, task, vector=None, limit=TOP_K, facets=True):
        reason = self.adapter.unsupported_reason(task)
        if reason:
            pytest.skip(f"{self.adapter.name}: {task.mode}:{task.base} N/A — {reason}")
        request = self.adapter.render(task, vector, limit, facets)
        return run_request(self.client, self.adapter, task, request, 30)

    def vec(self, doc_id: str) -> list[float]:
        return self.vectors[int(doc_id)].tolist()


@pytest.fixture(scope="module", params=ENGINES)
def engine(request):
    name = request.param
    cls = load_adapter_class(REPO_ROOT, name)
    scenario = load_scenario(REPO_ROOT, SCENARIO)
    docs, vectors = contract_docs()
    adapter = cls(IndexContext(
        scenario=scenario,
        config=load_engine_config(REPO_ROOT, name, SCENARIO).track("default"),
        index=INDEX,
        dim=vectors.shape[1],
        synonyms=load_term_groups(scenario.synonyms_path),
    ))
    runtime = DockerRuntime(REPO_ROOT, name, load_hardware(REPO_ROOT, "local"))
    runtime.up()
    client = httpx.Client(base_url=f"http://localhost:{cls.host_port}", timeout=60)
    try:
        wait_healthy(client, cls.health_path, 180)
        e = Engine(adapter, client, docs, {d["id"]: d for d in docs}, vectors)
        e.index()
        yield e
    finally:
        client.close()
        runtime.down()


def _ids_where(pred) -> set[str]:
    return {str(i) for i in range(100) if pred(i)}


def test_reindex_replaces_previous_index(engine):
    engine.index()  # second time into an existing index
    result = engine.search(make_task("semantic"), engine.vec("0"))
    assert len(result.ids) == len(set(result.ids)) == 100
    assert set(result.ids) == set(engine.docs_by_id)


def test_keyword_known_answer(engine):
    assert engine.search(make_task("keyword", text="white table")).ids[0] == "42"


def test_keyword_loads_shared_synonyms(engine):
    ids = engine.search(make_task("keyword", text="gray")).ids
    assert ids and set(ids) <= _ids_where(lambda i: COLORS[i // 10] == "grey")


def test_semantic_known_answer(engine):
    assert engine.search(make_task("semantic"), engine.vec("42")).ids[0] == "42"


def test_hybrid_known_answer(engine):
    assert engine.search(make_task("hybrid", text="white table"), engine.vec("42")).ids[0] == "42"


def test_filtered_keyword_returns_exactly_the_matching_docs(engine):
    f = Filter(field="average_rating", op="gte", value=3.0)
    task = make_task("filtered", "keyword", text="table", filter=f)
    result = engine.search(task)
    assert check_task(task, result, None, engine.docs_by_id, VERIFY_LIMIT).filter_violations == 0
    expected = _ids_where(lambda i: ITEMS[i % 10] == "table" and (rating(i) or 0) >= 3.0)
    assert set(result.ids) == expected


def test_filtered_semantic_prefilters(engine):
    f = Filter(field="product_class", op="eq", value="Sofas")
    task = make_task("filtered", "semantic", filter=f)
    result = engine.search(task, engine.vec("50"))
    assert result.ids[0] == "50"
    assert check_task(task, result, None, engine.docs_by_id, VERIFY_LIMIT).filter_violations == 0
    assert set(result.ids) == _ids_where(lambda i: ITEMS[i % 10] == "sofa")


def test_sorted_keyword_orders_and_puts_missing_values_last(engine):
    task = make_task("sorted", "keyword", text="chair",
                     sort=Sort(field="average_rating", direction="desc"))
    result = engine.search(task)
    assert check_task(task, result, None, engine.docs_by_id, VERIFY_LIMIT).order_violations == 0
    assert set(result.ids) == _ids_where(lambda i: ITEMS[i % 10] == "chair")


def test_sorted_semantic_orders(engine):
    task = make_task("sorted", "semantic", sort=Sort(field="review_count", direction="desc"))
    result = engine.search(task, engine.vec("42"))
    assert result.ids
    assert check_task(task, result, None, engine.docs_by_id, VERIFY_LIMIT).order_violations == 0


@pytest.mark.parametrize("base", ["keyword", "semantic"])
def test_facet_counts_match_the_result_set(engine, base):
    task = make_task("faceted", base, text="red", facets=["product_class"])
    vector = engine.vec("0") if base == "semantic" else None
    result = engine.search(task, vector)
    check_ids = engine.search(task, vector, limit=VERIFY_LIMIT, facets=False).ids
    check = check_task(task, result, check_ids, engine.docs_by_id, VERIFY_LIMIT)
    assert check.facet_verifiable is True
    assert check.facet_accuracy == 1.0, (result.facets, check_ids)
```

- [ ] **Step 3: Run the contract against each engine**

Run: `uv run pytest -m docker -k meilisearch -v 2>&1 | tail -30`
Then: `uv run pytest -m docker -k elasticsearch -v 2>&1 | tail -30`
Then: `uv run pytest -m docker -k qdrant -v 2>&1 | tail -30`
Expected: every test passes or skips with an N/A reason. Qdrant skips the keyword, hybrid and semantic-faceted tests; Meilisearch and Elasticsearch skip nothing. Each engine needs Docker running and its host port (17700 / 19200 / 16333) free.

The API details in Tasks 9–11 come from public docs. When a contract test fails, it shows where an engine behaves differently. In that case:
1. Read the failure with superpowers:systematic-debugging and the engine's docs for the pinned version.
2. If the adapter sends the wrong request, fix `adapter.py` and update that adapter's unit test in the same commit, so the unit test describes the request that works.
3. If the engine has no native way to do the mode, add an `unsupported` entry with a truthful reason in the config, and note it in the engine README.
4. Ledger each case as `Task 12: Ruling: <engine> <test> — <what differed> — <fix or N/A>`.

Do not weaken a contract assertion to make an engine pass. The contract is the parity bar.

- [ ] **Step 4: Confirm the default suite still deselects Docker tests**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: all unit tests pass with docker tests deselected; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add harness/tests/contract_fixture.py harness/tests/test_contract.py engines
git commit -m "test: Docker contract tests for Meilisearch, Elasticsearch and Qdrant"
```

---

### Task 13: Real WANDS runs on all three engines

**Files:** none expected. A run that exposes an adapter bug goes back through Task 12's procedure (fix, unit test, contract test, ruling).

**Interfaces:**
- Consumes: the whole pipeline; `bge-small-en-v1.5` vectors from Plan 1 (already cached under `.irbench/embeddings/wands/bge-small-en-v1.5/`).
- Produces: three succeeded runs under `.irbench/runs/ecommerce__product-search__<engine>__bge-small-en-v1.5__default__local/`, each with `manifest.json`, `scores.json`, `index.json`.

- [ ] **Step 1: Run Meilisearch end to end**

Run: `uv run irbench run --scenario ecommerce/product-search --engine meilisearch --model bge-small-en-v1.5 2>&1 | tail -30`
Expected: every stage `succeeded`; one nDCG@10 line per (mode, base): keyword, semantic, hybrid, plus filtered/sorted/faceted on both bases; no N/A lines; exit 0.

- [ ] **Step 2: Run Elasticsearch and Qdrant**

Run: `uv run irbench run --scenario ecommerce/product-search --engine elasticsearch --model bge-small-en-v1.5 2>&1 | tail -30`
Run: `uv run irbench run --scenario ecommerce/product-search --engine qdrant --model bge-small-en-v1.5 2>&1 | tail -30`
Expected: every stage `succeeded`. Qdrant prints N/A lines for keyword, hybrid, every keyword-based structured mode, and faceted/semantic.

- [ ] **Step 3: Sanity-check the three runs**

```bash
uv run python - <<'EOF'
import json, pathlib
for d in sorted(pathlib.Path(".irbench/runs").glob("ecommerce__product-search__*__bge-small-en-v1.5__default__local")):
    s = json.loads((d / "scores.json").read_text()); i = json.loads((d / "index.json").read_text())
    m = json.loads((d / "manifest.json").read_text())
    print(f"== {s['engine']}  status={m['status']}  image={m['engine_image']['digest']}  index={i['seconds']}s  errors={s['errors']}")
    for g in s["relevance"]:
        if g["perturbation"] in ("all", "original", "typo", "prefix", "synonym") and g["mode"] in ("keyword", "semantic", "hybrid"):
            print(f"  {g['mode']:9} {g['perturbation']:9} nDCG@10={g['metrics']['ndcg@10']:.3f} n={g['n']}")
    for c in s["checks"]:
        print(f"  check {c['mode']}/{c['base']}: filter_v={c['filter_violations']} order_v={c['order_violations']} facet_acc={c['facet_accuracy']} verified={c['facet_verified']} unverifiable={c['facet_unverifiable']} errors={c['errors']}")
EOF
```

Expected, per engine:
- `status=succeeded`, the image digest is set, and `errors=0`. A few timeouts are tolerable only if each one appears in `errors.jsonl` with its reason. Report them.
- Every `filter_violations` is 0.
- Semantic nDCG@10 for "all" lies between 0.3 and 0.9, and is similar for engines that use the same vectors. A gap above 0.05 between engines on semantic/original deserves a look, because the vectors are the same: check ANN defaults and quantization in the manifest and README before calling it engine behavior.
- Keyword nDCG@10 for typo and prefix is at or below original. If it is above, check that the variants loaded correctly.

Any `order_violations > 0` or `facet_accuracy < 1.0` is a real finding about that engine. Record it in the final message, with the task IDs from `checks.jsonl`. Do not hide it and do not "fix" it in the harness.

- [ ] **Step 4: Report**

No commit. Put the table printed by Step 3 in the final message, together with any non-zero checks and any rulings made along the way.

---

## Self-Review

**Spec coverage**

| Spec section | Where |
|---|---|
| §3 Engine adapter responsibilities (create index, load docs + vectors, wait, render, parse) | Task 3 (contract), Tasks 9–11 |
| §3 Engine config per engine per scenario; §7 tracks default/tuned | Task 2, configs in Tasks 9–11 |
| §4 `engines/<engine>/{adapter.py, configs/, docker/, README.md}`, `hardware/` | Tasks 4, 9–11 |
| §5.3 up: pinned image, resource limits from profile, healthcheck | Task 4 (override file, `--wait`), compose files Tasks 9–11 |
| §5.4 index: wall-clock time, peak RSS, on-disk size | Task 7 `index_stage` + Task 4 inspection helpers |
| §5.5 render: every (variant × mode) to an inspectable file | Task 6 (`rendered/<mode>/<base>/<qid>.json`) |
| §5.6 relevance: sequential, top 100, TREC run | Task 7 |
| §5.8 score per perturbation; §5 filtered/sorted/faceted checks | Tasks 5, 8 |
| §5.9 down: stop container, delete volumes | Task 4 (`down -v`), Task 8 (always after a started run) |
| §5 manifest fields | Task 8 (`run_pipeline`; harness git SHA via Plan 1 `write_manifest`) |
| §5 CLI `irbench run … --track --stages` | Task 8 |
| §7 same searchable fields + order, shared synonyms, same vectors/cosine/top-k, ANN defaults, native hybrid only, N/A for unsupported | Global Constraints; Tasks 9–11; Task 8 N/A reporting |
| §8 reference hardware profile | Task 4 |
| §9 failed stage → manifest, batch continues; 10 s timeout → scored 0 + error log; index timeout | Tasks 7, 8 (`index_timeout_s` in Task 7) |
| §10 adapter contract tests on a 100-doc fixture, known answers per claimed mode | Task 12 |
| §12 v0.1 engines Meilisearch, Elasticsearch, Qdrant | Tasks 9–13 |
| §5.7 perf, loadgen; §11 report; §7 CI parity lint; publishing vectors | Plans 3 and 4 |

**Spec deviations, decided here.** Each must be ledgered as a ruling when its task runs:
- Rendered files go to `.irbench/runs/<run-id>/rendered/<mode>/<base>/<qid>.json`, not `rendered/<engine>/<scenario>/<model>/<mode>/<qid>.json`. The run ID already encodes engine/scenario/model/track/hardware, and the base level is needed because the structured modes run on two bases.
- A request file stores `path`, not `url`. The host comes from `--host` plus the adapter's port, so the same files replay against any host (Plan 3).
- Runs are written to `.irbench/runs/`. Plan 4 copies reference-machine runs into `results/` for publication.
- Engine ports on the host are offset (17700/19200/16333), so the benchmark never collides with an engine a developer already runs.

**Placeholder scan.** Every code step contains its code. Task 12 Step 3 and Task 13 have open outcomes by nature: they verify against real engines. Both state the exact procedure and the ruling format.

**Type consistency.**
- `Request`/`SearchResult`/`IndexContext`/`EngineAdapter` (Task 3) are used unchanged in Tasks 6–12.
- `host_port` is used everywhere; no bare `port`.
- `RunContext.rendered_path(task, check=)` is shared by render and relevance.
- `TaskCheck`/`CheckSummary` field names match between Task 5, Task 7's assertions and Task 8's `scores.json` assertions.
- `run_request` is defined in Task 7 and consumed in Task 12.
- `STAGES` order is identical in `runs.py`, the CLI default, and the Task 8 assertion.
