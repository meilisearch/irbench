# IRBench Plan 1 — Harness Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the engine-independent half of IRBench: download and normalize WANDS, generate seeded messy query variants, derive per-mode query tasks with their relevance judgments, compute and cache embeddings, and score TREC run files — all driven by an `irbench` CLI.

**Architecture:** A Python package (`harness/irbench`) with one module per responsibility; every stage reads and writes plain files under `.irbench/` so later plans (engine adapters, Rust loadgen, report) consume them without importing harness internals. Scenarios and datasets are declarative YAML in the repo; generated data is never committed.

**Tech Stack:** Python 3.12, uv, typer, pydantic v2, PyYAML, httpx, pyarrow, numpy, ranx 0.3.21; optional sentence-transformers and openai for embeddings; pytest + ruff.

**Spec:** `docs/superpowers/specs/2026-10-09-irbench-design.md`

**Plan series:** This is Plan 1 of 4 for v0.1. Plan 2: Docker runtime + Meilisearch/Elasticsearch/Qdrant adapters (`up`, `index`, `render`, `relevance`, `down`, contract tests; wires filter/order/facet checks from Task 14 into run scoring). Plan 3: Rust loadgen + `perf` stage. Plan 4: static HTML report, Mintlify docs, CI parity lint, publishing vectors to Hugging Face and the `trec_eval` cross-check in CI.

## Global Constraints

- Python `>=3.12,<3.13` (ranx depends on numba).
- `ranx>=0.3.21`; metrics computed with `make_comparable=True`.
- Metrics: nDCG@10, Recall@10, Recall@100, MRR@10, Success@1 (ranx name `hit_rate@1`, reported as `success@1`); overall and per perturbation type.
- Perturbation kinds: `original|typo|prefix|synonym`. Typo: 1 edit for words of 4–7 chars, 2 edits for ≥ 8 chars; edit types adjacent-key substitution (QWERTY), deletion, duplication, transposition; words < 4 chars untouched; one word per query. Prefix: truncate the last word to 40–70% of its length, min 2 chars. Synonym: one term replaced from the reviewed per-scenario vocabulary-mismatch list; no applicable term → no variant.
- Perturbations are seeded; the same seed always yields byte-identical output. Variants inherit the source query's qrels.
- Modes: `keyword, semantic, hybrid, filtered, sorted, faceted`.
- Filtered ground truth: qrels ∩ docs matching the filter; filter chosen (seeded) among filters keeping ≥ 1 relevant doc.
- WANDS grades: Exact = 2, Partial = 1, Irrelevant = 0. Relevant means grade > 0.
- Embeddings are computed once per model, cached as Parquet keyed by text sha256; engines never embed.
- Generated data lives under `.irbench/` (gitignored). Raw datasets are never committed.
- Commits must NOT contain `Co-Authored-By` lines (user rule).

## Review Focus

1. **Query with no relevant judgment** (all Irrelevant / no labels) — ranx scores it 0, dragging averages down; it must be excluded from tasks and counted in prepare stats. Test: Task 9 `test_query_without_positives_is_excluded`.
2. **Engine returns the same doc ID twice** — the first (higher) rank must count, not the last. Test: Task 13 `test_duplicate_ids_keep_first_rank`.
3. **Engine returns nothing / errored for a task** — the task still counts and scores 0, never silently dropped from the mean. Test: Task 13 `test_missing_run_entry_scores_zero`.
4. **Embedding run interrupted midway** (API error, Ctrl-C) — re-running must reuse already-computed vectors, not pay again. Test: Task 12 `test_interrupted_embed_resumes_from_cache`.
5. **Docs with missing numeric fields** (9,452 WANDS products have no rating) — a missing value never satisfies a `gte`/`lte` filter, and in sorted results missing values may only appear after all present values. Tests: Task 9 `test_matches_missing_value_is_false`, Task 14 `test_order_missing_values_must_be_last`.

---

## File Structure

```
pyproject.toml                          # package, deps, pytest/ruff config
.gitignore
datasets/wands/dataset.yaml             # pinned URLs + sha256 + license
scenarios/ecommerce/product-search/
  scenario.yaml                         # schema, modes, templates, seed
  vocab_mismatch.txt                    # perturbation synonym groups
  synonyms.txt                          # engine-loadable synonym groups (disjoint from vocab_mismatch)
harness/irbench/
  __init__.py                           # __version__
  cli.py                                # typer app: version, prepare, embed, score
  paths.py                              # Workspace: .irbench/ layout
  models.py                             # Query, Filter, Sort, QueryTask, type aliases
  io.py                                 # jsonl, TREC qrels, TREC run read/write
  terms.py                              # term-group file parser
  scenario.py                           # Scenario model + loader
  filters.py                            # matches(doc, Filter)
  tasks.py                              # build_tasks: variants × modes → tasks + derived qrels
  manifest.py                           # git sha + manifest writer
  datasets/__init__.py                  # NORMALIZERS registry
  datasets/fetch.py                     # checksum-verified download
  datasets/registry.py                  # DatasetSpec loader + fetch_dataset
  datasets/wands.py                     # WANDS → docs/queries/qrels
  perturb/__init__.py
  perturb/keyboard.py                   # QWERTY neighbours
  perturb/typo.py
  perturb/prefix.py
  perturb/synonym.py
  perturb/variants.py                   # make_variants
  embed/__init__.py
  embed/base.py                         # Embedder protocol, render_doc_text
  embed/vectors.py                      # Parquet vector read/write
  embed/cache.py                        # VectorCache, text_key, embed_with_cache
  embed/models.py                       # sentence-transformers + OpenAI embedders, registry
  stages/__init__.py
  stages/prepare.py
  stages/embed.py
  score/__init__.py
  score/relevance.py                    # ranx grouping
  score/constraints.py                  # filter/order violations, facet accuracy
harness/tests/
  conftest.py                           # REPO_ROOT, mini WANDS fixture
  test_cli.py  test_io.py  test_fetch.py  test_wands.py  test_scenario.py
  test_typo.py test_prefix_synonym.py test_variants.py test_tasks.py
  test_prepare.py test_embed_cache.py test_embed_stage.py
  test_score_relevance.py test_constraints.py test_cli_e2e.py
```

---

### Task 1: Project scaffold and CLI skeleton

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `harness/irbench/__init__.py`, `harness/irbench/cli.py`
- Test: `harness/tests/test_cli.py`

**Interfaces:**
- Produces: `irbench.__version__: str = "0.1.0"`; `irbench.cli.app: typer.Typer` (later tasks add commands with `@app.command()`).

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "irbench"
version = "0.1.0"
description = "Neutral, reproducible information retrieval benchmark for search engines and embedding models"
requires-python = ">=3.12,<3.13"
dependencies = [
  "typer>=0.12",
  "pydantic>=2.7",
  "pyyaml>=6.0",
  "httpx>=0.27",
  "pyarrow>=16",
  "numpy>=1.26",
  "ranx>=0.3.21",
]

[project.optional-dependencies]
embed-local = ["sentence-transformers>=3.0"]
embed-api = ["openai>=1.40"]

[project.scripts]
irbench = "irbench.cli:app"

[dependency-groups]
dev = ["pytest>=8.2", "ruff>=0.6"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["harness/irbench"]

[tool.pytest.ini_options]
testpaths = ["harness/tests"]
markers = ["slow: needs network or large models"]
addopts = "-m 'not slow'"

[tool.ruff]
line-length = 100
target-version = "py312"
```

- [ ] **Step 2: Write `.gitignore`**

```
.irbench/
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
```

- [ ] **Step 3: Write the failing test** — `harness/tests/test_cli.py`

```python
from typer.testing import CliRunner

from irbench.cli import app


def test_version_command_prints_version():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"
```

- [ ] **Step 4: Run it to verify it fails**

Run: `uv sync && uv run pytest harness/tests/test_cli.py -v`
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'irbench'` or missing `cli`.

- [ ] **Step 5: Implement**

`harness/irbench/__init__.py`:
```python
__version__ = "0.1.0"
```

`harness/irbench/cli.py`:
```python
import typer

from irbench import __version__

app = typer.Typer(no_args_is_help=True, help="IRBench: information retrieval benchmark harness.")


@app.callback()
def main() -> None:
    """IRBench CLI."""


@app.command()
def version() -> None:
    """Print the harness version."""
    typer.echo(__version__)
```

- [ ] **Step 6: Run tests and lint**

Run: `uv run pytest -v && uv run ruff check harness`
Expected: 1 passed; ruff clean.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .gitignore harness/
git commit -m "feat: scaffold irbench package and CLI"
```

---

### Task 2: Core models and file formats

**Files:**
- Create: `harness/irbench/models.py`, `harness/irbench/io.py`
- Test: `harness/tests/test_io.py`

**Interfaces:**
- Produces:
  - `Perturbation = Literal["original","typo","prefix","synonym"]`, `Mode = Literal["keyword","semantic","hybrid","filtered","sorted","faceted"]`, `BaseMode = Literal["keyword","semantic","hybrid"]`, `Qrels = dict[str, dict[str, int]]`, `RunResults = dict[str, list[str]]` (task_id → ranked doc ids).
  - `Query(qid, text, variant_of, perturbation)`, `Filter(field, op: "eq"|"gte"|"lte", value: str|float)`, `Sort(field, direction: "asc"|"desc")`, `QueryTask(task_id, qid, variant_of, perturbation, text, mode, base, filter=None, sort=None, facets=[])`.
  - `write_jsonl(path, rows) -> int`, `read_jsonl(path) -> Iterator[dict]`, `write_qrels(path, qrels)`, `read_qrels(path) -> Qrels`, `write_run(path, results, tag)`, `read_run(path) -> RunResults`.

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_io.py`

```python
import pytest

from irbench.io import read_jsonl, read_qrels, read_run, write_jsonl, write_qrels, write_run
from irbench.models import Filter, QueryTask


def test_jsonl_roundtrip_keeps_unicode(tmp_path):
    rows = [{"id": "1", "name": "café table"}, {"id": "2", "name": "sofa"}]
    assert write_jsonl(tmp_path / "a" / "x.jsonl", rows) == 2
    assert list(read_jsonl(tmp_path / "a" / "x.jsonl")) == rows
    assert "café" in (tmp_path / "a" / "x.jsonl").read_text(encoding="utf-8")


def test_qrels_roundtrip_trec_format(tmp_path):
    qrels = {"q2": {"d9": 0}, "q1": {"d2": 1, "d1": 2}}
    write_qrels(tmp_path / "q.tsv", qrels)
    assert (tmp_path / "q.tsv").read_text().splitlines() == [
        "q1 0 d1 2",
        "q1 0 d2 1",
        "q2 0 d9 0",
    ]
    assert read_qrels(tmp_path / "q.tsv") == qrels


def test_run_roundtrip_preserves_rank_order(tmp_path):
    results = {"t1": ["d3", "d1", "d2"], "t0": ["d5"]}
    write_run(tmp_path / "r.trec", results, tag="meili")
    lines = (tmp_path / "r.trec").read_text().splitlines()
    assert lines[0] == "t0 Q0 d5 1 1 meili"
    assert lines[1] == "t1 Q0 d3 1 3 meili"
    assert read_run(tmp_path / "r.trec") == results


def test_ids_with_whitespace_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="whitespace"):
        write_qrels(tmp_path / "q.tsv", {"q 1": {"d1": 1}})
    with pytest.raises(ValueError, match="whitespace"):
        write_run(tmp_path / "r.trec", {"t1": ["d 1"]}, tag="x")


def test_query_task_json_roundtrip():
    task = QueryTask(
        task_id="1-typo:filtered:keyword",
        qid="1-typo",
        variant_of="1",
        perturbation="typo",
        text="platfrom bed",
        mode="filtered",
        base="keyword",
        filter=Filter(field="average_rating", op="gte", value=4.0),
    )
    assert QueryTask.model_validate(task.model_dump(mode="json")) == task


def test_filter_keeps_string_values_as_strings():
    assert Filter(field="product_class", op="eq", value="Beds").value == "Beds"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest harness/tests/test_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'irbench.io'`.

- [ ] **Step 3: Implement** — `harness/irbench/models.py`

```python
from typing import Literal

from pydantic import BaseModel, ConfigDict

Perturbation = Literal["original", "typo", "prefix", "synonym"]
Mode = Literal["keyword", "semantic", "hybrid", "filtered", "sorted", "faceted"]
BaseMode = Literal["keyword", "semantic", "hybrid"]
BASE_MODES: tuple[BaseMode, ...] = ("keyword", "semantic", "hybrid")

Qrels = dict[str, dict[str, int]]
RunResults = dict[str, list[str]]


class Query(BaseModel):
    model_config = ConfigDict(frozen=True)

    qid: str
    text: str
    variant_of: str
    perturbation: Perturbation


class Filter(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str
    op: Literal["eq", "gte", "lte"]
    value: str | float


class Sort(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str
    direction: Literal["asc", "desc"]


class QueryTask(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    qid: str
    variant_of: str
    perturbation: Perturbation
    text: str
    mode: Mode
    base: BaseMode
    filter: Filter | None = None
    sort: Sort | None = None
    facets: list[str] = []
```

`harness/irbench/io.py`:
```python
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from irbench.models import Qrels, RunResults


def _check_id(value: str) -> str:
    if not value or any(c.isspace() for c in value):
        raise ValueError(f"id must be non-empty and contain no whitespace: {value!r}")
    return value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_qrels(path: Path, qrels: Qrels) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid in sorted(qrels):
            for doc_id in sorted(qrels[qid]):
                f.write(f"{_check_id(qid)} 0 {_check_id(doc_id)} {qrels[qid][doc_id]}\n")


def read_qrels(path: Path) -> Qrels:
    qrels: Qrels = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qid, _, doc_id, grade = line.split()
            qrels.setdefault(qid, {})[doc_id] = int(grade)
    return qrels


def write_run(path: Path, results: RunResults, tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for task_id in sorted(results):
            ids = results[task_id]
            for rank, doc_id in enumerate(ids, start=1):
                score = len(ids) - rank + 1
                f.write(f"{_check_id(task_id)} Q0 {_check_id(doc_id)} {rank} {score} {tag}\n")


def read_run(path: Path) -> RunResults:
    ranked: dict[str, list[tuple[int, str]]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            task_id, _, doc_id, rank, _score, _tag = line.split()
            ranked.setdefault(task_id, []).append((int(rank), doc_id))
    return {tid: [doc for _, doc in sorted(pairs)] for tid, pairs in ranked.items()}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest harness/tests/test_io.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add harness/irbench/models.py harness/irbench/io.py harness/tests/test_io.py
git commit -m "feat: core models and jsonl/TREC file formats"
```

---

### Task 3: Workspace layout and checksum-verified dataset download

**Files:**
- Create: `harness/irbench/paths.py`, `harness/irbench/datasets/__init__.py` (empty for now), `harness/irbench/datasets/fetch.py`, `harness/irbench/datasets/registry.py`, `datasets/wands/dataset.yaml`
- Test: `harness/tests/test_fetch.py`

**Interfaces:**
- Produces:
  - `Workspace(root: Path)` with `.cache`, `.raw_dir(dataset_id)`, `.dataset_dir(dataset_id)`, `.scenario_dir(scenario_id)`, `.embeddings_dir(dataset_id, model_id)`.
  - `ChecksumError(RuntimeError)`, `sha256_file(path) -> str`, `fetch_file(url, dest, sha256, client: httpx.Client | None = None) -> Path`.
  - `FileSpec(url, sha256)`, `DatasetSpec(id, version, license, source, files: dict[str, FileSpec])`, `load_dataset_spec(root, dataset_id) -> DatasetSpec`, `fetch_dataset(spec, raw_dir, client=None) -> dict[str, Path]` (keys = `files` keys).

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_fetch.py`

```python
import hashlib
from pathlib import Path

import httpx
import pytest

from irbench.datasets.fetch import ChecksumError, fetch_file
from irbench.datasets.registry import load_dataset_spec
from irbench.paths import Workspace

REPO_ROOT = Path(__file__).resolve().parents[2]
BODY = b"product_id\tproduct_name\n0\tbed\n"
SHA = hashlib.sha256(BODY).hexdigest()


def _client(body: bytes, calls: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_downloads_and_verifies(tmp_path):
    calls: list[str] = []
    dest = fetch_file("https://x/p.csv", tmp_path / "raw" / "p.csv", SHA, _client(BODY, calls))
    assert dest.read_bytes() == BODY
    assert calls == ["https://x/p.csv"]


def test_fetch_skips_when_file_already_valid(tmp_path):
    dest = tmp_path / "p.csv"
    dest.write_bytes(BODY)
    calls: list[str] = []
    fetch_file("https://x/p.csv", dest, SHA, _client(BODY, calls))
    assert calls == []


def test_fetch_rejects_checksum_mismatch_and_leaves_no_file(tmp_path):
    dest = tmp_path / "p.csv"
    with pytest.raises(ChecksumError, match="expected sha256"):
        fetch_file("https://x/p.csv", dest, SHA, _client(b"tampered", []))
    assert not dest.exists()
    assert not (tmp_path / "p.csv.part").exists()


def test_wands_spec_is_pinned():
    spec = load_dataset_spec(REPO_ROOT, "wands")
    assert spec.license == "MIT"
    assert set(spec.files) == {"product", "query", "label"}
    for f in spec.files.values():
        assert "/main/" not in f.url
        assert len(f.sha256) == 64


def test_workspace_paths(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.raw_dir("wands") == tmp_path / ".irbench" / "data" / "wands" / "raw"
    assert ws.scenario_dir("ecommerce/product-search") == (
        tmp_path / ".irbench" / "scenarios" / "ecommerce__product-search"
    )
    assert ws.embeddings_dir("wands", "BAAI/bge") == (
        tmp_path / ".irbench" / "embeddings" / "wands" / "BAAI__bge"
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest harness/tests/test_fetch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'irbench.datasets'`.

- [ ] **Step 3: Implement**

`harness/irbench/paths.py`:
```python
from dataclasses import dataclass
from pathlib import Path


def _slug(value: str) -> str:
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
        return self.cache / "scenarios" / _slug(scenario_id)

    def embeddings_dir(self, dataset_id: str, model_id: str) -> Path:
        return self.cache / "embeddings" / dataset_id / _slug(model_id)
```

`harness/irbench/datasets/__init__.py`: empty file.

`harness/irbench/datasets/fetch.py`:
```python
import hashlib
from pathlib import Path

import httpx


class ChecksumError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_file(url: str, dest: Path, sha256: str, client: httpx.Client | None = None) -> Path:
    if dest.exists() and sha256_file(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    owned = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=120)
    try:
        with http.stream("GET", url) as response:
            response.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
    finally:
        if owned:
            http.close()
    actual = sha256_file(tmp)
    if actual != sha256:
        tmp.unlink()
        raise ChecksumError(f"{url}: expected sha256 {sha256}, got {actual}")
    tmp.replace(dest)
    return dest
```

`harness/irbench/datasets/registry.py`:
```python
from pathlib import Path
from urllib.parse import urlparse

import httpx
import yaml
from pydantic import BaseModel

from irbench.datasets.fetch import fetch_file


class FileSpec(BaseModel):
    url: str
    sha256: str


class DatasetSpec(BaseModel):
    id: str
    version: str
    license: str
    source: str
    files: dict[str, FileSpec]


def load_dataset_spec(root: Path, dataset_id: str) -> DatasetSpec:
    path = root / "datasets" / dataset_id / "dataset.yaml"
    spec = DatasetSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if spec.id != dataset_id:
        raise ValueError(f"{path}: id {spec.id!r} does not match directory {dataset_id!r}")
    return spec


def fetch_dataset(
    spec: DatasetSpec, raw_dir: Path, client: httpx.Client | None = None
) -> dict[str, Path]:
    return {
        name: fetch_file(f.url, raw_dir / Path(urlparse(f.url).path).name, f.sha256, client)
        for name, f in spec.files.items()
    }
```

`datasets/wands/dataset.yaml`:
```yaml
id: wands
version: "3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5"
license: MIT
source: https://github.com/wayfair/WANDS
files:
  product:
    url: https://github.com/wayfair/WANDS/raw/3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5/dataset/product.csv
    sha256: d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6
  query:
    url: https://github.com/wayfair/WANDS/raw/3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5/dataset/query.csv
    sha256: 63b61660560fecc33ec490804c7e2b81402ee3e7c31a9cbb5e03736639f68e95
  label:
    url: https://github.com/wayfair/WANDS/raw/3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5/dataset/label.csv
    sha256: c11fe81ad62f17f56f316b0ec9630ebe8fbe1393578cb0ca4f05c17253a180ef
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest harness/tests/test_fetch.py -v`
Expected: 5 passed.

- [ ] **Step 5: Verify the real pinned files once (network)**

Run:
```bash
uv run python -c "
from pathlib import Path
from irbench.datasets.registry import load_dataset_spec, fetch_dataset
from irbench.paths import Workspace
ws = Workspace(Path('.'))
print(fetch_dataset(load_dataset_spec(Path('.'), 'wands'), ws.raw_dir('wands')))"
```
Expected: three paths printed under `.irbench/data/wands/raw/`, no `ChecksumError`.

- [ ] **Step 6: Commit**

```bash
git add harness/irbench/paths.py harness/irbench/datasets datasets/ harness/tests/test_fetch.py
git commit -m "feat: checksum-verified dataset download and pinned WANDS spec"
```

---

### Task 4: WANDS normalizer

**Files:**
- Create: `harness/irbench/datasets/wands.py`, `harness/tests/conftest.py`
- Modify: `harness/irbench/datasets/__init__.py`
- Test: `harness/tests/test_wands.py`

**Interfaces:**
- Consumes: `write_jsonl`, `write_qrels` (Task 2).
- Produces:
  - `NormalizedDataset(docs: Path, queries: Path, qrels: Path, n_docs: int, n_queries: int)` dataclass.
  - `normalize_wands(raw: dict[str, Path], out_dir: Path) -> NormalizedDataset` — `raw` keys `product`, `query`, `label`.
  - `NORMALIZERS: dict[str, Callable[[dict[str, Path], Path], NormalizedDataset]]` in `irbench.datasets`.
  - Doc shape: `{"id": str, "name": str, "product_class": list[str], "category": list[str], "description": str, "features": list[str], "rating_count": float|None, "average_rating": float|None, "review_count": float|None}`.
  - Query shape: `{"qid": str, "text": str, "query_class": str}`.
  - conftest fixtures: `REPO_ROOT: Path`, `wands_raw(tmp_path) -> dict[str, Path]` (mini WANDS used by Tasks 4, 10, 15).

- [ ] **Step 1: Write the shared fixture** — `harness/tests/conftest.py`

```python
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

PRODUCT_TSV = (
    "product_id\tproduct_name\tproduct_class\tcategory hierarchy\tproduct_description"
    "\tproduct_features\trating_count\taverage_rating\treview_count\n"
    '0\tsolid wood platform bed\tBeds\tFurniture / Bedroom Furniture / Beds\ta "sturdy" bed'
    "\tcolor : brown|material : wood\t15.0\t4.5\t15.0\n"
    "1\tgrey velvet sofa\tSofas|Sectionals\tFurniture / Living Room Furniture / Sofas"
    "\tsoft sofa\tcolor : grey\t\t\t\n"
    "2\toak coffee table\tCoffee & Cocktail Tables\tFurniture / Living Room Furniture / Tables"
    "\tround table\tmaterial : oak\t3.0\t3.0\t3.0\n"
    "3\tmodern nightstand\tNightstands\tFurniture / Bedroom Furniture / Nightstands"
    "\ttwo drawers\t\t8.0\t4.0\t8.0\n"
)
QUERY_TSV = (
    "query_id\tquery\tquery_class\n"
    "0\tplatform bed\tBeds\n"
    "1\tvelvet couch\tSofas\n"
    "2\tsalon chair\tMassage Chairs\n"
)
LABEL_TSV = (
    "id\tquery_id\tproduct_id\tlabel\n"
    "0\t0\t0\tExact\n"
    "1\t0\t3\tPartial\n"
    "2\t1\t1\tExact\n"
    "3\t1\t2\tIrrelevant\n"
    "4\t2\t2\tIrrelevant\n"
)


@pytest.fixture
def wands_raw(tmp_path: Path) -> dict[str, Path]:
    raw = tmp_path / "wands_raw"
    raw.mkdir()
    files = {"product": PRODUCT_TSV, "query": QUERY_TSV, "label": LABEL_TSV}
    out = {}
    for name, body in files.items():
        path = raw / f"{name}.csv"
        path.write_text(body, encoding="utf-8")
        out[name] = path
    return out
```

- [ ] **Step 2: Write the failing tests** — `harness/tests/test_wands.py`

```python
import pytest

from irbench.datasets import NORMALIZERS
from irbench.io import read_jsonl, read_qrels


def test_normalize_wands_docs(wands_raw, tmp_path):
    result = NORMALIZERS["wands"](wands_raw, tmp_path / "out")
    docs = {d["id"]: d for d in read_jsonl(result.docs)}
    assert result.n_docs == 4
    assert docs["0"] == {
        "id": "0",
        "name": "solid wood platform bed",
        "product_class": ["Beds"],
        "category": ["Furniture", "Bedroom Furniture", "Beds"],
        "description": 'a "sturdy" bed',
        "features": ["color : brown", "material : wood"],
        "rating_count": 15.0,
        "average_rating": 4.5,
        "review_count": 15.0,
    }
    assert docs["1"]["product_class"] == ["Sofas", "Sectionals"]
    assert docs["1"]["average_rating"] is None
    assert docs["3"]["features"] == []


def test_normalize_wands_queries_and_qrels(wands_raw, tmp_path):
    result = NORMALIZERS["wands"](wands_raw, tmp_path / "out")
    queries = list(read_jsonl(result.queries))
    assert queries[0] == {"qid": "0", "text": "platform bed", "query_class": "Beds"}
    assert result.n_queries == 3
    assert read_qrels(result.qrels) == {
        "0": {"0": 2, "3": 1},
        "1": {"1": 2, "2": 0},
        "2": {"2": 0},
    }


def test_unknown_label_is_an_error(wands_raw, tmp_path):
    wands_raw["label"].write_text("id\tquery_id\tproduct_id\tlabel\n0\t0\t0\tMaybe\n")
    with pytest.raises(ValueError, match="Maybe"):
        NORMALIZERS["wands"](wands_raw, tmp_path / "out")
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest harness/tests/test_wands.py -v`
Expected: FAIL — `ImportError: cannot import name 'NORMALIZERS'`.

- [ ] **Step 4: Implement** — `harness/irbench/datasets/wands.py`

```python
import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from irbench.io import write_jsonl, write_qrels
from irbench.models import Qrels

csv.field_size_limit(1 << 30)

GRADES = {"Exact": 2, "Partial": 1, "Irrelevant": 0}


@dataclass(frozen=True)
class NormalizedDataset:
    docs: Path
    queries: Path
    qrels: Path
    n_docs: int
    n_queries: int


def _read_tsv(path: Path) -> Iterator[dict[str, str]]:
    # WANDS is tab-separated with unescaped quotes inside text: disable quoting.
    with path.open(encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)


def _split(value: str, sep: str) -> list[str]:
    return [part.strip() for part in value.split(sep) if part.strip()]


def _float_or_none(value: str) -> float | None:
    value = value.strip()
    return float(value) if value else None


def _product_to_doc(row: dict[str, str]) -> dict[str, Any]:
    return {
        "id": row["product_id"].strip(),
        "name": row["product_name"].strip(),
        "product_class": _split(row["product_class"], "|"),
        "category": _split(row["category hierarchy"], " / "),
        "description": row["product_description"].strip(),
        "features": _split(row["product_features"], "|"),
        "rating_count": _float_or_none(row["rating_count"]),
        "average_rating": _float_or_none(row["average_rating"]),
        "review_count": _float_or_none(row["review_count"]),
    }


def normalize_wands(raw: dict[str, Path], out_dir: Path) -> NormalizedDataset:
    docs_path = out_dir / "docs.jsonl"
    queries_path = out_dir / "queries.jsonl"
    qrels_path = out_dir / "qrels.tsv"

    n_docs = write_jsonl(docs_path, (_product_to_doc(r) for r in _read_tsv(raw["product"])))
    n_queries = write_jsonl(
        queries_path,
        (
            {
                "qid": r["query_id"].strip(),
                "text": " ".join(r["query"].split()),
                "query_class": r["query_class"].strip(),
            }
            for r in _read_tsv(raw["query"])
        ),
    )

    qrels: Qrels = {}
    for r in _read_tsv(raw["label"]):
        label = r["label"].strip()
        if label not in GRADES:
            raise ValueError(f"unknown WANDS label {label!r} (row id {r['id']})")
        qrels.setdefault(r["query_id"].strip(), {})[r["product_id"].strip()] = GRADES[label]
    write_qrels(qrels_path, qrels)

    return NormalizedDataset(docs_path, queries_path, qrels_path, n_docs, n_queries)
```

`harness/irbench/datasets/__init__.py`:
```python
from collections.abc import Callable
from pathlib import Path

from irbench.datasets.wands import NormalizedDataset, normalize_wands

NORMALIZERS: dict[str, Callable[[dict[str, Path], Path], NormalizedDataset]] = {
    "wands": normalize_wands,
}
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest harness/tests/test_wands.py -v`
Expected: 3 passed.

- [ ] **Step 6: Verify against the real file (uses the download from Task 3)**

Run:
```bash
uv run python -c "
from pathlib import Path
from irbench.datasets import NORMALIZERS
raw = Path('.irbench/data/wands/raw')
r = NORMALIZERS['wands']({'product': raw/'product.csv','query': raw/'query.csv','label': raw/'label.csv'}, Path('.irbench/data/wands'))
print(r.n_docs, r.n_queries)"
```
Expected: `42994 480`.

- [ ] **Step 7: Commit**

```bash
git add harness/irbench/datasets harness/tests/conftest.py harness/tests/test_wands.py
git commit -m "feat: normalize WANDS into docs, queries and graded qrels"
```

---

### Task 5: Scenario definition and term-group files

**Files:**
- Create: `harness/irbench/terms.py`, `harness/irbench/scenario.py`, `scenarios/ecommerce/product-search/scenario.yaml`, `scenarios/ecommerce/product-search/vocab_mismatch.txt`, `scenarios/ecommerce/product-search/synonyms.txt`
- Test: `harness/tests/test_scenario.py`

**Interfaces:**
- Consumes: `Mode`, `BaseMode`, `Sort` (Task 2).
- Produces:
  - `load_term_groups(path) -> list[list[str]]` — one comma-separated group per line, `#` comments, lowercased; raises `ValueError` if a term appears in two groups or a group has < 2 terms.
  - `IndexFields(searchable, filterable, sortable, facets: list[str])`, `FilterTemplate(field, op, values: list[float] | None)`, `PerturbationSpec(seed: int, kinds: list[...], vocabulary: str)`.
  - `Scenario(id, dataset, id_field, index_fields, modes, structured_bases, embedding_fields, filter_templates, sort_templates, top_k, index_timeout_s, perturbations, synonyms)` with properties `directory: Path`, `vocabulary_path: Path`, `synonyms_path: Path`.
  - `load_scenario(root, scenario_id) -> Scenario`.

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_scenario.py`

```python
import pytest
import yaml

from irbench.scenario import load_scenario
from irbench.terms import load_term_groups

from conftest import REPO_ROOT

SCENARIO = "ecommerce/product-search"


def test_load_product_search_scenario():
    s = load_scenario(REPO_ROOT, SCENARIO)
    assert s.dataset == "wands"
    assert s.modes == ["keyword", "semantic", "hybrid", "filtered", "sorted", "faceted"]
    assert s.index_fields.searchable[0] == "name"
    assert s.vocabulary_path.exists()
    assert s.synonyms_path.exists()


def test_term_groups_parse(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("# comment\nSofa, couch , settee\n\ntv stand, media console\n")
    assert load_term_groups(path) == [["sofa", "couch", "settee"], ["tv stand", "media console"]]


def test_term_in_two_groups_is_rejected(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("sofa, couch\ncouch, settee\n")
    with pytest.raises(ValueError, match="couch"):
        load_term_groups(path)


def test_vocab_mismatch_and_engine_synonyms_are_disjoint():
    # The synonyms file engines may load must not be the answer key for the
    # synonym perturbation, otherwise that perturbation measures nothing.
    s = load_scenario(REPO_ROOT, SCENARIO)
    vocab = {t for g in load_term_groups(s.vocabulary_path) for t in g}
    engine = {t for g in load_term_groups(s.synonyms_path) for t in g}
    assert vocab.isdisjoint(engine), vocab & engine


def test_filter_template_must_target_filterable_field(tmp_path):
    raw = yaml.safe_load((REPO_ROOT / "scenarios" / SCENARIO / "scenario.yaml").read_text())
    raw["filter_templates"].append({"field": "description", "op": "eq"})
    target = tmp_path / "scenarios" / SCENARIO
    target.mkdir(parents=True)
    (target / "scenario.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="description"):
        load_scenario(tmp_path, SCENARIO)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest harness/tests/test_scenario.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'irbench.scenario'`.

- [ ] **Step 3: Implement** — `harness/irbench/terms.py`

```python
from pathlib import Path


def load_term_groups(path: Path) -> list[list[str]]:
    groups: list[list[str]] = []
    seen: dict[str, int] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        group = [" ".join(t.split()).lower() for t in line.split(",") if t.strip()]
        if len(group) < 2:
            raise ValueError(f"{path}:{lineno}: a group needs at least two terms")
        for term in group:
            if term in seen:
                raise ValueError(
                    f"{path}:{lineno}: term {term!r} already used on line {seen[term]}"
                )
            seen[term] = lineno
        groups.append(group)
    return groups
```

`harness/irbench/scenario.py`:
```python
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, PrivateAttr, model_validator

from irbench.models import BaseMode, Mode, Sort


class IndexFields(BaseModel):
    searchable: list[str]
    filterable: list[str]
    sortable: list[str]
    facets: list[str]


class FilterTemplate(BaseModel):
    field: str
    op: Literal["eq", "gte", "lte"]
    values: list[float] | None = None


class PerturbationSpec(BaseModel):
    seed: int
    kinds: list[Literal["typo", "prefix", "synonym"]]
    vocabulary: str


class Scenario(BaseModel):
    id: str
    dataset: str
    id_field: str = "id"
    index_fields: IndexFields
    modes: list[Mode]
    structured_bases: list[BaseMode]
    embedding_fields: list[str]
    filter_templates: list[FilterTemplate]
    sort_templates: list[Sort]
    top_k: int = 100
    index_timeout_s: int
    perturbations: PerturbationSpec
    synonyms: str

    _directory: Path = PrivateAttr()

    @model_validator(mode="after")
    def _check_consistency(self) -> "Scenario":
        f = self.index_fields
        for t in self.filter_templates:
            if t.field not in f.filterable:
                raise ValueError(f"filter template field {t.field!r} is not filterable")
            if (t.op == "eq") != (t.values is None):
                raise ValueError(f"filter template {t.field!r}: eq takes no values, ranges need values")
        for s in self.sort_templates:
            if s.field not in f.sortable:
                raise ValueError(f"sort template field {s.field!r} is not sortable")
        if "faceted" in self.modes and not f.facets:
            raise ValueError("faceted mode needs at least one facet field")
        return self

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def vocabulary_path(self) -> Path:
        return self._directory / self.perturbations.vocabulary

    @property
    def synonyms_path(self) -> Path:
        return self._directory / self.synonyms


def load_scenario(root: Path, scenario_id: str) -> Scenario:
    directory = root / "scenarios" / scenario_id
    path = directory / "scenario.yaml"
    scenario = Scenario.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if scenario.id != scenario_id:
        raise ValueError(f"{path}: id {scenario.id!r} does not match {scenario_id!r}")
    scenario._directory = directory
    return scenario
```

`scenarios/ecommerce/product-search/scenario.yaml`:
```yaml
id: ecommerce/product-search
dataset: wands
id_field: id

# Order of `searchable` is the field priority every engine must respect.
index_fields:
  searchable: [name, product_class, category, features, description]
  filterable: [product_class, category, average_rating, review_count]
  sortable: [average_rating, review_count]
  facets: [product_class]

modes: [keyword, semantic, hybrid, filtered, sorted, faceted]
# filtered/sorted/faceted run on top of each of these retrieval bases.
structured_bases: [keyword, semantic]

# Doc text sent to embedding models: these fields, joined with ". ", empty ones skipped.
embedding_fields: [name, product_class, description]

filter_templates:
  - {field: product_class, op: eq}
  - {field: average_rating, op: gte, values: [3.0, 4.0, 4.5]}

sort_templates:
  - {field: average_rating, direction: desc}
  - {field: review_count, direction: desc}

top_k: 100
index_timeout_s: 1800

perturbations:
  seed: 20261009
  kinds: [typo, prefix, synonym]
  vocabulary: vocab_mismatch.txt

synonyms: synonyms.txt
```

`scenarios/ecommerce/product-search/vocab_mismatch.txt`:
```
# Vocabulary-mismatch groups used to generate `synonym` query variants.
# Shoppers use one word, the catalog uses another. Reviewed by hand.
# Must stay disjoint from synonyms.txt (enforced by tests).
sofa, couch, settee
armchair, accent chair
nightstand, bedside table
dresser, chest of drawers
wardrobe, armoire
rug, carpet
ottoman, footstool, pouf
bookcase, bookshelf
tv stand, media console, entertainment center
comforter, duvet
drapes, curtains
sconce, wall light
chandelier, ceiling light
vanity, dressing table
loveseat, two seater sofa
sectional, corner sofa
recliner, lounger
barstool, counter stool
faucet, tap
sink, basin
crib, cot
office chair, desk chair
coffee table, cocktail table
end table, side table
console table, entryway table
mattress topper, mattress pad
throw pillow, cushion
```

`scenarios/ecommerce/product-search/synonyms.txt`:
```
# Synonyms an engine MAY load into its synonym feature (shared by all engines).
# Typical merchant configuration: spelling variants and number words.
# Must stay disjoint from vocab_mismatch.txt (enforced by tests).
grey, gray
colour, color
microfibre, microfiber
television, tv
bathtub, tub
three seater, 3 seater
two drawer, 2 drawer
kids, children
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest harness/tests/test_scenario.py -v`
Expected: 5 passed. (Pytest's rootdir-based import makes `from conftest import REPO_ROOT` work; if it doesn't, add `pythonpath = ["harness/tests"]` under `[tool.pytest.ini_options]`.)

- [ ] **Step 5: Commit**

```bash
git add harness/irbench/terms.py harness/irbench/scenario.py scenarios/ harness/tests/test_scenario.py
git commit -m "feat: ecommerce/product-search scenario and term-group files"
```

---

### Task 6: Typo perturbation

**Files:**
- Create: `harness/irbench/perturb/__init__.py` (empty), `harness/irbench/perturb/keyboard.py`, `harness/irbench/perturb/typo.py`
- Test: `harness/tests/test_typo.py`

**Interfaces:**
- Produces: `QWERTY_NEIGHBORS: dict[str, list[str]]`; `typo(text: str, rng: random.Random) -> str | None` (None when no word is eligible).

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_typo.py`

```python
import random

from irbench.perturb.keyboard import QWERTY_NEIGHBORS
from irbench.perturb.typo import typo


def _edit_distance(a: str, b: str) -> int:
    # Optimal string alignment distance (counts a transposition as 1 edit).
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


def test_neighbors_follow_keyboard_layout():
    assert QWERTY_NEIGHBORS["f"] == ["c", "d", "g", "r", "t", "v"]
    assert QWERTY_NEIGHBORS["a"] == ["q", "s", "w", "z"]


def test_same_seed_same_output():
    outs = {typo("solid wood platform bed", random.Random("s:1:typo")) for _ in range(5)}
    assert len(outs) == 1


def test_only_one_word_changes_and_short_words_untouched():
    for seed in range(200):
        out = typo("a tv platform bed", random.Random(seed))
        words = out.split()
        assert words[0] == "a" and words[1] == "tv"
        changed = [i for i, (x, y) in enumerate(zip(words, "a tv platform bed".split())) if x != y]
        assert len(changed) == 1


def test_edit_budget_by_word_length():
    for seed in range(300):
        short = typo("table", random.Random(seed))  # 5 chars -> 1 edit
        assert 1 <= _edit_distance(short, "table") <= 1
        long = typo("nightstand", random.Random(seed))  # 10 chars -> 2 edits
        assert 1 <= _edit_distance(long, "nightstand") <= 2
        assert long != "nightstand"


def test_no_eligible_word_returns_none():
    assert typo("tv 3d", random.Random(1)) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest harness/tests/test_typo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'irbench.perturb'`.

- [ ] **Step 3: Implement**

`harness/irbench/perturb/keyboard.py`:
```python
ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")

# Row offsets for a staggered QWERTY layout: the row above is shifted right,
# the row below is shifted left, relative to the key's index in its row.
_OFFSETS = ((0, (-1, 1)), (-1, (0, 1)), (1, (-1, 0)))


def _build() -> dict[str, list[str]]:
    neighbors: dict[str, list[str]] = {}
    for r, row in enumerate(ROWS):
        for i, key in enumerate(row):
            found: set[str] = set()
            for dr, offsets in _OFFSETS:
                rr = r + dr
                if 0 <= rr < len(ROWS):
                    for d in offsets:
                        j = i + d
                        if 0 <= j < len(ROWS[rr]):
                            found.add(ROWS[rr][j])
            neighbors[key] = sorted(found)
    return neighbors


QWERTY_NEIGHBORS = _build()
```

`harness/irbench/perturb/typo.py`:
```python
import random

from irbench.perturb.keyboard import QWERTY_NEIGHBORS

EDIT_KINDS = ("substitute", "delete", "duplicate", "transpose")
MIN_WORD_LEN = 4
TWO_EDITS_FROM_LEN = 8
MAX_ATTEMPTS = 10


def _edit(word: str, rng: random.Random) -> str:
    i = rng.randrange(len(word))
    kind = rng.choice(EDIT_KINDS)
    if kind == "substitute":
        neighbors = QWERTY_NEIGHBORS.get(word[i].lower())
        if neighbors:
            return word[:i] + rng.choice(neighbors) + word[i + 1 :]
        kind = "delete"
    if kind == "delete":
        return word[:i] + word[i + 1 :]
    if kind == "duplicate":
        return word[:i] + word[i] + word[i:]
    if i == len(word) - 1:
        i -= 1
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def typo(text: str, rng: random.Random) -> str | None:
    words = text.split()
    eligible = [i for i, w in enumerate(words) if len(w) >= MIN_WORD_LEN and w.isalpha()]
    if not eligible:
        return None
    index = rng.choice(eligible)
    word = words[index]
    edits = 1 if len(word) < TWO_EDITS_FROM_LEN else 2
    for _ in range(MAX_ATTEMPTS):
        candidate = word
        for _ in range(edits):
            candidate = _edit(candidate, rng)
        if candidate != word:
            words[index] = candidate
            return " ".join(words)
    return None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest harness/tests/test_typo.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add harness/irbench/perturb harness/tests/test_typo.py
git commit -m "feat: seeded keyboard-aware typo perturbation"
```

---

### Task 7: Prefix and synonym perturbations

**Files:**
- Create: `harness/irbench/perturb/prefix.py`, `harness/irbench/perturb/synonym.py`
- Test: `harness/tests/test_prefix_synonym.py`

**Interfaces:**
- Produces: `prefix(text, rng) -> str | None`; `synonym(text, rng, groups: list[list[str]]) -> str | None` (returns lowercased text; WANDS queries are already lowercase).

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_prefix_synonym.py`

```python
import random

from irbench.perturb.prefix import prefix
from irbench.perturb.synonym import synonym

GROUPS = [["sofa", "couch", "settee"], ["coffee table", "cocktail table"], ["rug", "carpet"]]


def test_prefix_truncates_last_word_within_bounds():
    for seed in range(300):
        out = prefix("solid wood nightstand", random.Random(seed))
        head, last = out.rsplit(" ", 1)
        assert head == "solid wood"
        assert "nightstand".startswith(last)
        assert round(10 * 0.4) <= len(last) <= round(10 * 0.7)


def test_prefix_min_two_chars_and_short_word_none():
    assert prefix("big bed", random.Random(3)) == "big be"
    assert prefix("a tv", random.Random(3)) is None


def test_synonym_replaces_whole_words_only():
    out = synonym("rugged sofa", random.Random(1), GROUPS)
    assert out in {"rugged couch", "rugged settee"}


def test_synonym_prefers_longest_phrase():
    for seed in range(20):
        assert synonym("oak coffee table", random.Random(seed), GROUPS) == "oak cocktail table"


def test_synonym_no_match_returns_none():
    assert synonym("platform bed", random.Random(1), GROUPS) is None


def test_synonym_is_deterministic():
    a = synonym("sofa with rug", random.Random("x"), GROUPS)
    b = synonym("sofa with rug", random.Random("x"), GROUPS)
    assert a == b
```

Note on `test_prefix_min_two_chars_and_short_word_none`: "bed" has 3 chars → keep = max(2, round(3 × u)) with u ∈ [0.4, 0.7] → 2, so the output is always "big be".

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest harness/tests/test_prefix_synonym.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`harness/irbench/perturb/prefix.py`:
```python
import random

MIN_KEEP = 2
MIN_WORD_LEN = 3


def prefix(text: str, rng: random.Random) -> str | None:
    words = text.split()
    if not words or len(words[-1]) < MIN_WORD_LEN:
        return None
    last = words[-1]
    keep = max(MIN_KEEP, round(len(last) * rng.uniform(0.4, 0.7)))
    if keep >= len(last):
        return None
    words[-1] = last[:keep]
    return " ".join(words)
```

`harness/irbench/perturb/synonym.py`:
```python
import random
import re


def _pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")


def synonym(text: str, rng: random.Random, groups: list[list[str]]) -> str | None:
    lowered = " ".join(text.lower().split())
    matches = [(term, group) for group in groups for term in group if _pattern(term).search(lowered)]
    if not matches:
        return None
    longest = max(len(term) for term, _ in matches)
    candidates = sorted((m for m in matches if len(m[0]) == longest), key=lambda m: m[0])
    term, group = rng.choice(candidates)
    replacement = rng.choice([t for t in group if t != term])
    return _pattern(term).sub(replacement, lowered, count=1)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest harness/tests/test_prefix_synonym.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add harness/irbench/perturb/prefix.py harness/irbench/perturb/synonym.py harness/tests/test_prefix_synonym.py
git commit -m "feat: prefix and vocabulary-mismatch synonym perturbations"
```

---

### Task 8: Variant generation

**Files:**
- Create: `harness/irbench/perturb/variants.py`
- Test: `harness/tests/test_variants.py`

**Interfaces:**
- Consumes: `Query` (Task 2), `typo` (Task 6), `prefix`, `synonym` (Task 7).
- Produces: `make_variants(queries: list[Query], seed: int, kinds: list[str], vocabulary: list[list[str]]) -> list[Query]` — returns originals plus variants with `qid=f"{source}-{kind}"`, `variant_of=source`, in input order; a variant whose text equals the original (case/whitespace-insensitive) is dropped. RNG per variant: `random.Random(f"{seed}:{qid}:{kind}")`.

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_variants.py`

```python
from irbench.models import Query
from irbench.perturb.variants import make_variants

VOCAB = [["sofa", "couch"]]


def _q(qid: str, text: str) -> Query:
    return Query(qid=qid, text=text, variant_of=qid, perturbation="original")


def test_variants_are_linked_to_source():
    out = make_variants([_q("7", "velvet sofa")], 1, ["typo", "prefix", "synonym"], VOCAB)
    assert [q.qid for q in out] == ["7", "7-typo", "7-prefix", "7-synonym"]
    assert all(q.variant_of == "7" for q in out)
    assert out[3].text == "velvet couch"


def test_variants_are_byte_identical_across_runs():
    queries = [_q(str(i), t) for i, t in enumerate(["platform bed", "grey velvet sofa", "rug"])]
    a = make_variants(queries, 42, ["typo", "prefix", "synonym"], VOCAB)
    b = make_variants(queries, 42, ["typo", "prefix", "synonym"], VOCAB)
    assert [q.model_dump() for q in a] == [q.model_dump() for q in b]


def test_inapplicable_perturbations_produce_no_variant():
    out = make_variants([_q("1", "tv")], 1, ["typo", "prefix", "synonym"], VOCAB)
    assert [q.qid for q in out] == ["1"]


def test_seed_changes_output():
    q = [_q("1", "upholstered platform nightstand")]
    texts = {make_variants(q, s, ["typo"], VOCAB)[1].text for s in range(20)}
    assert len(texts) > 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest harness/tests/test_variants.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `harness/irbench/perturb/variants.py`

```python
import random

from irbench.models import Query
from irbench.perturb.prefix import prefix
from irbench.perturb.synonym import synonym
from irbench.perturb.typo import typo


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _apply(kind: str, text: str, rng: random.Random, vocabulary: list[list[str]]) -> str | None:
    if kind == "typo":
        return typo(text, rng)
    if kind == "prefix":
        return prefix(text, rng)
    if kind == "synonym":
        return synonym(text, rng, vocabulary)
    raise ValueError(f"unknown perturbation kind {kind!r}")


def make_variants(
    queries: list[Query], seed: int, kinds: list[str], vocabulary: list[list[str]]
) -> list[Query]:
    out: list[Query] = []
    for query in queries:
        out.append(query)
        for kind in kinds:
            rng = random.Random(f"{seed}:{query.qid}:{kind}")
            text = _apply(kind, query.text, rng, vocabulary)
            if text is None or _norm(text) == _norm(query.text):
                continue
            out.append(
                Query(qid=f"{query.qid}-{kind}", text=text, variant_of=query.qid, perturbation=kind)
            )
    return out
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest harness/tests/test_variants.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add harness/irbench/perturb/variants.py harness/tests/test_variants.py
git commit -m "feat: seeded query variant generation"
```

---

### Task 9: Filter evaluation and per-mode task building

**Files:**
- Create: `harness/irbench/filters.py`, `harness/irbench/tasks.py`
- Test: `harness/tests/test_tasks.py`

**Interfaces:**
- Consumes: `Query`, `QueryTask`, `Filter`, `Sort`, `Qrels`, `BASE_MODES` (Task 2); `Scenario` (Task 5).
- Produces:
  - `matches(doc: dict, f: Filter) -> bool` — list fields match on membership; missing/None never matches a range.
  - `TaskSet(tasks: list[QueryTask], qrels: Qrels, skipped_queries: list[str])` dataclass.
  - `build_tasks(scenario, variants: list[Query], qrels: Qrels, docs_by_id: dict[str, dict]) -> TaskSet`.
  - Task id format: `f"{variant_qid}:{mode}:{base}"`.

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_tasks.py`

```python
from irbench.filters import matches
from irbench.models import Filter, Query
from irbench.scenario import load_scenario
from irbench.tasks import build_tasks

from conftest import REPO_ROOT

DOCS = {
    "0": {"id": "0", "product_class": ["Beds"], "average_rating": 4.5, "review_count": 15.0},
    "3": {"id": "3", "product_class": ["Nightstands"], "average_rating": 4.0, "review_count": 8.0},
    "1": {"id": "1", "product_class": ["Sofas", "Sectionals"], "average_rating": None},
    "2": {"id": "2", "product_class": ["Coffee & Cocktail Tables"], "average_rating": 3.0},
}
QRELS = {"0": {"0": 2, "3": 1}, "1": {"1": 2, "2": 0}, "2": {"2": 0}}


def _variants():
    return [
        Query(qid="0", text="platform bed", variant_of="0", perturbation="original"),
        Query(qid="0-typo", text="platfrom bed", variant_of="0", perturbation="typo"),
        Query(qid="1", text="velvet couch", variant_of="1", perturbation="original"),
        Query(qid="2", text="salon chair", variant_of="2", perturbation="original"),
    ]


def _build():
    return build_tasks(load_scenario(REPO_ROOT, "ecommerce/product-search"), _variants(), QRELS, DOCS)


def test_matches_eq_on_list_and_ranges():
    assert matches(DOCS["1"], Filter(field="product_class", op="eq", value="Sectionals"))
    assert not matches(DOCS["0"], Filter(field="product_class", op="eq", value="Sofas"))
    assert matches(DOCS["0"], Filter(field="average_rating", op="gte", value=4.5))
    assert matches(DOCS["2"], Filter(field="average_rating", op="lte", value=3.0))


def test_matches_missing_value_is_false():
    assert not matches(DOCS["1"], Filter(field="average_rating", op="gte", value=0.0))
    assert not matches(DOCS["1"], Filter(field="average_rating", op="lte", value=5.0))
    assert not matches({"id": "9"}, Filter(field="product_class", op="eq", value="Beds"))


def test_query_without_positives_is_excluded():
    result = _build()
    assert result.skipped_queries == ["2"]
    assert not any(t.variant_of == "2" for t in result.tasks)


def test_task_ids_and_mode_expansion():
    result = _build()
    ids = {t.task_id for t in result.tasks if t.qid == "0"}
    assert {"0:keyword:keyword", "0:semantic:semantic", "0:hybrid:hybrid"} <= ids
    assert {"0:filtered:keyword", "0:filtered:semantic", "0:sorted:keyword", "0:faceted:semantic"} <= ids
    faceted = next(t for t in result.tasks if t.task_id == "0:faceted:keyword")
    assert faceted.facets == ["product_class"]


def test_filtered_qrels_are_intersection_and_keep_a_positive():
    result = _build()
    for t in result.tasks:
        if t.mode != "filtered":
            continue
        grades = result.qrels[t.task_id]
        assert any(g > 0 for g in grades.values())
        assert all(matches(DOCS[d], t.filter) for d in grades)


def test_variants_share_the_source_filter_and_sort():
    result = _build()
    by_id = {t.task_id: t for t in result.tasks}
    assert by_id["0:filtered:keyword"].filter == by_id["0-typo:filtered:keyword"].filter
    assert by_id["0:sorted:keyword"].sort == by_id["0-typo:sorted:keyword"].sort


def test_unfiltered_modes_keep_source_qrels():
    result = _build()
    assert result.qrels["0-typo:keyword:keyword"] == QRELS["0"]
    assert result.qrels["0:sorted:semantic"] == QRELS["0"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest harness/tests/test_tasks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'irbench.filters'`.

- [ ] **Step 3: Implement**

`harness/irbench/filters.py`:
```python
from typing import Any

from irbench.models import Filter


def field_values(doc: dict[str, Any], field: str) -> list[Any]:
    value = doc.get(field)
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def matches(doc: dict[str, Any], f: Filter) -> bool:
    values = field_values(doc, f.field)
    if f.op == "eq":
        return f.value in values
    numeric = [v for v in values if isinstance(v, int | float)]
    if f.op == "gte":
        return any(v >= f.value for v in numeric)
    return any(v <= f.value for v in numeric)
```

`harness/irbench/tasks.py`:
```python
import random
from dataclasses import dataclass, field
from typing import Any

from irbench.filters import field_values, matches
from irbench.models import BASE_MODES, Filter, Qrels, Query, QueryTask, Sort
from irbench.scenario import FilterTemplate, Scenario


@dataclass
class TaskSet:
    tasks: list[QueryTask] = field(default_factory=list)
    qrels: Qrels = field(default_factory=dict)
    skipped_queries: list[str] = field(default_factory=list)


def _choose_filter(
    templates: list[FilterTemplate], positives: list[dict[str, Any]], rng: random.Random
) -> Filter | None:
    candidates: list[Filter] = []
    for t in templates:
        if t.op == "eq":
            values = sorted({str(v) for d in positives for v in field_values(d, t.field)})
            candidates += [Filter(field=t.field, op="eq", value=v) for v in values]
        else:
            for v in t.values or []:
                f = Filter(field=t.field, op=t.op, value=float(v))
                if any(matches(d, f) for d in positives):
                    candidates.append(f)
    return rng.choice(candidates) if candidates else None


def build_tasks(
    scenario: Scenario, variants: list[Query], qrels: Qrels, docs_by_id: dict[str, dict[str, Any]]
) -> TaskSet:
    seed = scenario.perturbations.seed
    result = TaskSet()
    filters: dict[str, Filter | None] = {}
    sorts: dict[str, Sort] = {}

    def add(query: Query, mode: str, base: str, grades: dict[str, int], **extra: Any) -> None:
        task = QueryTask(
            task_id=f"{query.qid}:{mode}:{base}",
            qid=query.qid,
            variant_of=query.variant_of,
            perturbation=query.perturbation,
            text=query.text,
            mode=mode,
            base=base,
            **extra,
        )
        result.tasks.append(task)
        result.qrels[task.task_id] = grades

    for query in variants:
        source = query.variant_of
        grades = qrels.get(source, {})
        positive_ids = [d for d, g in grades.items() if g > 0]
        if not positive_ids:
            if query.perturbation == "original":
                result.skipped_queries.append(source)
            continue
        positives = [docs_by_id[d] for d in sorted(positive_ids) if d in docs_by_id]

        for mode in scenario.modes:
            if mode in BASE_MODES:
                add(query, mode, mode, dict(grades))
                continue
            for base in scenario.structured_bases:
                if mode == "filtered":
                    if source not in filters:
                        rng = random.Random(f"{seed}:{source}:filter")
                        filters[source] = _choose_filter(scenario.filter_templates, positives, rng)
                    f = filters[source]
                    if f is None:
                        continue
                    kept = {
                        d: g for d, g in grades.items() if d in docs_by_id and matches(docs_by_id[d], f)
                    }
                    add(query, mode, base, kept, filter=f)
                elif mode == "sorted":
                    if source not in sorts:
                        rng = random.Random(f"{seed}:{source}:sort")
                        sorts[source] = rng.choice(scenario.sort_templates)
                    add(query, mode, base, dict(grades), sort=sorts[source])
                else:
                    add(query, mode, base, dict(grades), facets=list(scenario.index_fields.facets))
    return result
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest harness/tests/test_tasks.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add harness/irbench/filters.py harness/irbench/tasks.py harness/tests/test_tasks.py
git commit -m "feat: per-mode query tasks with derived filtered qrels"
```

---

### Task 10: `prepare` stage, manifest and CLI command

**Files:**
- Create: `harness/irbench/manifest.py`, `harness/irbench/stages/__init__.py` (empty), `harness/irbench/stages/prepare.py`
- Modify: `harness/irbench/cli.py`
- Test: `harness/tests/test_prepare.py`

**Interfaces:**
- Consumes: everything from Tasks 2–9.
- Produces:
  - `git_sha(root) -> str | None`; `write_manifest(path, payload: dict) -> None` (adds `irbench_version`, `git_sha`, `created_at` ISO-8601 UTC).
  - `PrepareResult(scenario_dir, n_docs, n_queries, n_variants, n_tasks, skipped_queries)`.
  - `prepare(root: Path, scenario_id: str, client: httpx.Client | None = None) -> PrepareResult`.
  - Files in `Workspace.scenario_dir(id)`: `queries.variants.jsonl`, `tasks.jsonl`, `qrels.tasks.tsv`, `prepare.manifest.json`. Docs stay at `Workspace.dataset_dir(dataset)/docs.jsonl`.
  - CLI: `irbench prepare --scenario <id> [--root <path>]`.

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_prepare.py`

```python
import json
import shutil

import httpx
import yaml
from typer.testing import CliRunner

from irbench.cli import app
from irbench.datasets.fetch import sha256_file
from irbench.io import read_jsonl, read_qrels
from irbench.stages.prepare import prepare

from conftest import REPO_ROOT

SCENARIO = "ecommerce/product-search"


def make_root(tmp_path, wands_raw):
    """A repo-shaped root whose dataset.yaml points at the mini fixture files."""
    root = tmp_path / "root"
    shutil.copytree(REPO_ROOT / "scenarios", root / "scenarios")
    spec = {
        "id": "wands",
        "version": "fixture",
        "license": "MIT",
        "source": "fixture",
        "files": {
            name: {"url": f"https://fixture/{name}.csv", "sha256": sha256_file(path)}
            for name, path in wands_raw.items()
        },
    }
    (root / "datasets" / "wands").mkdir(parents=True)
    (root / "datasets" / "wands" / "dataset.yaml").write_text(yaml.safe_dump(spec))
    return root


def fixture_client(wands_raw):
    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.strip("/").removesuffix(".csv")
        return httpx.Response(200, content=wands_raw[name].read_bytes())

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_prepare_writes_all_outputs(tmp_path, wands_raw):
    root = make_root(tmp_path, wands_raw)
    result = prepare(root, SCENARIO, client=fixture_client(wands_raw))

    assert result.n_docs == 4
    assert result.n_queries == 3
    assert result.skipped_queries == ["2"]
    tasks = list(read_jsonl(result.scenario_dir / "tasks.jsonl"))
    assert len(tasks) == result.n_tasks > 0
    qrels = read_qrels(result.scenario_dir / "qrels.tasks.tsv")
    assert {t["task_id"] for t in tasks} == set(qrels)

    manifest = json.loads((result.scenario_dir / "prepare.manifest.json").read_text())
    assert manifest["scenario"] == SCENARIO
    assert manifest["dataset"]["version"] == "fixture"
    assert manifest["seed"] == 20261009
    assert manifest["counts"]["skipped_queries"] == 1
    assert "created_at" in manifest and "irbench_version" in manifest


def test_prepare_is_deterministic(tmp_path, wands_raw):
    root = make_root(tmp_path, wands_raw)
    first = prepare(root, SCENARIO, client=fixture_client(wands_raw))
    a = (first.scenario_dir / "tasks.jsonl").read_bytes()
    second = prepare(root, SCENARIO, client=fixture_client(wands_raw))
    assert (second.scenario_dir / "tasks.jsonl").read_bytes() == a


def test_cli_prepare_uses_cached_raw_files(tmp_path, wands_raw):
    root = make_root(tmp_path, wands_raw)
    raw_dir = root / ".irbench" / "data" / "wands" / "raw"
    raw_dir.mkdir(parents=True)
    for name, path in wands_raw.items():
        shutil.copy(path, raw_dir / f"{name}.csv")  # valid checksum -> no network

    result = CliRunner().invoke(app, ["prepare", "--scenario", SCENARIO, "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "skipped 1 queries without relevant judgments" in result.stdout
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest harness/tests/test_prepare.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'irbench.stages'`.

- [ ] **Step 3: Implement**

`harness/irbench/manifest.py`:
```python
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from irbench import __version__


def git_sha(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    root = path
    while root != root.parent and not (root / ".git").exists():
        root = root.parent
    manifest = {
        "irbench_version": __version__,
        "git_sha": git_sha(root),
        "created_at": datetime.now(UTC).isoformat(),
        **payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
```

`harness/irbench/stages/__init__.py`: empty file.

`harness/irbench/stages/prepare.py`:
```python
from dataclasses import dataclass
from pathlib import Path

import httpx

from irbench.datasets import NORMALIZERS
from irbench.datasets.registry import fetch_dataset, load_dataset_spec
from irbench.io import read_jsonl, read_qrels, write_jsonl, write_qrels
from irbench.manifest import write_manifest
from irbench.models import Query
from irbench.paths import Workspace
from irbench.perturb.variants import make_variants
from irbench.scenario import load_scenario
from irbench.tasks import build_tasks
from irbench.terms import load_term_groups


@dataclass(frozen=True)
class PrepareResult:
    scenario_dir: Path
    n_docs: int
    n_queries: int
    n_variants: int
    n_tasks: int
    skipped_queries: list[str]


def prepare(root: Path, scenario_id: str, client: httpx.Client | None = None) -> PrepareResult:
    ws = Workspace(root)
    scenario = load_scenario(root, scenario_id)
    spec = load_dataset_spec(root, scenario.dataset)

    raw = fetch_dataset(spec, ws.raw_dir(spec.id), client)
    normalized = NORMALIZERS[spec.id](raw, ws.dataset_dir(spec.id))

    docs_by_id = {d[scenario.id_field]: d for d in read_jsonl(normalized.docs)}
    queries = [
        Query(qid=r["qid"], text=r["text"], variant_of=r["qid"], perturbation="original")
        for r in read_jsonl(normalized.queries)
    ]
    qrels = read_qrels(normalized.qrels)

    p = scenario.perturbations
    variants = make_variants(queries, p.seed, list(p.kinds), load_term_groups(scenario.vocabulary_path))
    task_set = build_tasks(scenario, variants, qrels, docs_by_id)

    out = ws.scenario_dir(scenario.id)
    write_jsonl(out / "queries.variants.jsonl", (v.model_dump() for v in variants))
    write_jsonl(out / "tasks.jsonl", (t.model_dump(mode="json") for t in task_set.tasks))
    write_qrels(out / "qrels.tasks.tsv", task_set.qrels)

    result = PrepareResult(
        scenario_dir=out,
        n_docs=normalized.n_docs,
        n_queries=normalized.n_queries,
        n_variants=len(variants),
        n_tasks=len(task_set.tasks),
        skipped_queries=task_set.skipped_queries,
    )
    write_manifest(
        out / "prepare.manifest.json",
        {
            "stage": "prepare",
            "scenario": scenario.id,
            "dataset": {
                "id": spec.id,
                "version": spec.version,
                "files": {name: f.sha256 for name, f in spec.files.items()},
            },
            "seed": p.seed,
            "counts": {
                "docs": result.n_docs,
                "queries": result.n_queries,
                "variants": result.n_variants,
                "tasks": result.n_tasks,
                "skipped_queries": len(result.skipped_queries),
            },
            "skipped_queries": result.skipped_queries,
        },
    )
    return result
```

Add to `harness/irbench/cli.py` (below `version`; add `from pathlib import Path` and `from irbench.stages.prepare import prepare as run_prepare` to the imports):
```python
ROOT_OPTION = typer.Option(Path.cwd(), "--root", help="Repository root.")


@app.command()
def prepare(
    scenario: str = typer.Option(..., "--scenario", help="Scenario id, e.g. ecommerce/product-search."),
    root: Path = ROOT_OPTION,
) -> None:
    """Download the dataset, generate query variants and per-mode tasks."""
    r = run_prepare(root, scenario)
    typer.echo(
        f"{scenario}: {r.n_docs} docs, {r.n_queries} queries, {r.n_variants} variants, "
        f"{r.n_tasks} tasks; skipped {len(r.skipped_queries)} queries without relevant judgments"
    )
    typer.echo(f"output: {r.scenario_dir}")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -v`
Expected: all tests pass (including earlier tasks).

- [ ] **Step 5: Run on the real dataset**

Run: `uv run irbench prepare --scenario ecommerce/product-search`
Expected: `42994 docs, 480 queries`, a variant count between 480 and 1920 (most queries get typo and prefix variants, fewer get synonym), and a non-zero task count. Write down the skipped query count in the commit message body.

- [ ] **Step 6: Commit**

```bash
git add harness/irbench/manifest.py harness/irbench/stages harness/irbench/cli.py harness/tests/test_prepare.py
git commit -m "feat: prepare stage producing variants, tasks, qrels and manifest"
```

---

### Task 11: Vector storage and embedding cache

**Files:**
- Create: `harness/irbench/embed/__init__.py` (empty), `harness/irbench/embed/base.py`, `harness/irbench/embed/vectors.py`, `harness/irbench/embed/cache.py`
- Test: `harness/tests/test_embed_cache.py`

**Interfaces:**
- Produces:
  - `TextKind = Literal["query", "doc"]`; `Embedder` Protocol: `model_id: str`, `dim: int`, `embed(texts: list[str], kind: TextKind) -> np.ndarray` (float32, shape `(len(texts), dim)`, L2-normalized).
  - `render_doc_text(doc: dict, fields: list[str], max_chars: int = 2000) -> str`.
  - `write_vectors(path, ids: list[str], vectors: np.ndarray)`, `read_vectors(path) -> tuple[list[str], np.ndarray]` (Parquet: `id` string, `vector` fixed_size_list<float32>).
  - `text_key(kind, text) -> str` (sha256 hex of `f"{kind}\0{text}"`); `VectorCache(path)` with `.get(key)`, `.put(key, vec)`, `.save()`, `len()`; `embed_with_cache(embedder, texts, kind, cache, batch_size=64, save_every=50) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_embed_cache.py`

```python
import numpy as np
import pytest

from irbench.embed.base import render_doc_text
from irbench.embed.cache import VectorCache, embed_with_cache, text_key
from irbench.embed.vectors import read_vectors, write_vectors


class FakeEmbedder:
    model_id = "fake"
    dim = 4

    def __init__(self, fail_after_calls: int | None = None):
        self.calls: list[list[str]] = []
        self.fail_after_calls = fail_after_calls

    def embed(self, texts, kind):
        if self.fail_after_calls is not None and len(self.calls) >= self.fail_after_calls:
            raise RuntimeError("API down")
        self.calls.append(list(texts))
        vecs = np.array([[len(t), ord(t[0]), 1.0 if kind == "query" else 0.0, 1.0] for t in texts])
        return (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)


def test_vectors_roundtrip(tmp_path):
    vecs = np.random.default_rng(0).random((3, 5), dtype=np.float32)
    write_vectors(tmp_path / "v.parquet", ["a", "b", "c"], vecs)
    ids, back = read_vectors(tmp_path / "v.parquet")
    assert ids == ["a", "b", "c"]
    np.testing.assert_array_equal(back, vecs)


def test_text_key_depends_on_kind():
    assert text_key("query", "sofa") != text_key("doc", "sofa")


def test_cache_avoids_recomputation_and_persists(tmp_path):
    emb = FakeEmbedder()
    cache = VectorCache(tmp_path / "cache.parquet")
    first = embed_with_cache(emb, ["sofa", "bed", "sofa"], "doc", cache, batch_size=2)
    assert first.shape == (3, 4)
    assert sum(len(c) for c in emb.calls) == 2  # "sofa" embedded once
    cache.save()

    emb2 = FakeEmbedder()
    again = embed_with_cache(emb2, ["bed", "sofa"], "doc", VectorCache(tmp_path / "cache.parquet"))
    assert emb2.calls == []
    np.testing.assert_array_equal(again, first[[1, 0]])


def test_empty_text_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        embed_with_cache(FakeEmbedder(), ["ok", "  "], "doc", VectorCache(tmp_path / "c.parquet"))


def test_wrong_shape_from_embedder_is_rejected(tmp_path):
    class Bad(FakeEmbedder):
        def embed(self, texts, kind):
            return np.zeros((len(texts), 3), dtype=np.float32)

    with pytest.raises(ValueError, match="shape"):
        embed_with_cache(Bad(), ["a"], "doc", VectorCache(tmp_path / "c.parquet"))


def test_render_doc_text_skips_empty_and_joins_lists():
    doc = {"name": "grey sofa", "product_class": ["Sofas", "Sectionals"], "description": ""}
    assert render_doc_text(doc, ["name", "product_class", "description"]) == (
        "grey sofa. Sofas, Sectionals"
    )
    assert len(render_doc_text({"name": "x" * 5000}, ["name"])) == 2000
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest harness/tests/test_embed_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'irbench.embed'`.

- [ ] **Step 3: Implement**

`harness/irbench/embed/base.py`:
```python
from typing import Any, Literal, Protocol

import numpy as np

TextKind = Literal["query", "doc"]


class Embedder(Protocol):
    model_id: str
    dim: int

    def embed(self, texts: list[str], kind: TextKind) -> np.ndarray: ...


def render_doc_text(doc: dict[str, Any], fields: list[str], max_chars: int = 2000) -> str:
    parts: list[str] = []
    for name in fields:
        value = doc.get(name)
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        text = "" if value is None else str(value).strip()
        if text:
            parts.append(text)
    return ". ".join(parts)[:max_chars]
```

`harness/irbench/embed/vectors.py`:
```python
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_vectors(path: Path, ids: list[str], vectors: np.ndarray) -> None:
    if vectors.ndim != 2 or len(ids) != vectors.shape[0]:
        raise ValueError(f"ids ({len(ids)}) and vectors {vectors.shape} do not line up")
    flat = pa.array(np.ascontiguousarray(vectors, dtype=np.float32).reshape(-1))
    column = pa.FixedSizeListArray.from_arrays(flat, vectors.shape[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"id": pa.array(ids, pa.string()), "vector": column}), path)


def read_vectors(path: Path) -> tuple[list[str], np.ndarray]:
    table = pq.read_table(path)
    ids = table.column("id").to_pylist()
    column = table.column("vector").combine_chunks()
    dim = column.type.list_size
    flat = column.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    return ids, flat.reshape(-1, dim)
```

`harness/irbench/embed/cache.py`:
```python
import hashlib
from pathlib import Path

import numpy as np

from irbench.embed.base import Embedder, TextKind
from irbench.embed.vectors import read_vectors, write_vectors


def text_key(kind: TextKind, text: str) -> str:
    return hashlib.sha256(f"{kind}\0{text}".encode()).hexdigest()


class VectorCache:
    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, np.ndarray] = {}
        if path.exists():
            keys, vectors = read_vectors(path)
            self._data = dict(zip(keys, vectors))

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str) -> np.ndarray | None:
        return self._data.get(key)

    def put(self, key: str, vector: np.ndarray) -> None:
        self._data[key] = np.asarray(vector, dtype=np.float32)

    def save(self) -> None:
        if not self._data:
            return
        keys = list(self._data)
        write_vectors(self.path, keys, np.stack([self._data[k] for k in keys]))


def embed_with_cache(
    embedder: Embedder,
    texts: list[str],
    kind: TextKind,
    cache: VectorCache,
    batch_size: int = 64,
    save_every: int = 50,
) -> np.ndarray:
    for i, t in enumerate(texts):
        if not t.strip():
            raise ValueError(f"cannot embed empty text at position {i}")
    keys = [text_key(kind, t) for t in texts]
    text_by_key = dict(zip(keys, texts))
    missing = [k for k in dict.fromkeys(keys) if cache.get(k) is None]

    for n, start in enumerate(range(0, len(missing), batch_size), start=1):
        batch = missing[start : start + batch_size]
        vectors = embedder.embed([text_by_key[k] for k in batch], kind)
        if vectors.shape != (len(batch), embedder.dim):
            raise ValueError(
                f"{embedder.model_id}: expected shape {(len(batch), embedder.dim)}, got {vectors.shape}"
            )
        for key, vector in zip(batch, vectors):
            cache.put(key, vector)
        if n % save_every == 0:
            cache.save()

    return np.stack([cache.get(k) for k in keys])
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest harness/tests/test_embed_cache.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add harness/irbench/embed harness/tests/test_embed_cache.py
git commit -m "feat: parquet vector storage and content-addressed embedding cache"
```

---

### Task 12: Embedding models, `embed` stage and CLI command

**Files:**
- Create: `harness/irbench/embed/models.py`, `harness/irbench/stages/embed.py`
- Modify: `harness/irbench/cli.py`
- Test: `harness/tests/test_embed_stage.py`

**Interfaces:**
- Consumes: Task 11 (`Embedder`, `VectorCache`, `embed_with_cache`, `render_doc_text`, `write_vectors`); Task 10 outputs (`queries.variants.jsonl`, dataset `docs.jsonl`).
- Produces:
  - `SentenceTransformerEmbedder(model_id, hf_name, dim, query_prefix="")`, `OpenAIEmbedder(model_id, dim)`, `MODELS: dict[str, Callable[[], Embedder]]`, `get_embedder(model_id) -> Embedder`.
  - `EmbedResult(docs_path, queries_path, n_docs, n_queries, n_computed)`.
  - `embed_scenario(root, scenario_id, model_id, embedder: Embedder | None = None) -> EmbedResult`.
  - Output files: docs → `Workspace.embeddings_dir(dataset, model)/docs.parquet`; query variants → `Workspace.scenario_dir(id)/embeddings/<model slug>/queries.parquet`; cache → `Workspace.embeddings_dir(dataset, model)/cache.parquet`.
  - CLI: `irbench embed --scenario <id> --model <model_id> [--root]`.

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_embed_stage.py`

```python
import numpy as np
import pytest

from irbench.embed.models import MODELS, get_embedder
from irbench.embed.vectors import read_vectors
from irbench.stages.embed import embed_scenario
from irbench.stages.prepare import prepare

from test_embed_cache import FakeEmbedder
from test_prepare import fixture_client, make_root

SCENARIO = "ecommerce/product-search"


@pytest.fixture
def prepared_root(tmp_path, wands_raw):
    root = make_root(tmp_path, wands_raw)
    prepare(root, SCENARIO, client=fixture_client(wands_raw))
    return root


def test_embed_writes_doc_and_query_vectors(prepared_root):
    result = embed_scenario(prepared_root, SCENARIO, "fake", embedder=FakeEmbedder())
    doc_ids, doc_vecs = read_vectors(result.docs_path)
    assert doc_ids == ["0", "1", "2", "3"]
    assert doc_vecs.shape == (4, 4)
    q_ids, _ = read_vectors(result.queries_path)
    assert "0" in q_ids and "0-typo" in q_ids


def test_second_run_computes_nothing(prepared_root):
    embed_scenario(prepared_root, SCENARIO, "fake", embedder=FakeEmbedder())
    again = embed_scenario(prepared_root, SCENARIO, "fake", embedder=FakeEmbedder())
    assert again.n_computed == 0


def test_interrupted_embed_resumes_from_cache(prepared_root, monkeypatch):
    monkeypatch.setattr("irbench.stages.embed.BATCH_SIZE", 1)
    monkeypatch.setattr("irbench.stages.embed.SAVE_EVERY", 1)
    first = FakeEmbedder(fail_after_calls=2)
    with pytest.raises(RuntimeError, match="API down"):
        embed_scenario(prepared_root, SCENARIO, "fake", embedder=first)
    already = {t for call in first.calls for t in call}
    assert len(already) == 2

    resumed = FakeEmbedder()
    result = embed_scenario(prepared_root, SCENARIO, "fake", embedder=resumed)
    seen_again = {t for call in resumed.calls for t in call}
    assert already.isdisjoint(seen_again)  # vectors paid for before the crash are reused
    assert result.n_computed == sum(len(c) for c in resumed.calls)


def test_embed_requires_prepare(tmp_path):
    with pytest.raises(FileNotFoundError, match="irbench prepare"):
        embed_scenario(tmp_path, SCENARIO, "fake", embedder=FakeEmbedder())


def test_model_registry():
    assert {"bge-small-en-v1.5", "text-embedding-3-small"} <= set(MODELS)
    with pytest.raises(ValueError, match="unknown model"):
        get_embedder("nope")


@pytest.mark.slow
def test_bge_small_real_model():
    emb = get_embedder("bge-small-en-v1.5")
    vecs = emb.embed(["grey velvet sofa"], "query")
    assert vecs.shape == (1, 384)
    assert np.isclose(np.linalg.norm(vecs[0]), 1.0, atol=1e-4)
```

Note: `test_embed_requires_prepare` uses an empty `tmp_path` with no `scenarios/` directory. `embed_scenario` must check for prepared outputs before loading the scenario. Implement it so the check against `Workspace(root).scenario_dir(scenario_id)` runs first.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest harness/tests/test_embed_stage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'irbench.embed.models'`.

- [ ] **Step 3: Implement**

`harness/irbench/embed/models.py`:
```python
from collections.abc import Callable

import numpy as np

from irbench.embed.base import Embedder, TextKind


class SentenceTransformerEmbedder:
    def __init__(self, model_id: str, hf_name: str, dim: int, query_prefix: str = ""):
        from sentence_transformers import SentenceTransformer  # optional extra: embed-local

        self.model_id = model_id
        self.dim = dim
        self.query_prefix = query_prefix
        self._model = SentenceTransformer(hf_name)

    def embed(self, texts: list[str], kind: TextKind) -> np.ndarray:
        inputs = [self.query_prefix + t for t in texts] if kind == "query" else texts
        vectors = self._model.encode(inputs, normalize_embeddings=True, batch_size=64)
        return np.asarray(vectors, dtype=np.float32)


class OpenAIEmbedder:
    def __init__(self, model_id: str, dim: int):
        from openai import OpenAI  # optional extra: embed-api; reads OPENAI_API_KEY

        self.model_id = model_id
        self.dim = dim
        self._client = OpenAI()

    def embed(self, texts: list[str], kind: TextKind) -> np.ndarray:
        response = self._client.embeddings.create(model=self.model_id, input=texts)
        data = sorted(response.data, key=lambda d: d.index)
        vectors = np.asarray([d.embedding for d in data], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


MODELS: dict[str, Callable[[], Embedder]] = {
    "bge-small-en-v1.5": lambda: SentenceTransformerEmbedder(
        "bge-small-en-v1.5",
        "BAAI/bge-small-en-v1.5",
        384,
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "text-embedding-3-small": lambda: OpenAIEmbedder("text-embedding-3-small", 1536),
}


def get_embedder(model_id: str) -> Embedder:
    if model_id not in MODELS:
        raise ValueError(f"unknown model {model_id!r}; known: {', '.join(sorted(MODELS))}")
    return MODELS[model_id]()
```

`harness/irbench/stages/embed.py`:
```python
from dataclasses import dataclass
from pathlib import Path

from irbench.embed.base import Embedder, render_doc_text
from irbench.embed.cache import VectorCache, embed_with_cache
from irbench.embed.models import get_embedder
from irbench.embed.vectors import write_vectors
from irbench.io import read_jsonl
from irbench.paths import Workspace
from irbench.scenario import load_scenario

BATCH_SIZE = 64
SAVE_EVERY = 50


@dataclass(frozen=True)
class EmbedResult:
    docs_path: Path
    queries_path: Path
    n_docs: int
    n_queries: int
    n_computed: int


def embed_scenario(
    root: Path, scenario_id: str, model_id: str, embedder: Embedder | None = None
) -> EmbedResult:
    ws = Workspace(root)
    variants_path = ws.scenario_dir(scenario_id) / "queries.variants.jsonl"
    if not variants_path.exists():
        raise FileNotFoundError(f"{variants_path} missing: run `irbench prepare --scenario {scenario_id}` first")

    scenario = load_scenario(root, scenario_id)
    emb = embedder or get_embedder(model_id)
    emb_dir = ws.embeddings_dir(scenario.dataset, model_id)
    cache = VectorCache(emb_dir / "cache.parquet")
    cached_before = len(cache)

    docs = list(read_jsonl(ws.dataset_dir(scenario.dataset) / "docs.jsonl"))
    variants = list(read_jsonl(variants_path))
    try:
        doc_vectors = embed_with_cache(
            emb,
            [render_doc_text(d, scenario.embedding_fields) for d in docs],
            "doc",
            cache,
            batch_size=BATCH_SIZE,
            save_every=SAVE_EVERY,
        )
        query_vectors = embed_with_cache(
            emb, [v["text"] for v in variants], "query", cache,
            batch_size=BATCH_SIZE, save_every=SAVE_EVERY,
        )
    finally:
        cache.save()

    docs_path = emb_dir / "docs.parquet"
    queries_path = ws.scenario_dir(scenario_id) / "embeddings" / emb_dir.name / "queries.parquet"
    write_vectors(docs_path, [d[scenario.id_field] for d in docs], doc_vectors)
    write_vectors(queries_path, [v["qid"] for v in variants], query_vectors)
    return EmbedResult(docs_path, queries_path, len(docs), len(variants), len(cache) - cached_before)
```

`n_computed` counts newly cached unique texts. Identical texts (for example a typo variant and a prefix variant that both come out as "salon chai") are embedded only once.

Add to `harness/irbench/cli.py` (import `from irbench.stages.embed import embed_scenario`):
```python
@app.command()
def embed(
    scenario: str = typer.Option(..., "--scenario"),
    model: str = typer.Option(..., "--model", help="Model id from the embedding registry."),
    root: Path = ROOT_OPTION,
) -> None:
    """Embed docs and query variants with one model (cached, resumable)."""
    r = embed_scenario(root, scenario, model)
    typer.echo(f"{model}: {r.n_docs} docs, {r.n_queries} queries, {r.n_computed} newly embedded")
    typer.echo(f"docs: {r.docs_path}\nqueries: {r.queries_path}")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -v`
Expected: all pass; the `slow` test is deselected.

- [ ] **Step 5: Run the real local model once**

Run: `uv sync --extra embed-local && uv run pytest -m slow -v && uv run irbench embed --scenario ecommerce/product-search --model bge-small-en-v1.5`
Expected: the slow test passes; embed prints `42994 docs`. Re-running prints `0 newly embedded`.

- [ ] **Step 6: Commit**

```bash
git add harness/irbench/embed/models.py harness/irbench/stages/embed.py harness/irbench/cli.py harness/tests/test_embed_stage.py
git commit -m "feat: embed stage with bge-small and OpenAI models, resumable cache"
```

---

### Task 13: Relevance scoring

**Files:**
- Create: `harness/irbench/score/__init__.py` (empty), `harness/irbench/score/relevance.py`
- Test: `harness/tests/test_score_relevance.py`

**Interfaces:**
- Consumes: `QueryTask`, `Qrels`, `RunResults` (Task 2).
- Produces:
  - `METRICS = ["ndcg@10", "recall@10", "recall@100", "mrr@10", "hit_rate@1"]`; reported names map `hit_rate@1 → success@1`.
  - `GroupScore(mode, base, perturbation, n, metrics: dict[str, float])` (pydantic; `perturbation="all"` for the overall group).
  - `score_relevance(tasks: list[QueryTask], qrels: Qrels, run: RunResults) -> list[GroupScore]` — groups sorted by `(mode, base, perturbation)`.

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_score_relevance.py`

Expected values are hand-computed with `trec_eval`'s definitions (linear gain, log2(rank+1) discount). For t1: DCG = 0/1 + 2/log2(3) = 1.26186; IDCG = 2/1 + 1/log2(3) = 2.63093; nDCG = 0.47962. Plan 4 adds a real `trec_eval` cross-check in CI.

```python
import pytest

from irbench.models import QueryTask
from irbench.score.relevance import score_relevance


def _task(tid: str, perturbation: str = "original", mode: str = "keyword") -> QueryTask:
    return QueryTask(
        task_id=tid, qid=tid, variant_of=tid, perturbation=perturbation,
        text="x", mode=mode, base="keyword" if mode != "semantic" else "semantic",
    )


QRELS = {"t1": {"d1": 2, "d2": 1, "d3": 0}, "t2": {"d5": 1}}


def _group(scores, perturbation="all", mode="keyword"):
    return next(s for s in scores if s.perturbation == perturbation and s.mode == mode)


def test_hand_computed_metrics():
    scores = score_relevance([_task("t1")], QRELS, {"t1": ["d3", "d1", "d9"]})
    m = _group(scores).metrics
    assert m["ndcg@10"] == pytest.approx(0.47962, abs=1e-4)
    assert m["recall@10"] == pytest.approx(0.5)
    assert m["mrr@10"] == pytest.approx(0.5)
    assert m["success@1"] == pytest.approx(0.0)


def test_missing_run_entry_scores_zero():
    scores = score_relevance([_task("t1"), _task("t2")], QRELS, {"t1": ["d3", "d1", "d9"]})
    overall = _group(scores)
    assert overall.n == 2
    assert overall.metrics["ndcg@10"] == pytest.approx(0.47962 / 2, abs=1e-4)
    assert overall.metrics["recall@100"] == pytest.approx(0.25)


def test_duplicate_ids_keep_first_rank():
    deduped = score_relevance([_task("t1")], QRELS, {"t1": ["d1", "d2"]})
    with_dupes = score_relevance([_task("t1")], QRELS, {"t1": ["d1", "d2", "d1", "d1"]})
    assert _group(with_dupes).metrics == _group(deduped).metrics
    assert _group(deduped).metrics["ndcg@10"] == pytest.approx(1.0)


def test_groups_by_mode_and_perturbation():
    tasks = [_task("t1"), _task("t2", perturbation="typo")]
    scores = score_relevance(tasks, QRELS, {"t1": ["d1"], "t2": ["d5"]})
    keys = {(s.mode, s.base, s.perturbation) for s in scores}
    assert keys == {
        ("keyword", "keyword", "all"),
        ("keyword", "keyword", "original"),
        ("keyword", "keyword", "typo"),
    }
    assert _group(scores, "typo").n == 1
    assert _group(scores, "typo").metrics["success@1"] == pytest.approx(1.0)


def test_empty_run_is_all_zero():
    scores = score_relevance([_task("t1")], QRELS, {})
    assert all(v == 0.0 for v in _group(scores).metrics.values())
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest harness/tests/test_score_relevance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'irbench.score'`.

- [ ] **Step 3: Implement** — `harness/irbench/score/relevance.py`

```python
from collections import defaultdict

from pydantic import BaseModel
from ranx import Qrels as RanxQrels
from ranx import Run as RanxRun
from ranx import evaluate

from irbench.models import Qrels, QueryTask, RunResults

METRICS = ["ndcg@10", "recall@10", "recall@100", "mrr@10", "hit_rate@1"]
REPORTED = {"hit_rate@1": "success@1"}


class GroupScore(BaseModel):
    mode: str
    base: str
    perturbation: str
    n: int
    metrics: dict[str, float]


def _ranked_scores(ids: list[str]) -> dict[str, float]:
    unique = list(dict.fromkeys(ids))  # duplicates keep their first (best) rank
    return {doc_id: float(len(unique) - i) for i, doc_id in enumerate(unique)}


def _evaluate(task_ids: list[str], qrels: Qrels, run: RunResults) -> dict[str, float]:
    run_dict = {t: _ranked_scores(run[t]) for t in task_ids if run.get(t)}
    if not run_dict:
        return {REPORTED.get(m, m): 0.0 for m in METRICS}
    values = evaluate(
        RanxQrels.from_dict({t: qrels[t] for t in task_ids}),
        RanxRun.from_dict(run_dict),
        METRICS,
        make_comparable=True,
    )
    return {REPORTED.get(m, m): float(values[m]) for m in METRICS}


def score_relevance(tasks: list[QueryTask], qrels: Qrels, run: RunResults) -> list[GroupScore]:
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for task in tasks:
        if task.task_id not in qrels:
            raise ValueError(f"task {task.task_id} has no qrels")
        groups[(task.mode, task.base, "all")].append(task.task_id)
        groups[(task.mode, task.base, task.perturbation)].append(task.task_id)
    return [
        GroupScore(mode=mode, base=base, perturbation=p, n=len(ids), metrics=_evaluate(ids, qrels, run))
        for (mode, base, p), ids in sorted(groups.items())
    ]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest harness/tests/test_score_relevance.py -v`
Expected: 5 passed. If `test_missing_run_entry_scores_zero` fails because ranx averaged over fewer queries, `make_comparable` has not padded the run. Check that `t2` exists in the qrels passed to ranx.

- [ ] **Step 5: Commit**

```bash
git add harness/irbench/score harness/tests/test_score_relevance.py
git commit -m "feat: grouped relevance scoring with ranx"
```

---

### Task 14: Structured-mode correctness checks

**Files:**
- Create: `harness/irbench/score/constraints.py`
- Test: `harness/tests/test_constraints.py`

**Interfaces:**
- Consumes: `Filter`, `Sort` (Task 2); `matches`, `field_values` (Task 9).
- Produces (Plan 2 calls these on engine responses):
  - `filter_violations(ids: list[str], f: Filter, docs_by_id) -> int` — unknown IDs count as violations.
  - `order_violations(ids: list[str], sort: Sort, docs_by_id) -> int` — adjacent out-of-order pairs; a missing value before a present one is a violation.
  - `expected_facets(ids: list[str], field: str, docs_by_id) -> dict[str, int]`.
  - `facet_accuracy(reported: dict[str, int], expected: dict[str, int]) -> float` — share of facet values (union of keys) whose counts match exactly; `1.0` when both are empty.

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_constraints.py`

```python
import pytest

from irbench.models import Filter, Sort
from irbench.score.constraints import (
    expected_facets,
    facet_accuracy,
    filter_violations,
    order_violations,
)

DOCS = {
    "a": {"product_class": ["Beds"], "average_rating": 4.5},
    "b": {"product_class": ["Sofas", "Sectionals"], "average_rating": 3.0},
    "c": {"product_class": ["Beds"], "average_rating": None},
    "d": {"product_class": [], "average_rating": 4.0},
}


def test_filter_violations_count_non_matching_and_unknown():
    f = Filter(field="product_class", op="eq", value="Beds")
    assert filter_violations(["a", "c"], f, DOCS) == 0
    assert filter_violations(["a", "b", "zzz"], f, DOCS) == 2


def test_order_violations_desc_and_asc():
    desc = Sort(field="average_rating", direction="desc")
    assert order_violations(["a", "d", "b"], desc, DOCS) == 0
    assert order_violations(["b", "a", "d"], desc, DOCS) == 1
    asc = Sort(field="average_rating", direction="asc")
    assert order_violations(["b", "d", "a"], asc, DOCS) == 0


def test_order_missing_values_must_be_last():
    desc = Sort(field="average_rating", direction="desc")
    assert order_violations(["a", "b", "c"], desc, DOCS) == 0
    assert order_violations(["c", "a", "b"], desc, DOCS) == 1


def test_expected_facets_counts_list_values():
    assert expected_facets(["a", "b", "c", "d"], "product_class", DOCS) == {
        "Beds": 2,
        "Sofas": 1,
        "Sectionals": 1,
    }


def test_facet_accuracy():
    expected = {"Beds": 2, "Sofas": 1}
    assert facet_accuracy({"Beds": 2, "Sofas": 1}, expected) == 1.0
    assert facet_accuracy({"Beds": 2, "Sofas": 3}, expected) == pytest.approx(0.5)
    assert facet_accuracy({"Beds": 2, "Sofas": 1, "Rugs": 1}, expected) == pytest.approx(2 / 3)
    assert facet_accuracy({}, {}) == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest harness/tests/test_constraints.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `harness/irbench/score/constraints.py`

```python
from collections import Counter
from typing import Any

from irbench.filters import field_values, matches
from irbench.models import Filter, Sort

Docs = dict[str, dict[str, Any]]


def filter_violations(ids: list[str], f: Filter, docs_by_id: Docs) -> int:
    return sum(1 for d in ids if d not in docs_by_id or not matches(docs_by_id[d], f))


def _sort_value(doc: dict[str, Any] | None, field: str) -> Any:
    if doc is None:
        return None
    values = field_values(doc, field)
    return values[0] if values else None


def order_violations(ids: list[str], sort: Sort, docs_by_id: Docs) -> int:
    values = [_sort_value(docs_by_id.get(d), sort.field) for d in ids]
    violations = 0
    for a, b in zip(values, values[1:]):
        if a is None:
            violations += b is not None
        elif b is not None and (a > b if sort.direction == "asc" else a < b):
            violations += 1
    return violations


def expected_facets(ids: list[str], field: str, docs_by_id: Docs) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for d in ids:
        if d in docs_by_id:
            counts.update(str(v) for v in field_values(docs_by_id[d], field))
    return dict(counts)


def facet_accuracy(reported: dict[str, int], expected: dict[str, int]) -> float:
    keys = set(reported) | set(expected)
    if not keys:
        return 1.0
    return sum(1 for k in keys if reported.get(k, 0) == expected.get(k, 0)) / len(keys)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest harness/tests/test_constraints.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add harness/irbench/score/constraints.py harness/tests/test_constraints.py
git commit -m "feat: filter, sort-order and facet correctness checks"
```

---

### Task 15: `score` CLI command and end-to-end check

**Files:**
- Modify: `harness/irbench/cli.py`
- Test: `harness/tests/test_cli_e2e.py`

**Interfaces:**
- Consumes: `prepare` outputs (Task 10), `read_run` (Task 2), `score_relevance` (Task 13).
- Produces: CLI `irbench score --scenario <id> --run <path.trec> --out <path.json> [--root]`. It writes `{"scenario": id, "run": path, "groups": [GroupScore...]}` and prints one line per `perturbation == "all"` group: `"<mode>/<base>: nDCG@10=<x.xxxx> n=<n>"`. It exits with code 1 if the run file references a task id that is not in `tasks.jsonl`.

- [ ] **Step 1: Write the failing tests** — `harness/tests/test_cli_e2e.py`

```python
import json

from typer.testing import CliRunner

from irbench.cli import app
from irbench.io import read_jsonl, read_qrels, write_run
from irbench.stages.prepare import prepare

from test_prepare import fixture_client, make_root

SCENARIO = "ecommerce/product-search"


def _perfect_run(scenario_dir):
    qrels = read_qrels(scenario_dir / "qrels.tasks.tsv")
    return {
        tid: [d for d, g in sorted(grades.items(), key=lambda kv: -kv[1]) if g > 0]
        for tid, grades in qrels.items()
    }


def test_prepare_then_score_perfect_run(tmp_path, wands_raw):
    root = make_root(tmp_path, wands_raw)
    result = prepare(root, SCENARIO, client=fixture_client(wands_raw))
    write_run(tmp_path / "perfect.trec", _perfect_run(result.scenario_dir), tag="oracle")

    out = tmp_path / "score.json"
    cli = CliRunner().invoke(
        app,
        ["score", "--scenario", SCENARIO, "--run", str(tmp_path / "perfect.trec"),
         "--out", str(out), "--root", str(root)],
    )
    assert cli.exit_code == 0, cli.output
    assert "keyword/keyword: nDCG@10=1.0000" in cli.stdout
    report = json.loads(out.read_text())
    n_tasks = len(list(read_jsonl(result.scenario_dir / "tasks.jsonl")))
    overall = [g for g in report["groups"] if g["perturbation"] == "all"]
    assert sum(g["n"] for g in overall) == n_tasks
    assert all(g["metrics"]["ndcg@10"] == 1.0 for g in overall)


def test_score_rejects_unknown_task_ids(tmp_path, wands_raw):
    root = make_root(tmp_path, wands_raw)
    prepare(root, SCENARIO, client=fixture_client(wands_raw))
    write_run(tmp_path / "bad.trec", {"nope:keyword:keyword": ["0"]}, tag="x")
    cli = CliRunner().invoke(
        app,
        ["score", "--scenario", SCENARIO, "--run", str(tmp_path / "bad.trec"),
         "--out", str(tmp_path / "s.json"), "--root", str(root)],
    )
    assert cli.exit_code == 1
    assert "nope:keyword:keyword" in cli.output
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest harness/tests/test_cli_e2e.py -v`
Expected: FAIL — `No such command 'score'` (exit code 2).

- [ ] **Step 3: Implement** — add to `harness/irbench/cli.py` (imports: `import json`, `from irbench.io import read_jsonl, read_qrels, read_run`, `from irbench.models import QueryTask`, `from irbench.paths import Workspace`, `from irbench.score.relevance import score_relevance`)

```python
@app.command()
def score(
    scenario: str = typer.Option(..., "--scenario"),
    run: Path = typer.Option(..., "--run", exists=True, dir_okay=False, help="TREC run file."),
    out: Path = typer.Option(..., "--out", help="Where to write the JSON score report."),
    root: Path = ROOT_OPTION,
) -> None:
    """Score a TREC run file against the scenario's task qrels."""
    scenario_dir = Workspace(root).scenario_dir(scenario)
    tasks = [QueryTask.model_validate(r) for r in read_jsonl(scenario_dir / "tasks.jsonl")]
    qrels = read_qrels(scenario_dir / "qrels.tasks.tsv")
    results = read_run(run)

    unknown = sorted(set(results) - {t.task_id for t in tasks})
    if unknown:
        typer.echo(f"run references unknown task ids: {', '.join(unknown[:10])}", err=True)
        raise typer.Exit(code=1)

    groups = score_relevance(tasks, qrels, results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"scenario": scenario, "run": str(run), "groups": [g.model_dump() for g in groups]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for g in groups:
        if g.perturbation == "all":
            typer.echo(f"{g.mode}/{g.base}: nDCG@10={g.metrics['ndcg@10']:.4f} n={g.n}")
```

- [ ] **Step 4: Run the full suite and lint**

Run: `uv run pytest -v && uv run ruff check harness && uv run ruff format --check harness`
Expected: all pass, lint clean. If formatting fails, run `uv run ruff format harness` and re-run.

- [ ] **Step 5: Smoke-test on real data with an oracle run**

Run:
```bash
uv run python -c "
from pathlib import Path
from irbench.io import read_qrels, write_run
q = read_qrels(Path('.irbench/scenarios/ecommerce__product-search/qrels.tasks.tsv'))
write_run(Path('.irbench/oracle.trec'), {t: [d for d, g in sorted(v.items(), key=lambda kv: -kv[1]) if g > 0][:100] for t, v in q.items()}, 'oracle')"
uv run irbench score --scenario ecommerce/product-search --run .irbench/oracle.trec --out .irbench/oracle.score.json
```
Expected: every `nDCG@10=1.0000`. Recall@100 may be below 1 for queries with more than 100 positives. That is expected and documented.

- [ ] **Step 6: Commit**

```bash
git add harness/irbench/cli.py harness/tests/test_cli_e2e.py
git commit -m "feat: score command and end-to-end prepare→score test"
```

---

## Spec coverage (Plan 1 scope)

| Spec section | Covered by |
|---|---|
| §3 Dataset pinned by version + checksum | Task 3 |
| §3 Scenario schema, modes, synonyms | Task 5 |
| §3 Query set + variants inheriting qrels | Tasks 8, 9 |
| §5.1 prepare | Tasks 3, 4, 10 |
| §5.2 embed (cache keyed by text sha256) | Tasks 11, 12 (publishing → Plan 4) |
| §5.8 score, per perturbation | Task 13 |
| §5 manifest | Task 10 (later stages extend it in Plans 2–3) |
| §5 scoring of filtered/sorted/faceted | Tasks 9, 14 (wired to engine output in Plan 2) |
| §6 Perturbation rules | Tasks 6, 7, 8 |
| §7 Shared synonyms file disjoint from perturbation answer key | Task 5 |
| §10 Perturbation determinism, qrels inheritance, run parsing | Tasks 2, 8, 9 |
| §10 Metric correctness | Task 13 (hand-computed); real `trec_eval` in CI → Plan 4 |
| §5.3–5.7, 5.9 up/index/render/relevance/perf/down | Plans 2–3 |
| §11 Report, docs | Plan 4 |
