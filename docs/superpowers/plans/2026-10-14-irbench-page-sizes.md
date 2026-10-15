# Speed at Two Page Sizes + Meilisearch Tuned Track — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure speed at page sizes 20 (headline) and 100, keep relevance unchanged, and add a Meilisearch `tuned` track with `matchingStrategy: frequency`.

**Architecture:**
- The scenario gains `perf.page_sizes`.
- Adapters gain a `depth` (retrieval pool size) next to `limit`.
- The render stage writes one request directory per extra page size.
- The perf stage runs the unchanged Rust loadgen once per page size into `perf-p<N>.json`.
- A small `irbench.perf_files` module reads those files (and legacy `perf.json`, as page size 100) for scoring, publishing and the report.

**Tech Stack:** Python 3.12 + uv (pydantic, typer, Jinja2, ranx), pytest, ruff. The Rust loadgen is not modified.

**Spec:** `docs/superpowers/specs/2026-10-14-irbench-page-sizes-design.md`

## Global Constraints

- Page sizes: `[20, 100]`, measured both; **20 is the headline**, and the first entry of `page_sizes` is the headline.
- `page_sizes`:
  - non-empty;
  - strictly ascending (so the headline is the smallest, and the report can order by value);
  - every value is between 1 and `top_k`;
  - absent means `[top_k]`.
- `rendered/` must stay **byte-identical** to today (`limit = depth = top_k`), and facet checks keep `depth = top_k`. Relevance must not change.
- Pools that adapters derived from `scenario.top_k` now come from `depth`. Engine-level settings (pgvector `hnsw.ef_search = 100`, ParadeDB `plan_cache_mode`) do not change per page size.
- Rendered directory for page size N: `rendered/` when N == `top_k`, else `rendered-p<N>/`, with no `*.check.json` files.
- Perf output per page size: `perf-p<N>.json`, in today's `perf.json` format.
- `scores.json` `perf` is keyed by page size as a string: `{"20": {...}, "100": {...}}`.
- Legacy bundles (a single `perf.json`, or a flat `scores.perf`) are read as page size **100** everywhere.
- A failure at any page size fails the perf stage.
- Meilisearch tuned track: `matchingStrategy: frequency` on the keyword and hybrid bases, with a `justification:`.
- No `Co-Authored-By` lines in commits.
- Commands:
  - `uv run pytest -q` (Docker tests are deselected by default);
  - `uv run ruff check --fix .`
  - `uv run ruff format harness engines`
  - `uv run irbench lint`

## Review Focus

1. **A scenario whose `top_k` is not in `page_sizes`** (e.g. `page_sizes: [20]`). Expected: relevance still uses `rendered/`, and perf runs only `rendered-p20/`. Test in Task 3.
2. **Re-running the render stage after a `page_sizes` change.** Expected: stale `rendered-p<N>/` directories from the earlier run are deleted, not replayed. Test in Task 3.
3. **A run directory holding both a legacy `perf.json` and new `perf-p<N>.json` files** (an old run re-run with `--stages perf`). Expected: the perf stage deletes the legacy file, and readers prefer the new files. Tests in Tasks 3 and 4.
4. **A report mixing a legacy bundle (100 only) with new bundles (20 and 100).** Expected: the page-20 section shows the legacy run as "not measured", not a crash and not a silent omission. Test in Task 5.
5. **Facet verification at page size 20.** Expected: no check files in `rendered-p20/`; the checks in `rendered/` are unchanged, including Qdrant's `max(limit, depth)` prefetch. Tests in Tasks 2 and 3.

---

### Task 1: Scenario `perf.page_sizes`

**Files:**
- Modify: `harness/irbench/scenario.py` (`PerfSpec`, `Scenario`)
- Modify: `scenarios/ecommerce/product-search/scenario.yaml` (`perf.page_sizes`)
- Test: `harness/tests/test_scenario.py`

**Interfaces:**
- Produces: `Scenario.page_sizes -> list[int]` (property): `perf.page_sizes`, or `[top_k]` when unset. Later tasks use only this property, never `perf.page_sizes` directly.

- [ ] **Step 1: Write the failing tests** (append to `harness/tests/test_scenario.py`, reusing its `_scenario_with` helper)

```python
def test_product_search_measures_speed_at_20_and_100():
    s = load_scenario(REPO_ROOT, SCENARIO)
    assert s.page_sizes == [20, 100]
    assert s.top_k == 100


def test_page_sizes_default_to_top_k(tmp_path):
    def drop(raw):
        raw["perf"].pop("page_sizes", None)

    assert _scenario_with(tmp_path, drop)().page_sizes == [100]


@pytest.mark.parametrize(
    "sizes, message",
    [
        ([], "non-empty"),
        ([20, 20], "strictly ascending"),
        ([100, 20], "strictly ascending"),
        ([0, 20], "between 1 and top_k"),
        ([20, 101], "between 1 and top_k"),
    ],
)
def test_page_sizes_are_validated(tmp_path, sizes, message):
    def set_sizes(raw):
        raw["perf"]["page_sizes"] = sizes

    with pytest.raises(ValueError, match=message):
        _scenario_with(tmp_path, set_sizes)()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest harness/tests/test_scenario.py -q -k page_sizes`
Expected: FAIL (`Scenario` has no attribute `page_sizes`).

- [ ] **Step 3: Implement**

In `PerfSpec` add the field (after `warmup_concurrency`):

```python
    # Result counts the perf stage measures, headline first (spec 2026-10-14 §3.1).
    # None means [top_k]; read it through Scenario.page_sizes.
    page_sizes: list[int] | None = None
```

In `Scenario._check_consistency`, before `return self`:

```python
        sizes = self.perf.page_sizes
        if sizes is not None:
            if not sizes:
                raise ValueError("perf.page_sizes must be non-empty")
            if any(b <= a for a, b in zip(sizes, sizes[1:])):
                raise ValueError(f"perf.page_sizes must be strictly ascending, got {sizes}")
            if any(not 1 <= n <= self.top_k for n in sizes):
                raise ValueError(
                    f"perf.page_sizes must each be between 1 and top_k ({self.top_k}), got {sizes}"
                )
```

Add the property next to `directory`:

```python
    @property
    def page_sizes(self) -> list[int]:
        """Page sizes the perf stage measures, headline (smallest) first."""
        return list(self.perf.page_sizes or [self.top_k])
```

In `scenarios/ecommerce/product-search/scenario.yaml`, under `perf:` after `warmup_concurrency: 4`:

```yaml
  # Speed is measured at each page size, same protocol; relevance always uses top_k.
  # 20: a search results page (headline). 100: deep retrieval (reranking, RAG).
  page_sizes: [20, 100]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest harness/tests/test_scenario.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add harness/irbench/scenario.py scenarios/ecommerce/product-search/scenario.yaml harness/tests/test_scenario.py
git commit -m "feat: scenario perf.page_sizes (20 and 100 for product search)"
```

---

### Task 2: Adapter `depth` (retrieval pool size)

**Files:**
- Modify: `harness/irbench/engines/base.py` (`EngineAdapter.render` signature, new `pool_depth` helper)
- Modify: `engines/meilisearch/adapter.py`, `engines/elasticsearch/adapter.py`, `engines/qdrant/adapter.py`, `harness/irbench/engines/postgres.py` (both Postgres engines inherit it; check `engines/postgres*/adapter.py` for any `render` override or `top_k` use and update it the same way)
- Modify: `harness/tests/fake_engine.py` (`FakeAdapter.render`)
- Create: `harness/tests/render_golden.py` (fixture generator), `harness/tests/fixtures/render_golden/<engine>.json`
- Test: `harness/tests/test_render_depth.py`

**Interfaces:**
- Produces: `EngineAdapter.render(self, task, vector, limit, facets=True, *, depth=None) -> Request | SqlRequest`. `depth=None` means `scenario.top_k`; `facets` stays the 4th positional parameter, since existing tests call `render(task, v, limit, facets)` positionally.
- Produces: `EngineAdapter.pool_depth(self, depth: int | None) -> int`, which returns `depth if depth is not None else self.ctx.scenario.top_k`.

- [ ] **Step 1: Capture golden renders from the UNCHANGED adapters**

Create `harness/tests/render_golden.py`:

```python
"""Golden renders: every adapter's requests at (limit=top_k, depth=top_k) and its facet checks.

Regenerate ONLY when a request shape change is intended:
    uv run python harness/tests/render_golden.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from adapter_helpers import make_adapter, make_task  # noqa: E402
from irbench.models import Filter, Sort  # noqa: E402

ENGINES = ["meilisearch", "elasticsearch", "qdrant", "postgres", "postgres-paradedb"]
GOLDEN_DIR = Path(__file__).parent / "fixtures" / "render_golden"
V = [0.1, 0.2, 0.3]


def cases():
    eq = Filter(field="product_class", op="eq", value="Beds")
    sort = Sort(field="average_rating", direction="desc")
    for base in ("keyword", "semantic", "hybrid"):
        yield f"{base}", make_task(base)
    for base in ("keyword", "semantic"):
        yield f"filtered-{base}", make_task("filtered", base, filter=eq)
        yield f"sorted-{base}", make_task("sorted", base, sort=sort)
        yield f"faceted-{base}", make_task("faceted", base, facets=["product_class"])


def render_all(engine: str, **kwargs) -> dict:
    adapter = make_adapter(engine, n_docs=1000)
    top_k = adapter.ctx.scenario.top_k
    verify = adapter.ctx.scenario.facet_verify_limit
    out = {}
    for name, task in cases():
        vector = None if task.base == "keyword" else V
        for label, limit, facets in (("main", top_k, True), ("check", verify, False)):
            if label == "check" and not task.facets:
                continue
            try:
                req = adapter.render(task, vector, limit, facets, **kwargs)
            except ValueError as e:  # e.g. Qdrant does not render faceted tasks
                out[f"{name}/{label}"] = {"error": str(e)}
                continue
            out[f"{name}/{label}"] = req.to_json()
    return out


if __name__ == "__main__":
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for engine in ENGINES:
        path = GOLDEN_DIR / f"{engine}.json"
        path.write_text(json.dumps(render_all(engine), indent=1, sort_keys=True) + "\n")
        print(f"wrote {path}")
```

Check `make_task` accepts `filter=`, `sort=` and `facets=` keyword arguments; it forwards `**extra` to `QueryTask`, and the field names are in `irbench/models.py`. If a field name differs, use the model's name. Then run it **before touching any adapter**:

Run: `uv run python harness/tests/render_golden.py`
Expected: five `wrote ...` lines; `git diff --stat` shows only new fixture files.

- [ ] **Step 2: Write the failing tests** (`harness/tests/test_render_depth.py`)

```python
import json

import pytest

from adapter_helpers import make_adapter, make_task
from render_golden import ENGINES, GOLDEN_DIR, V, render_all


@pytest.mark.parametrize("engine", ENGINES)
def test_requests_at_top_k_are_unchanged(engine):
    golden = json.loads((GOLDEN_DIR / f"{engine}.json").read_text())
    assert render_all(engine) == golden
    assert render_all(engine, depth=100) == golden


def _body(engine, task, limit, depth, vector=V):
    return make_adapter(engine, n_docs=1000).render(task, vector, limit, depth=depth).body


def test_meilisearch_page_of_20():
    assert _body("meilisearch", make_task("hybrid"), 20, 20)["limit"] == 20


def test_elasticsearch_knn_k_follows_depth():
    body = _body("elasticsearch", make_task("hybrid"), 20, 20)
    assert body["size"] == 20 and body["knn"]["k"] == 20


def test_qdrant_pools_follow_depth():
    hyb = _body("qdrant", make_task("hybrid"), 20, 20)
    assert hyb["limit"] == 20
    keyword_side, dense_side = hyb["prefetch"]
    assert dense_side["limit"] == 20 and keyword_side["limit"] == 20
    assert {leaf["limit"] for leaf in keyword_side["prefetch"]} == {20}
    kw = _body("qdrant", make_task("keyword"), 20, 20, vector=None)
    assert {leaf["limit"] for leaf in kw["prefetch"]} == {20}
    from irbench.models import Sort

    sort = Sort(field="average_rating", direction="desc")
    sem_sorted = _body("qdrant", make_task("sorted", "semantic", sort=sort), 20, 20)
    assert sem_sorted["prefetch"]["limit"] == 20


@pytest.mark.parametrize("engine", ["postgres", "postgres-paradedb"])
def test_postgres_hybrid_pools_follow_depth(engine):
    req = make_adapter(engine, n_docs=1000).render(make_task("hybrid"), V, 20, depth=20)
    ints = [p for p in req.params if isinstance(p, int)]
    # the RRF side limits (depth) and the outer limit are 20; no top_k (100) left anywhere
    assert 20 in ints and 100 not in ints


@pytest.mark.parametrize("engine", ["postgres", "postgres-paradedb"])
def test_postgres_semantic_sorted_pool_follows_depth(engine):
    from irbench.models import Sort

    sort = Sort(field="average_rating", direction="desc")
    task = make_task("sorted", "semantic", sort=sort)
    req = make_adapter(engine, n_docs=1000).render(task, V, 20, depth=20)
    ints = [p for p in req.params if isinstance(p, int)]
    assert 20 in ints and 100 not in ints
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest harness/tests/test_render_depth.py -q`
Expected: the golden tests at default depth PASS. The `depth=100` variant and the depth-20 tests FAIL with `TypeError: ... unexpected keyword argument 'depth'`.

- [ ] **Step 4: Implement**

`harness/irbench/engines/base.py`, abstract `render`:

```python
    @abstractmethod
    def render(
        self,
        task: QueryTask,
        vector: list[float] | None,
        limit: int,
        facets: bool = True,
        *,
        depth: int | None = None,
    ) -> "Request | SqlRequest":
        """The search request for one task, returning at most `limit` hits.

        `depth` is the retrieval pool size (how many candidates each retrieval side considers;
        None means the scenario's top_k). `facets=False` renders the facet verification fetch.
        """

    def pool_depth(self, depth: int | None) -> int:
        return self.ctx.scenario.top_k if depth is None else depth
```

Every adapter's `render` takes the same signature (`facets: bool = True, *, depth: int | None = None`) and computes `depth = self.pool_depth(depth)` once at the top. Then replace each `scenario.top_k` / `self.ctx.scenario.top_k` used as a pool:

- **Meilisearch** (`engines/meilisearch/adapter.py`): signature only; `depth` is unused (Meilisearch has no pool parameter).
- **Elasticsearch** (`engines/elasticsearch/adapter.py`): `"k": scenario.top_k` → `"k": depth`.
- **Qdrant** (`engines/qdrant/adapter.py`):
  - `_keyword(text, limit, query_filter)` gains a `depth: int` parameter, and the leaf `"limit": max(limit, self.ctx.scenario.top_k)` becomes `max(limit, depth)`;
  - `_sort_pool(task)` → `_sort_pool(task, depth)`, using `depth` where it used `top_k` (keyword: `max(self.ctx.n_docs, depth)`; else `depth`);
  - `_retrieve(task, vector, limit, query_filter)` gains `depth` and uses it for both hybrid prefetch sides instead of `top_k`, passing `depth` to `_keyword`;
  - `render` passes `depth` down;
  - docstrings say "depth (top_k by default)" where they said top_k.
- **Postgres** (`harness/irbench/engines/postgres.py`):
  - `_ranked(...)` gains `depth: int`; the hybrid branch's `top_k = q.p(self.ctx.scenario.top_k, "int")` becomes `top_k = q.p(depth, "int")` (rename the local to `side`);
  - in `render`, `top_k = self.ctx.scenario.top_k` becomes `depth = self.pool_depth(depth)`; the sorted pool is `None if task.base == "keyword" else depth`;
  - pass `depth` to every `_ranked` call;
  - the faceted `every` pool stays as it is;
  - update the `# hybrid: RRF over the keyword and vector top_k` comment to say `depth`.
- **`harness/tests/fake_engine.py`** `FakeAdapter.render`: signature `def render(self, task, vector, limit, facets=True, *, depth=None):`, and add `"depth": self.pool_depth(depth),` to the body dict.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest harness/tests/test_render_depth.py harness/tests/test_adapter_*.py harness/tests/test_render_stage.py -q`
Expected: PASS, including the golden comparisons (proof that `rendered/` is byte-identical).

Then the full suite: `uv run pytest -q`. Expected: PASS. If a fake-engine test compares a whole body dict and now sees `depth`, add `"depth": 100` to that expected dict.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix . && uv run ruff format harness engines
git add harness/irbench/engines engines harness/tests
git commit -m "feat: adapters take a retrieval depth separate from the page size"
```

---

### Task 3: Render per page size, and stage outputs

**Files:**
- Modify: `harness/irbench/stages/context.py` (`RunContext`)
- Modify: `harness/irbench/stages/render.py`
- Modify: `harness/irbench/stages/run.py` (`STAGE_OUTPUTS` and its invalidation loop)
- Test: `harness/tests/test_render_stage.py`, `harness/tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: `Scenario.page_sizes` (Task 1); `render(..., depth=...)` (Task 2).
- Produces:
  - `RunContext.rendered_dir_for(page_size: int) -> Path`: `self.rendered_dir` when `page_size == scenario.top_k`, else `self.run_dir / f"rendered-p{page_size}"`.
  - `RunContext.rendered_path(task, check=False, page_size=None) -> Path`: `page_size=None` means `rendered/`.
  - The perf stage (Task 4) reads requests through `rendered_dir_for`.

- [ ] **Step 1: Write the failing tests** (append to `harness/tests/test_render_stage.py`; `set_scenario` is in `fake_engine.py`)

```python
from fake_engine import set_scenario


def test_render_writes_a_request_set_per_extra_page_size(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    ctx = RunContext(root, FAKE_SPEC)
    assert ctx.scenario.page_sizes == [20, 100]
    render_stage(ctx)
    assert ctx.rendered_dir_for(100) == ctx.rendered_dir
    p20 = ctx.rendered_dir_for(20)
    assert p20 == ctx.run_dir / "rendered-p20"

    def files(d):
        return sorted(p.relative_to(d) for p in d.rglob("*.json"))

    main = [f for f in files(ctx.rendered_dir) if not f.name.endswith(".check.json")]
    assert files(p20) == main  # same tasks, same N/A set, no check files
    keyword = next(t for t in ctx.tasks if t.mode == "keyword")
    body20 = Request.from_json(read_json(ctx.rendered_path(keyword, page_size=20))).body
    body100 = Request.from_json(read_json(ctx.rendered_path(keyword))).body
    assert (body20["limit"], body20["depth"]) == (20, 20)
    assert (body100["limit"], body100["depth"]) == (100, 100)
    faceted = next(t for t in ctx.tasks if t.mode == "faceted")
    check = Request.from_json(read_json(ctx.rendered_path(faceted, check=True))).body
    assert (check["limit"], check["depth"]) == (1000, 100)


def test_top_k_not_measured_still_renders_relevance_requests(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    raw_perf = RunContext(root, FAKE_SPEC).scenario.perf.model_dump()
    set_scenario(root, perf={**raw_perf, "page_sizes": [20]})
    ctx = RunContext(root, FAKE_SPEC)
    render_stage(ctx)
    assert any(ctx.rendered_dir.rglob("*.json"))
    assert any(ctx.rendered_dir_for(20).rglob("*.json"))


def test_rerender_removes_page_size_dirs_no_longer_configured(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    ctx = RunContext(root, FAKE_SPEC)
    stale = ctx.run_dir / "rendered-p50" / "keyword" / "keyword" / "1.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    render_stage(ctx)
    assert not (ctx.run_dir / "rendered-p50").exists()
```

In `harness/tests/test_run_pipeline.py`, add a test that re-running `render` invalidates `rendered-p20/` and `perf-p20.json` through `STAGE_OUTPUTS`. Read how the existing invalidation tests in that file are built and follow the same pattern: create the files, run the pipeline from the `render` stage with the fake engine, and assert the stale files are gone or rewritten.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest harness/tests/test_render_stage.py -q`
Expected: FAIL (`rendered_dir_for` missing).

- [ ] **Step 3: Implement**

`context.py`, in `RunContext`:

```python
    def rendered_dir_for(self, page_size: int) -> Path:
        """Requests for one page size: rendered/ at top_k (shared with relevance)."""
        if page_size == self.scenario.top_k:
            return self.rendered_dir
        return self.run_dir / f"rendered-p{page_size}"

    def rendered_path(
        self, task: QueryTask, check: bool = False, page_size: int | None = None
    ) -> Path:
        name = f"{task.qid}.check.json" if check else f"{task.qid}.json"
        base = self.rendered_dir if page_size is None else self.rendered_dir_for(page_size)
        return base / task.mode / task.base / name
```

`render.py`:

```python
def render_stage(ctx: RunContext) -> RenderResult:
    """Write every supported task as a request file; unsupported ones go to na.json.

    rendered/ holds the top_k requests (relevance, and perf at top_k) plus facet checks;
    each other page size gets rendered-p<N>/ with limit = depth = N and no checks.
    """
    for stale in [ctx.rendered_dir, *ctx.run_dir.glob("rendered-p*")]:
        if stale.exists():
            shutil.rmtree(stale)
    adapter, scenario = ctx.adapter, ctx.scenario
    extra_sizes = [n for n in scenario.page_sizes if n != scenario.top_k]
    unsupported: dict[str, str] = {}
    rendered = 0
    for task in ctx.tasks:
        reason = adapter.unsupported_reason(task)
        if reason:
            unsupported[task.task_id] = reason
            continue
        vector = ctx.query_vector(task)
        request = adapter.render(task, vector, scenario.top_k, depth=scenario.top_k)
        write_json(ctx.rendered_path(task), request.to_json(), indent=None)
        if task.facets:
            check = adapter.render(
                task, vector, scenario.facet_verify_limit, facets=False, depth=scenario.top_k
            )
            write_json(ctx.rendered_path(task, check=True), check.to_json(), indent=None)
        for n in extra_sizes:
            page = adapter.render(task, vector, n, depth=n)
            write_json(ctx.rendered_path(task, page_size=n), page.to_json(), indent=None)
        rendered += 1
    write_json(ctx.run_dir / "na.json", unsupported)
    return RenderResult(n_rendered=rendered, n_unsupported=len(unsupported))
```

`run.py`: make `STAGE_OUTPUTS` glob patterns and expand them in the invalidation loop:

```python
STAGE_OUTPUTS = {
    "index": ("index.json",),
    "render": ("rendered", "rendered-p*", "na.json"),
    "relevance": ("run.trec", "checks.jsonl", "errors.jsonl"),
    "perf": ("perf.json", "perf-p*.json"),
    "score": ("scores.json",),
}
```

```python
        for pattern in STAGE_OUTPUTS.get(stage, ()):
            for path in run_dir.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
```

(`Path.glob` with a literal name matches only that name when it exists, so literal entries behave as before.)

- [ ] **Step 4: Run the tests**

Run: `uv run pytest harness/tests/test_render_stage.py harness/tests/test_run_pipeline.py -q`
Expected: the render tests PASS. `test_run_pipeline.py` tests that read `perf.json` or flat `scores["perf"]` may now fail; they are fixed in Task 4. Note which ones fail in the report, without changing them here.

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix . && uv run ruff format harness engines
git add harness/irbench/stages harness/tests
git commit -m "feat: render one request set per page size"
```

---

### Task 4: Perf per page size, scoring, CLI and publish

**Files:**
- Create: `harness/irbench/perf_files.py`
- Modify: `harness/irbench/stages/perf.py`, `harness/irbench/stages/score_run.py`, `harness/irbench/cli.py` (run summary), `harness/irbench/publish.py`
- Modify: `harness/tests/fake_engine.py` only if needed (the fake loadgen already writes wherever `--out` points)
- Test: `harness/tests/test_perf_stage.py`, `harness/tests/test_perf_files.py` (new), `harness/tests/test_run_pipeline.py`, `harness/tests/test_publish.py`

**Interfaces:**
- Consumes: `RunContext.rendered_dir_for(n)` (Task 3), `Scenario.page_sizes` (Task 1).
- Produces (`irbench.perf_files`):
  - `LEGACY_PAGE_SIZE = 100`
  - `perf_path(run_dir: Path, page_size: int) -> Path`, which returns `run_dir / f"perf-p{page_size}.json"`
  - `perf_report_paths(run_dir: Path) -> dict[int, Path]`: ascending by page size. If any `perf-p<N>.json` exists, only those; else `{100: run_dir / "perf.json"}` if that exists; else `{}`.
  - `read_perf_reports(run_dir: Path) -> dict[int, dict]`: the same keys, parsed JSON.
- Produces: `summarize_perf(report)` unchanged. `perf_stage(...) -> dict[str, dict[str, dict]]` keyed by page size as a string. `scores.json` `perf` has the same shape.

- [ ] **Step 1: Write the failing tests**

`harness/tests/test_perf_files.py`:

```python
import json

from irbench.perf_files import LEGACY_PAGE_SIZE, perf_path, perf_report_paths, read_perf_reports


def test_new_files_are_read_ascending(tmp_path):
    for n in (100, 20):
        perf_path(tmp_path, n).write_text(json.dumps({"n": n}))
    assert list(perf_report_paths(tmp_path)) == [20, 100]
    assert read_perf_reports(tmp_path) == {20: {"n": 20}, 100: {"n": 100}}


def test_legacy_perf_json_is_page_size_100(tmp_path):
    (tmp_path / "perf.json").write_text(json.dumps({"legacy": True}))
    assert read_perf_reports(tmp_path) == {LEGACY_PAGE_SIZE: {"legacy": True}}


def test_new_files_win_over_a_leftover_legacy_file(tmp_path):
    (tmp_path / "perf.json").write_text(json.dumps({"legacy": True}))
    perf_path(tmp_path, 20).write_text(json.dumps({"n": 20}))
    assert read_perf_reports(tmp_path) == {20: {"n": 20}}


def test_no_perf_files(tmp_path):
    assert read_perf_reports(tmp_path) == {}
```

In `harness/tests/test_perf_stage.py`:
- Rewrite `test_command_replays_the_rendered_dir_with_scenario_settings` to call `loadgen_command(ctx, LOCAL, taskset=None, page_size=20)` and assert `--requests == str(ctx.rendered_dir_for(20))` and `--out == str(ctx.run_dir / "perf-p20.json")`. The other assertions stay.
- Rewrite `test_perf_stage_runs_the_binary_and_summarizes`:

```python
def test_perf_stage_measures_each_page_size_in_order(tmp_path, wands_raw):
    ctx = RunContext(fake_root(tmp_path, wands_raw), FAKE_SPEC)
    render_stage(ctx)
    (ctx.run_dir / "perf.json").write_text("{}")  # leftover from a pre-page-size run
    calls = []
    summary = perf_stage(ctx, LOCAL, runner=fake_loadgen(calls=calls))
    assert calls[0][:3] == ["cargo", "build", "--release"]
    runs = [c for c in calls[1:]]
    assert [c[c.index("--requests") + 1] for c in runs] == [
        str(ctx.rendered_dir_for(20)),
        str(ctx.rendered_dir),
    ]
    assert [c[c.index("--out") + 1] for c in runs] == [
        str(ctx.run_dir / "perf-p20.json"),
        str(ctx.run_dir / "perf-p100.json"),
    ]
    assert not (ctx.run_dir / "perf.json").exists()
    assert list(summary) == ["20", "100"]
    assert summary["20"]["keyword/keyword"]["max_qps"] == 812.5
    assert "all" not in summary["100"]


def test_a_failing_page_size_fails_the_stage(tmp_path, wands_raw):
    ctx = RunContext(fake_root(tmp_path, wands_raw), FAKE_SPEC)
    render_stage(ctx)
    ok = fake_loadgen()

    def runner(cmd):
        if cmd[0] != "cargo" and cmd[cmd.index("--out") + 1].endswith("perf-p100.json"):
            raise RuntimeError("loadgen crashed")
        return ok(cmd)

    with pytest.raises(RuntimeError, match="loadgen crashed"):
        perf_stage(ctx, LOCAL, runner=runner)
```

- `test_perf_needs_rendered_requests_and_the_binary` keeps its intent: without rendered requests, `FileNotFoundError` mentioning "render".

In `harness/tests/test_run_pipeline.py`, update the perf assertions:
- `manifest["stages"]["perf"]["result"]["20"]["keyword/keyword"]["max_qps"] == 812.5`
- `scores["perf"]["100"]["keyword/keyword"]["max_qps_concurrency"] == 16`
- the CLI output check becomes `"perf p20 keyword/keyword: p50=1.50ms p99=4.00ms max_qps=812 (c=16)" in result.stdout`, plus the same line with `p100`.

In `harness/tests/test_publish.py`:
- `make_run` gets an extra file `perf-p20.json`;
- assert it is copied, next to the legacy `perf.json` case that already exists (a legacy bundle must still publish its `perf.json`).

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest harness/tests/test_perf_files.py harness/tests/test_perf_stage.py harness/tests/test_run_pipeline.py harness/tests/test_publish.py -q`
Expected: FAIL (no `irbench.perf_files`; `loadgen_command` has no `page_size`).

- [ ] **Step 3: Implement**

`harness/irbench/perf_files.py`:

```python
"""Where the perf stage writes its reports, and how every reader finds them.

One loadgen report per page size: perf-p<N>.json. Bundles from before page sizes hold a
single perf.json measured at top_k = 100, read as page size 100.
"""

import re
from pathlib import Path
from typing import Any

from irbench.io import read_json

LEGACY_PAGE_SIZE = 100
LEGACY_NAME = "perf.json"
_NAME = re.compile(r"perf-p(\d+)\.json")


def perf_path(run_dir: Path, page_size: int) -> Path:
    return run_dir / f"perf-p{page_size}.json"


def perf_report_paths(run_dir: Path) -> dict[int, Path]:
    found = {
        int(m.group(1)): p
        for p in run_dir.glob("perf-p*.json")
        if (m := _NAME.fullmatch(p.name))
    }
    if found:
        return dict(sorted(found.items()))
    legacy = run_dir / LEGACY_NAME
    return {LEGACY_PAGE_SIZE: legacy} if legacy.exists() else {}


def read_perf_reports(run_dir: Path) -> dict[int, dict[str, Any]]:
    return {n: read_json(p) for n, p in perf_report_paths(run_dir).items()}
```

`stages/perf.py`:
- `loadgen_command(ctx, profile, taskset, page_size)` sets `--requests` to `str(ctx.rendered_dir_for(page_size))` and `--out` to `str(perf_path(ctx.run_dir, page_size))`. The rest is unchanged.
- `perf_stage`:

```python
def perf_stage(
    ctx: RunContext, profile: HardwareProfile, runner: Runner = run_command
) -> dict[str, dict[str, dict[str, Any]]]:
    """Replay the rendered requests with the Rust loadgen, once per page size (spec §4)."""
    sizes = ctx.scenario.page_sizes
    missing = [n for n in sizes if not ctx.rendered_dir_for(n).exists()]
    if missing:
        raise FileNotFoundError(
            f"rendered requests for page sizes {missing} missing: run the render stage first"
        )
    manifest = str(ctx.root / "loadgen" / "Cargo.toml")
    runner(["cargo", "build", "--release", "--quiet", "--manifest-path", manifest])
    binary = ctx.root / LOADGEN_PATH
    if not binary.exists():
        raise FileNotFoundError(
            f"{binary} missing: loadgen build produced no binary ({BUILD_HINT})"
        )
    # A pre-page-size report would otherwise shadow nothing but confuse readers; drop it.
    (ctx.run_dir / LEGACY_NAME).unlink(missing_ok=True)
    taskset = shutil.which("taskset")
    summary = {}
    for n in sizes:
        runner(loadgen_command(ctx, profile, taskset, n))
        summary[str(n)] = summarize_perf(read_json(perf_path(ctx.run_dir, n)))
    return summary
```

`stages/score_run.py` (import `read_perf_reports` from `irbench.perf_files`): replace the `"perf": ...` entry with:

```python
        "perf": {
            str(n): summarize_perf(report)
            for n, report in read_perf_reports(run_dir).items()
        }
        or None,
```

`cli.py` (import `LEGACY_PAGE_SIZE` from `irbench.perf_files`) run summary, replacing the `for group, p in (scores.get("perf") or {}).items():` loop:

```python
        perf = scores.get("perf") or {}
        if perf and all("p50_ms" in v for v in perf.values()):  # legacy flat summary
            perf = {str(LEGACY_PAGE_SIZE): perf}
        for size, groups in perf.items():
            for group, p in groups.items():
                qps = (
                    "n/a"
                    if p["max_qps"] is None
                    else f"{p['max_qps']:.0f} (c={p['max_qps_concurrency']})"
                )
                typer.echo(
                    f"perf p{size} {group}: p50={p['p50_ms']:.2f}ms "
                    f"p99={p['p99_ms']:.2f}ms max_qps={qps}"
                )
```

`publish.py`: keep `BUNDLE_FILES` for the fixed names (the legacy `perf.json` stays in it), and also copy every `perf-p*.json`:

```python
    names = [n for n in BUNDLE_FILES if (src / n).exists()]
    names += sorted(p.name for p in src.glob("perf-p*.json"))
    for name in names:
        shutil.copy2(src / name, tmp / name)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q`
Expected: PASS, except report tests that read perf, which Task 5 covers. If any report test fails only because `scores["perf"]` changed shape, note it for Task 5 and leave it.

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix . && uv run ruff format harness engines
git add harness/irbench harness/tests
git commit -m "feat: measure speed once per page size; scores and CLI keyed by page size"
```

---

### Task 5: Report per page size

**Files:**
- Modify: `harness/irbench/report/data.py` (`RunResult.perf`, loading and validation)
- Modify: `harness/irbench/report/render.py` (`_latency_cell`, `_qps_cell`, `_series`, `perf_view`, `run_view`)
- Modify: `harness/irbench/report/templates/perf.html.j2`, `harness/irbench/report/templates/run.html.j2`
- Modify: `harness/tests/report_fixtures.py` (it can write `perf-p<N>.json` files)
- Test: `harness/tests/test_report_data.py`, `harness/tests/test_report_render.py`

**Interfaces:**
- Consumes: `irbench.perf_files.read_perf_reports`, `LEGACY_PAGE_SIZE` (Task 4).
- Produces:
  - `RunResult.perf: dict[int, dict[str, Any]] | None`, keyed by page size, ascending; `None` when the bundle has no perf file.
  - `PAGE_SIZE_TITLES = {20: "search results page", 100: "deep retrieval: reranking, RAG"}` in `render.py`; unknown sizes get no subtitle.

- [ ] **Step 1: Write the failing tests**

Read `report_fixtures.py` first. Its run builder takes `perf_report=` and writes `perf.json`. Add a `perf_reports: dict[int, dict] | None = None` argument that writes `perf-p<N>.json` per entry (keep `perf_report` as the legacy path). Then add the following tests.

In `test_report_data.py`:

```python
def test_runs_load_perf_per_page_size(tmp_path):
    # build one run with perf_reports={20: perf({...}), 100: perf({...})} using report_fixtures
    ...  # use the file's existing run-builder call pattern
    assert list(run.perf) == [20, 100]


def test_legacy_perf_json_loads_as_page_size_100(tmp_path):
    # build one run with perf_report=perf({...}) (legacy path)
    ...
    assert list(run.perf) == [100]
```

Write the `...` parts with the builder calls already used in that file; every other assertion is as shown. A perf file missing `limits`, `latency`, `throughput` or `skipped_groups` must still raise `ValueError` naming the file. Parametrize the existing validation test over `perf.json` and `perf-p20.json`.

In `test_report_render.py`:
- **Page-size sections:** a scenario with one new run (20 and 100) and one legacy run (100 only) renders a page-20 section first, titled with "20" and "search results page", and a page-100 section titled with "100" and "deep retrieval". In the page-20 section, the legacy run's cells say "not measured".
- **Run page:** the run page of the new run shows one performance table per page size, 20 first.
- **Existing perf tests:** keep passing, with expected HTML adjusted only for the new section headings.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest harness/tests/test_report_data.py harness/tests/test_report_render.py -q`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement**

`data.py`:
- In `_load_run`, load `scores` and `index` as today, and `perf` with `read_perf_reports(d) or None`. Remove `"perf"` from the generic `{name}.json` loop.
- `_validate_files` validates each perf report and names the real file in errors (`perf_report_paths(d)[n]`).
- `RunResult.perf` is typed `dict[int, dict[str, Any]] | None`.

`render.py`: the per-run cell helpers take the page size:

```python
def _perf(run: RunResult, size: int) -> dict[str, Any] | None:
    return (run.perf or {}).get(size)


def _latency_cell(run: RunResult, size: int, group: str) -> Cell:
    perf = _perf(run, size)
    if perf is None:
        return Cell("—", "missing", "not measured at this page size")
    lat = perf["latency"].get(group)
    ...  # unchanged body, reading `perf` instead of `run.perf`
```

Apply the same change to `_qps_cell(run, size, group)` and `_series(run, size, group, slots)`.

`perf_view(s, slots)` returns `{"sizes": [...], "indexing": ..., "unmeasured": ...}`:
- `sizes` is ascending over the union of every measured run's page sizes.
- Each entry is today's dict (`runs`, `limits`, `latency`, `max_qps`, `charts`) computed for that size, plus:
  - `"size": n`
  - `"title": PAGE_SIZE_TITLES.get(n, "")`
  - `"headline": i == 0`
- `runs` lists every measured run (perf present), so a run missing this size shows "not measured" cells.
- Chart titles include the size: `f"{fmt_group(group)} at {n} results: queries per second by concurrency, {model}"`.

`run_view`'s `"perf"` becomes a list of `{"size": n, "title": ..., "rows": [...]}`, one per size of that run, ascending.

`perf.html.j2`:
- Wrap today's latency, max-QPS and chart blocks in `{% for sz in view.perf.sizes %}`.
- Headline (`sz.headline`): render inside `<div class="page-size">` with `<h3>Page of {{ sz.size }} results{% if sz.title %} ({{ sz.title }}){% endif %}</h3>`.
- Other sizes: render inside `<details class="page-size"><summary>` carrying the same heading text, then the same blocks.
- Inside, references move from `view.perf.X` to `sz.X`.
- The intro note gains one sentence: "Speed is measured at each page size with the same protocol; relevance always uses the top 100."
- Indexing and unmeasured stay outside the loop.

`run.html.j2`: loop over `view.perf`, with one `<h3>` and one table per size, same columns as today.

- [ ] **Step 4: Run the tests and build the local report**

Run: `uv run pytest -q`
Expected: PASS (the whole suite).

Run: `uv run irbench report --results .irbench/runs --out /tmp/irbench-site-check`
Expected: it builds from the existing legacy local runs, with a page-100 section only (no run has page 20 yet), and no exception.

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix . && uv run ruff format harness engines
git add harness/irbench/report harness/tests
git commit -m "feat: report speed per page size, headline first; legacy bundles read as 100"
```

---

### Task 6: Meilisearch tuned track, docs, spec amendments

**Files:**
- Modify: `engines/meilisearch/configs/ecommerce/product-search.yaml`
- Modify: `docs/methodology/performance.mdx`, `docs/methodology/parity.mdx`
- Modify: `docs/superpowers/specs/2026-10-09-irbench-design.md` (§5 steps 5 and 7, §11)
- Test: `harness/tests/test_engine_config.py` (or the Meilisearch adapter test file, whichever holds track tests); `uv run irbench lint`

**Interfaces:**
- Consumes: the existing `TrackConfig` deep-merge of `tuned` over `default`, and `search.<base>` merging in the Meilisearch adapter.

- [ ] **Step 1: Write the failing test** (in `harness/tests/test_adapter_meilisearch.py`)

```python
def test_tuned_track_uses_frequency_matching_on_keyword_and_hybrid():
    config = load_engine_config(REPO_ROOT, "meilisearch", SCENARIO).track("tuned")
    adapter = make_adapter("meilisearch", config.model_dump())
    for base in ("keyword", "hybrid"):
        body = adapter.render(make_task(base), V, 100).body
        assert body["matchingStrategy"] == "frequency"
    sem = adapter.render(make_task("semantic"), V, 100).body
    assert "matchingStrategy" not in sem
    assert "matchingStrategy" in config.justification
```

Adjust the imports (`load_engine_config`, `REPO_ROOT`, `V`) to that file's existing ones. If `make_adapter` takes the raw track dict differently, pass whatever form its other tuned-track tests use.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest harness/tests/test_adapter_meilisearch.py -q -k tuned`
Expected: FAIL ("no tuned track defined").

- [ ] **Step 3: Implement the config** (append to the YAML, at top level next to `default:`)

```yaml
tuned:
  search:
    keyword: {matchingStrategy: frequency}
    hybrid: {matchingStrategy: frequency}
  justification:
    matchingStrategy: >-
      With the default `last`, Meilisearch drops words from the end of the query
      until something matches, so the product type (usually the last word) goes
      first and an unknown leading brand or modifier is kept. `frequency` drops
      the most frequent words first. Measured on WANDS (v1.54.3): keyword nDCG@10
      0.481 -> 0.508, zero-result queries 87 -> 14. A documented, supported
      search parameter; no ranking-rule change.
```

- [ ] **Step 4: Run the test and the lint**

Run: `uv run pytest harness/tests/test_adapter_meilisearch.py -q && uv run irbench lint`
Expected: PASS; lint reports 0 findings. If the lint flags the justification key as unknown for a search parameter, report it rather than change the lint (spec §6).

- [ ] **Step 5: Docs**

- `docs/methodology/performance.mdx`: add a section "Page sizes" after the protocol section:
  - speed is measured at each `perf.page_sizes` entry (product search: 20, the headline, a results page; and 100, deep retrieval for reranking and RAG), with the same protocol;
  - each request returns that many hits;
  - retrieval pools that the adapter sets per request (Elasticsearch kNN `k`, Qdrant prefetch, Postgres RRF sides) follow the page size, while engine-level settings do not (pgvector `hnsw.ef_search`);
  - relevance always uses `top_k = 100`;
  - one sentence on why both exist: some engines' cost grows with the result count.
- `docs/methodology/parity.mdx`: where it says hybrid RRF uses "each side `top_k`", say "each side the request's depth (`top_k` for relevance, the page size for speed)".
- `docs/superpowers/specs/2026-10-09-irbench-design.md`:
  - §5 step 5: add "Speed requests per page size: see `2026-10-14-irbench-page-sizes-design.md`".
  - §5 step 7: "the loadgen replays the rendered files once per page size (`perf-p<N>.json`)".
  - §11: one line on per-page-size sections.

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix . && uv run ruff format harness engines
git add engines/meilisearch docs harness/tests
git commit -m "feat: Meilisearch tuned track (matchingStrategy frequency); page-size docs"
```

---

### Task 7: Runs and report (controller, not a subagent)

Local Docker runs on the laptop profile; nothing is published or pushed.

- [ ] **Step 1:** For each engine in `meilisearch elasticsearch qdrant postgres postgres-paradedb`, run the full pipeline (default track):
  `uv run irbench run --scenario ecommerce/product-search --engine <e> --model bge-small-en-v1.5`
  Before each run, save the previous run's `scores.json` relevance block to the session scratchpad.
- [ ] **Step 2:** Verify relevance is identical to the saved block for every engine (all groups, all metrics). Any difference is a Task 2 bug: stop and fix it.
- [ ] **Step 3:** Run Meilisearch `--track tuned`. Expected keyword nDCG@10 ≈ 0.508 (the sweep measured 0.508).
- [ ] **Step 4:** `uv run irbench report --results .irbench/runs --out .irbench/site`, then open it. Check:
  - the page-20 section comes first;
  - Meilisearch semantic at 20 is close to the other engines;
  - the tuned run appears dashed.
- [ ] **Step 5:** Refresh the Linear project's "Detailed results" note with both page sizes and the tuned run.
