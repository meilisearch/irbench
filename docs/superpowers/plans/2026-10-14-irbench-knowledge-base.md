# IRBench Knowledge-Base Scenarios Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three knowledge-base scenarios (CQADupStack, TREC-COVID, LoTTE lifestyle search) to IRBench, with the generic harness changes they need, configs for all 5 engines, and local runs.

**Architecture:** Datasets stay declarative (`datasets/<id>/dataset.yaml`, pinned files) and gain zip-member files fetched by HTTP range requests. Three new normalizers share a BEIR reader. Scenarios gain optional synonyms and query groups; scores gain per-group rows that never collide with today's rows, so WANDS results are unchanged.

**Tech Stack:** Python 3.12, uv, pydantic, httpx, pytest, ruff; Jinja2 report; engines in Docker (OrbStack).

**Spec:** `docs/superpowers/specs/2026-10-14-irbench-knowledge-base-design.md`

## Global Constraints

- Never add `Co-Authored-By` lines to commits.
- Tests: `uv run pytest -q` from the repo root; lint: `uv run ruff check . && uv run ruff format --check harness`. Do not run `ruff format .` on the repo root (it rewrites `docs/superpowers/*.md`).
- Raw data is downloaded, never committed. Every dataset file is pinned by URL and sha256.
- WANDS results must not change: rows without a group keep today's exact shape (no `group` key).
- Perturbations for the new scenarios: `kinds: [typo, prefix]`; no synonyms, no vocabulary.
- All new scenarios: `top_k: 100`, `perf.page_sizes: [20, 100]`, same perf protocol as `scenarios/ecommerce/product-search/scenario.yaml`.
- A run that fails (e.g. an engine runs out of memory) is recorded as failed; scenarios are never shrunk to fit.

## Review Focus

1. A zip member whose compressed size spans many range requests (CQADupStack `queries.jsonl` members are 177–826 MB uncompressed): extraction must stream to disk, not hold the member in memory.
2. A CORD-19 `publish_time` that is empty, year-only (`2020`) or a full date: `publish_year`/`publish_date` must be null, `YYYY`/`YYYY0101`, or `YYYY`/`YYYYMMDD` respectively, never a crash.
3. Score lookups after groups exist: every reader of `relevance` rows (report `ndcg`, `run_view`, matrix) must ignore grouped rows, or a group's score silently replaces the overall score.
4. An engine receiving no synonyms: no empty synonym object, filter or thesaurus is created (Postgres thesaurus with an empty file).
5. A BEIR query whose qrels point only at dropped (empty) documents: the query ends up with no positives and is skipped by the task builder, and the skip is counted in the manifest.

Tests for each line are in the owning task (Tasks 1, 5, 3, 2, 4).

---

## File Structure

| File | Responsibility |
|---|---|
| `harness/irbench/datasets/fetch.py` | `fetch_file` (existing) + `fetch_zip_member` via HTTP range requests |
| `harness/irbench/datasets/registry.py` | `FileSpec.member`; `fetch_dataset` routes members |
| `harness/irbench/datasets/wands.py` | `NormalizedDataset` gains `dropped` counts |
| `harness/irbench/datasets/beir.py` (new) | BEIR jsonl/qrels readers, empty-doc dropping, writing normalized files |
| `harness/irbench/datasets/cqadupstack.py` (new) | CQADupStack normalizer |
| `harness/irbench/datasets/lotte.py` (new) | LoTTE normalizer |
| `harness/irbench/datasets/trec_covid.py` (new) | TREC-COVID normalizer with the CORD-19 metadata join |
| `harness/irbench/datasets/__init__.py` | `NORMALIZERS` registry |
| `harness/irbench/scenario.py` | optional `synonyms`, optional `perturbations.vocabulary` |
| `harness/irbench/stages/context.py`, `stages/prepare.py` | no synonyms → empty term groups; groups and drop counts flow through prepare |
| `engines/postgres/adapter.py` | no thesaurus when there are no groups |
| `harness/irbench/models.py`, `perturb/variants.py`, `tasks.py`, `score/relevance.py`, `stages/score_run.py` | query groups |
| `harness/irbench/report/data.py`, `report/render.py`, `report/templates/scenario.html.j2`, `run.html.j2` | ignore grouped rows in lookups; group table; perturbation columns per scenario |
| `datasets/{cqadupstack,trec-covid,lotte-lifestyle}/dataset.yaml` | pinned sources |
| `scenarios/knowledge-base/*/scenario.yaml` | the three scenarios |
| `engines/*/configs/knowledge-base/*.yaml` | 15 engine configs |
| `.github/workflows/ci.yml`, `docs/…` | CI prepares the new scenarios; docs pages |

---

### Task 1: Fetch single files out of a remote zip

**Files:**
- Modify: `harness/irbench/datasets/fetch.py`
- Modify: `harness/irbench/datasets/registry.py`
- Test: `harness/tests/test_fetch.py`

**Interfaces:**
- Produces: `fetch_zip_member(url: str, member: str, dest: Path, sha256: str, client: httpx.Client | None = None) -> Path`; `class RangeNotSupported(RuntimeError)`; `FileSpec.member: str | None = None`. `fetch_dataset` saves a member at `raw_dir / member` (the member path, which keeps forum subdirectories distinct) and a plain file at `raw_dir / <url basename>` as today.

- [ ] **Step 1: Write the failing tests** (append to `harness/tests/test_fetch.py`)

```python
import io
import re
import zipfile

from irbench.datasets.fetch import RangeNotSupported, fetch_zip_member
from irbench.datasets.registry import DatasetSpec, fetch_dataset


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, body in members.items():
            z.writestr(name, body)
    return buf.getvalue()


def _range_client(blob: bytes, calls: list[str], ranges: bool = True) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers.get("range", "full"))
        m = re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers.get("range", ""))
        if not ranges or not m:
            return httpx.Response(200, content=blob)
        start, end = int(m[1]), min(int(m[2]), len(blob) - 1)
        return httpx.Response(
            206,
            content=blob[start : end + 1],
            headers={"content-range": f"bytes {start}-{end}/{len(blob)}"},
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


MEMBERS = {
    "set/a/corpus.jsonl": b'{"_id": "1"}\n' * 5000,
    "set/b/corpus.jsonl": b"other\n",
}


def test_zip_member_is_extracted_and_verified(tmp_path):
    blob, calls = _zip_bytes(MEMBERS), []
    body = MEMBERS["set/a/corpus.jsonl"]
    dest = fetch_zip_member(
        "https://x/set.zip", "set/a/corpus.jsonl", tmp_path / "a.jsonl",
        hashlib.sha256(body).hexdigest(), _range_client(blob, calls),
    )
    assert dest.read_bytes() == body
    assert calls and "full" not in calls  # only range requests


def test_large_member_streams_over_many_range_requests(tmp_path):
    body = bytes(range(256)) * 4000  # ~1 MB, barely compressible
    blob, calls = _zip_bytes({"big.bin": body}), []
    dest = fetch_zip_member(
        "https://x/set.zip", "big.bin", tmp_path / "big.bin", hashlib.sha256(body).hexdigest(),
        _range_client(blob, calls), buffer_size=64 << 10,
    )
    assert dest.read_bytes() == body
    assert len(calls) > 10  # read in chunks, never as one download


def test_zip_member_checksum_mismatch_leaves_no_file(tmp_path):
    blob = _zip_bytes(MEMBERS)
    with pytest.raises(ChecksumError):
        fetch_zip_member(
            "https://x/set.zip", "set/a/corpus.jsonl", tmp_path / "a.jsonl", "0" * 64,
            _range_client(blob, []),
        )
    assert not (tmp_path / "a.jsonl").exists()
    assert not list(tmp_path.glob("*.part"))


def test_zip_member_needs_range_support(tmp_path):
    blob = _zip_bytes(MEMBERS)
    with pytest.raises(RangeNotSupported, match="https://x/set.zip"):
        fetch_zip_member(
            "https://x/set.zip", "set/a/corpus.jsonl", tmp_path / "a.jsonl", "0" * 64,
            _range_client(blob, [], ranges=False),
        )


def test_zip_member_already_valid_is_not_refetched(tmp_path):
    body = MEMBERS["set/b/corpus.jsonl"]
    dest = tmp_path / "b.jsonl"
    dest.write_bytes(body)
    calls: list[str] = []
    fetch_zip_member(
        "https://x/set.zip", "set/b/corpus.jsonl", dest, hashlib.sha256(body).hexdigest(),
        _range_client(_zip_bytes(MEMBERS), calls),
    )
    assert calls == []


def test_fetch_dataset_stores_members_under_their_path(tmp_path):
    blob = _zip_bytes(MEMBERS)
    spec = DatasetSpec.model_validate({
        "id": "set", "version": "1", "license": "x", "source": "x",
        "files": {
            "a": {"url": "https://x/set.zip", "member": "set/a/corpus.jsonl",
                  "sha256": hashlib.sha256(MEMBERS["set/a/corpus.jsonl"]).hexdigest()},
            "b": {"url": "https://x/set.zip", "member": "set/b/corpus.jsonl",
                  "sha256": hashlib.sha256(MEMBERS["set/b/corpus.jsonl"]).hexdigest()},
        },
    })
    paths = fetch_dataset(spec, tmp_path / "raw", _range_client(blob, []))
    assert paths["a"] == tmp_path / "raw" / "set" / "a" / "corpus.jsonl"
    assert paths["b"].read_bytes() == MEMBERS["set/b/corpus.jsonl"]
```

- [ ] **Step 2: Run them, expect FAIL** (`ImportError: cannot import name 'RangeNotSupported'`)

Run: `uv run pytest harness/tests/test_fetch.py -q`

- [ ] **Step 3: Implement** in `harness/irbench/datasets/fetch.py` (keep `fetch_file` as is; add below it)

```python
import io
import re
import zipfile


class RangeNotSupported(RuntimeError):
    pass


class _HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file, one HTTP range request per read."""

    def __init__(self, url: str, client: httpx.Client):
        self.url, self.client, self.pos = url, client, 0
        self.size = self._total()

    def _get(self, start: int, end: int) -> httpx.Response:
        r = self.client.get(self.url, headers={"Range": f"bytes={start}-{end}"})
        if r.status_code != 206:
            raise RangeNotSupported(
                f"{self.url}: the server answered {r.status_code} to a range request; "
                "single files cannot be read out of this archive"
            )
        return r

    def _total(self) -> int:
        m = re.search(r"/(\d+)$", self._get(0, 0).headers.get("content-range", ""))
        if not m:
            raise RangeNotSupported(f"{self.url}: no Content-Range total in the reply")
        return int(m[1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer) -> int:  # type: ignore[override]
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        data = self._get(self.pos, end).content
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch_zip_member(
    url: str,
    member: str,
    dest: Path,
    sha256: str,
    client: httpx.Client | None = None,
    buffer_size: int = 8 << 20,
) -> Path:
    """Extract one member of a remote zip, reading only the bytes it needs. The member is
    streamed to disk in chunks; `buffer_size` is the size of each range request."""
    if dest.exists() and sha256_file(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    owned = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=120)
    try:
        remote = io.BufferedReader(_HttpRangeFile(url, http), buffer_size=buffer_size)
        with zipfile.ZipFile(remote) as archive, archive.open(member) as src, tmp.open("wb") as out:
            for chunk in iter(lambda: src.read(1 << 20), b""):
                out.write(chunk)
    finally:
        if owned:
            http.close()
    actual = sha256_file(tmp)
    if actual != sha256:
        tmp.unlink()
        raise ChecksumError(f"{url}!{member}: expected sha256 {sha256}, got {actual}")
    tmp.replace(dest)
    return dest
```

Wrap the extraction in `try/except BaseException: tmp.unlink(missing_ok=True); raise` so an interrupted or failed extraction leaves no `.part` file (the checksum test asserts none remains).

In `harness/irbench/datasets/registry.py`:

```python
class FileSpec(BaseModel):
    url: str
    sha256: str
    # A path inside the zip at `url`; only that member is fetched (range requests).
    member: str | None = None


def fetch_dataset(
    spec: DatasetSpec, raw_dir: Path, client: httpx.Client | None = None
) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for name, f in spec.files.items():
        if f.member:
            out[name] = fetch_zip_member(f.url, f.member, raw_dir / f.member, f.sha256, client)
        else:
            dest = raw_dir / Path(urlparse(f.url).path).name
            out[name] = fetch_file(f.url, dest, f.sha256, client)
    return out
```

(import `fetch_zip_member` from `irbench.datasets.fetch`.)

- [ ] **Step 4: Run tests, expect PASS** — `uv run pytest harness/tests/test_fetch.py -q`, then the full suite.

- [ ] **Step 5: Commit** — `git commit -m "feat: fetch single files out of a remote zip by range requests"`

---

### Task 2: Scenarios without synonyms

**Files:**
- Modify: `harness/irbench/scenario.py`, `harness/irbench/stages/context.py`, `harness/irbench/stages/prepare.py`, `engines/postgres/adapter.py`
- Modify: `harness/irbench/report/render.py`, `harness/irbench/report/templates/scenario.html.j2`, `harness/irbench/report/templates/run.html.j2`
- Modify: `harness/tests/adapter_helpers.py` (a `synonyms` override)
- Test: `harness/tests/test_scenario.py`, `harness/tests/test_adapter_*.py`, `harness/tests/test_report_render.py`

**Interfaces:**
- Produces: `Scenario.synonyms: str | None = None`; `Scenario.synonyms_path -> Path | None`; `PerturbationSpec.vocabulary: str | None = None`; `Scenario.vocabulary_path -> Path | None`; `scenario_term_groups(path: Path | None) -> list[list[str]]` in `irbench.terms` (empty list for `None`). `make_adapter(engine, track=None, dim=3, n_docs=None, synonyms=None)` in `adapter_helpers` (`synonyms=[]` forces none). Report: `scenario_view(...)["perturbation_kinds"]: list[str]` — the perturbation kinds present in any run's rows, in `("typo", "prefix", "synonym")` order.

- [ ] **Step 1: Failing tests**

`harness/tests/test_scenario.py`:

```python
def _no_synonyms(raw):
    raw.pop("synonyms")
    raw["perturbations"] = {"seed": 1, "kinds": ["typo", "prefix"]}
    return raw


def test_scenario_without_synonyms_or_vocabulary(tmp_path):
    root = copy_scenario(tmp_path, _no_synonyms)  # helper: copy the WANDS scenario dir, edit yaml
    s = load_scenario(root, "ecommerce/product-search")
    assert s.synonyms_path is None and s.vocabulary_path is None


def test_synonym_kind_requires_a_vocabulary(tmp_path):
    def edit(raw):
        raw["perturbations"] = {"seed": 1, "kinds": ["typo", "synonym"]}
        return raw

    with pytest.raises(ValueError, match="vocabulary"):
        load_scenario(copy_scenario(tmp_path, edit), "ecommerce/product-search")
```

If `test_scenario.py` has no `copy_scenario` helper, add one: copy `REPO_ROOT / "scenarios" / "ecommerce" / "product-search"` into `tmp_path / "scenarios" / "ecommerce" / "product-search"`, load the yaml, apply `edit`, write it back, return `tmp_path`.

One test per adapter file (`test_adapter_meilisearch.py`, `test_adapter_elasticsearch.py`, `test_adapter_postgres.py`, `test_adapter_paradedb.py`, `test_adapter_qdrant.py`), using `make_adapter(engine, synonyms=[])`:

```python
def test_no_synonyms_creates_no_synonym_setting():  # meilisearch
    assert make_adapter("meilisearch", synonyms=[]).index_settings()["synonyms"] == {}

def test_no_synonyms_creates_no_synonym_filter():  # elasticsearch
    body = make_adapter("elasticsearch", synonyms=[]).index_body()  # the method that builds the create-index body
    assert "irbench_synonyms" not in json.dumps(body)

def test_no_synonyms_creates_no_thesaurus():  # postgres
    ddl = make_adapter("postgres", synonyms=[]).setup_ddl()
    assert "thesaurus" not in ddl and "_ths" not in ddl.split("DROP TEXT SEARCH DICTIONARY")[-1].split("\n", 1)[1]
```

For Elasticsearch use the adapter's existing method that returns the index body (read `engines/elasticsearch/adapter.py`; the synonym filter is built near line 63). For ParadeDB and Qdrant, assert index setup succeeds and matches the default-synonyms output (they load none either way): `assert make_adapter("qdrant", synonyms=[]).<index body method>() == make_adapter("qdrant").<same>()`.

Postgres expectation, stated plainly: with no thesaurus groups, `setup_ddl()` contains no `CREATE TEXT SEARCH DICTIONARY` and no `ALTER TEXT SEARCH CONFIGURATION … WITH …_ths` (the `DROP … IF EXISTS` lines stay). Write the assertion as `assert "CREATE TEXT SEARCH DICTIONARY" not in ddl`.

`harness/tests/test_report_render.py`: build a site from bundles whose relevance rows have only `all`, `original`, `typo`, `prefix` perturbations (filter `report_fixtures.relevance(...)` output to drop `synonym` rows); assert the scenario page's "Imperfect queries" header has `typo` and `prefix` columns and no `synonym` column, and the WANDS-shaped bundles (with synonym rows) still show the synonym column.

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement**

`scenario.py`:

```python
class PerturbationSpec(BaseModel):
    seed: int
    kinds: list[Literal["typo", "prefix", "synonym"]]
    vocabulary: str | None = None

    @model_validator(mode="after")
    def _vocabulary_for_synonyms(self) -> "PerturbationSpec":
        if "synonym" in self.kinds and not self.vocabulary:
            raise ValueError("perturbation kind 'synonym' needs a `vocabulary` file")
        return self
```

`Scenario.synonyms: str | None = None`; properties return `None` when unset:

```python
    @property
    def vocabulary_path(self) -> Path | None:
        v = self.perturbations.vocabulary
        return None if v is None else self._directory / v

    @property
    def synonyms_path(self) -> Path | None:
        return None if self.synonyms is None else self._directory / self.synonyms
```

`irbench/terms.py`:

```python
def scenario_term_groups(path: Path | None) -> list[list[str]]:
    """Term groups of a scenario file, or none when the scenario has no such file."""
    return [] if path is None else load_term_groups(path)
```

Use it in `stages/context.py` (`synonyms=scenario_term_groups(self.scenario.synonyms_path)`), `stages/prepare.py` (`vocabulary = scenario_term_groups(scenario.vocabulary_path)`) and `adapter_helpers.make_adapter` (`synonyms=scenario_term_groups(scenario.synonyms_path) if synonyms is None else synonyms`). Grep for every other `synonyms_path` / `vocabulary_path` use (`grep -rn "synonyms_path\|vocabulary_path" harness engines`) and route each through `scenario_term_groups`.

`engines/postgres/adapter.py` `setup_ddl`: create the thesaurus only when there is something in it:

```python
        ths = (
            ""
            if self.ctx.config.synonyms or not self.thesaurus_groups()
            else ( ...unchanged... )
        )
```

Also skip writing the thesaurus file when `thesaurus_groups()` is empty (find where `thesaurus_file()` is written into the container, `grep -n thesaurus_file engines/postgres`).

Report: in `render.py` `scenario_view`, add

```python
        "perturbation_kinds": [
            p for p in ("typo", "prefix", "synonym")
            if any(r["deltas"].get(p) is not None for r in perturbation_rows(s))
        ],
```

and in `scenario.html.j2` replace the fixed `typo/prefix/synonym` header cells and row cells with loops over `view.perturbation_kinds`, keeping the synonym dagger note on the synonym cell only. In `run.html.j2` (line ~72) loop over the perturbations that have a value in `r.by_perturbation` for any row, instead of the fixed list. Keep the introductory sentence accurate: say "or use a synonym" only when `"synonym" in view.perturbation_kinds`.

- [ ] **Step 4: Run tests, expect PASS** (full suite: WANDS pages unchanged except as asserted).

- [ ] **Step 5: Commit** — `git commit -m "feat: scenarios may have no synonyms or synonym perturbations"`

---

### Task 3: Query groups

**Files:**
- Modify: `harness/irbench/models.py`, `harness/irbench/perturb/variants.py`, `harness/irbench/tasks.py`, `harness/irbench/stages/prepare.py`, `harness/irbench/score/relevance.py`, `harness/irbench/stages/score_run.py`
- Modify: `harness/irbench/report/data.py`, `harness/irbench/report/render.py`, `harness/irbench/report/templates/scenario.html.j2`
- Test: `harness/tests/test_variants.py`, `harness/tests/test_tasks.py`, `harness/tests/test_score_relevance.py`, `harness/tests/test_report_data.py`, `harness/tests/test_report_render.py`, `harness/tests/test_prepare.py`

**Interfaces:**
- Produces: `Query.group: str | None = None`; `QueryTask.group: str | None = None`; `GroupScore.group: str | None = None`; `score_relevance` returns the existing rows first, then one row per `(mode, base, group)` with `perturbation="all"` for tasks that have a group; `score_stage` serializes rows with `model_dump(exclude_none=True)` so ungrouped rows have no `group` key. `RunResult.ndcg(group, perturbation="all")` and every other reader consider only rows without a `group` key. New `RunResult.group_ndcg(group: str, query_group: str) -> float | None`. `scenario_view(...)["query_groups"]: list[dict]` (see Step 3).

- [ ] **Step 1: Failing tests**

`test_variants.py`:

```python
def test_variants_keep_the_query_group():
    q = Query(qid="1", text="coronavirus origin", variant_of="1", perturbation="original", group="keyword")
    variants = make_variants([q], 7, ["typo", "prefix"], [])
    assert {v.group for v in variants} == {"keyword"}
```

`test_tasks.py`: build tasks from one grouped query (use the existing test's scenario/doc fixtures) and assert every task has `group == "keyword"`; an ungrouped query's tasks have `group is None` and `model_dump(mode="json", exclude_none=True)` has no `group` key.

`test_score_relevance.py`:

```python
def test_group_rows_are_added_after_the_existing_rows():
    tasks = [make_task("a", group="question"), make_task("b", group="keyword")]  # keyword/keyword tasks
    rows = score_relevance(tasks, QRELS, RUN)
    plain = [r for r in rows if r.group is None]
    grouped = [r for r in rows if r.group is not None]
    assert rows[: len(plain)] == plain
    assert {(r.mode, r.base, r.perturbation, r.group) for r in grouped} == {
        ("keyword", "keyword", "all", "question"),
        ("keyword", "keyword", "all", "keyword"),
    }
    assert sum(r.n for r in grouped) == next(r for r in plain if r.perturbation == "all").n


def test_ungrouped_tasks_produce_exactly_todays_rows():
    tasks = [make_task("a"), make_task("b")]
    assert all(r.group is None for r in score_relevance(tasks, QRELS, RUN))
```

(`make_task`, `QRELS`, `RUN`: reuse or add small helpers in that file building `QueryTask`s with `mode=base="keyword"`, qrels `{task_id: {"d1": 1}}` and a run `{task_id: ["d1"]}`.)

Golden regression in `test_score_relevance.py`: score a fixed ungrouped task list and compare `[r.model_dump(exclude_none=True) for r in rows]` with the same list computed from the rows' fields `mode, base, perturbation, n, metrics` only (i.e. assert no row has a `group` key).

`test_report_data.py`:

```python
def test_grouped_rows_never_replace_the_overall_score(tmp_path):
    rows = relevance({"keyword/keyword": 0.5})
    # A grouped row with the same mode/base/perturbation, placed FIRST to catch order bugs.
    rows.insert(0, {**rows[0], "group": "keyword", "metrics": {**rows[0]["metrics"], "ndcg@10": 0.99}})
    run = load_single_run(tmp_path, rows)  # write_bundle + load_results helper
    assert run.ndcg("keyword/keyword") == 0.5
    assert run.group_ndcg("keyword/keyword", "keyword") == 0.99
```

Also assert the bundle validator accepts a relevance item with a string `group` and rejects a non-string `group` (`ValueError` mentioning `group`).

`test_report_render.py`: a scenario whose runs carry grouped rows for `keyword/keyword` with groups `question` and `keyword` renders a "By query group" table with one row per run and one column per group; WANDS-shaped bundles render no such table.

`test_prepare.py`: a normalized `queries.jsonl` row with `"group": "keyword"` reaches `queries.variants.jsonl` and `tasks.jsonl` with that group (write a small fixture dataset through a test-only normalizer registered with `monkeypatch.setitem(NORMALIZERS, ...)`, or extend the WANDS fixture path; keep WANDS rows ungrouped).

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement**

`models.py`: add `group: str | None = None` to `Query` and `QueryTask`.

`variants.py`: the perturbed `Query(...)` gets `group=query.group`.

`tasks.py` `add`: pass `group=query.group`.

`prepare.py`: build queries with `group=r.get("group")`; when writing variants and tasks use `model_dump(exclude_none=True)` / `model_dump(mode="json", exclude_none=True)` so WANDS files keep today's shape. Check that readers (`QueryTask.model_validate`) still accept the files (the field defaults to `None`).

`score/relevance.py`:

```python
class GroupScore(BaseModel):
    mode: str
    base: str
    perturbation: str
    n: int
    metrics: dict[str, float]
    group: str | None = None


def score_relevance(tasks: list[QueryTask], qrels: Qrels, run: RunResults) -> list[GroupScore]:
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    by_query_group: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for task in tasks:
        if task.task_id not in qrels:
            raise ValueError(f"task {task.task_id} has no qrels")
        groups[(task.mode, task.base, "all")].append(task.task_id)
        groups[(task.mode, task.base, task.perturbation)].append(task.task_id)
        if task.group is not None:
            by_query_group[(task.mode, task.base, task.group)].append(task.task_id)
    rows = [
        GroupScore(mode=m, base=b, perturbation=p, n=len(ids), metrics=_evaluate(ids, qrels, run))
        for (m, b, p), ids in sorted(groups.items())
    ]
    rows += [
        GroupScore(
            mode=m, base=b, perturbation="all", group=g, n=len(ids),
            metrics=_evaluate(ids, qrels, run),
        )
        for (m, b, g), ids in sorted(by_query_group.items())
    ]
    return rows
```

`score_run.py`: `"relevance": [g.model_dump(exclude_none=True) for g in ...]`.

`report/data.py`: in `ndcg`, the `groups` property and `_validate_relevance_item`, only rows with no `group` key count for the existing lookups (`if "group" in g: continue`). Validation: when present, `group` must be a `str`. Add:

```python
    def group_ndcg(self, group: str, query_group: str) -> float | None:
        if self.scores is None:
            return None
        mode, _, base = group.partition("/")
        for g in self.scores["relevance"]:
            if (g["mode"], g["base"], g.get("group")) == (mode, base, query_group):
                return g["metrics"]["ndcg@10"]
        return None

    @property
    def query_groups(self) -> list[str]:
        if self.scores is None:
            return []
        return sorted({g["group"] for g in self.scores["relevance"] if "group" in g})
```

`report/render.py`: `run_view` skips rows with a `group` key when picking `all_metrics`. `scenario_view` adds:

```python
        "query_groups": query_group_tables(s),
```

```python
def query_group_tables(s: ScenarioResults) -> list[dict[str, Any]]:
    """nDCG@10 per query group, per retrieval base (empty when the scenario has no groups)."""
    names = sorted({q for r in s.runs for q in r.query_groups})
    if not names:
        return []
    tables = []
    for group in BASE_GROUPS:
        rows = [
            {"run": r, "cells": [r.group_ndcg(group, q) for q in names]}
            for r in s.runs
            if not r.failure and any(r.group_ndcg(group, q) is not None for q in names)
        ]
        if rows:
            tables.append({"group": group, "names": names, "rows": rows})
    return tables
```

`scenario.html.j2`: after the "Imperfect queries" section, when `view.query_groups`, a section "By query group" with the note "nDCG@10 for each group of queries (for example question vs keyword form); n per group is on the run page." and one table per base: header `Engine | Model | <group names>`, cells formatted with the existing `score` filter.

- [ ] **Step 4: Run tests, expect PASS.** Then regenerate the local WANDS scores with the new code and confirm they are identical to the current ones (no `group` keys, same numbers):

```bash
for d in .irbench/runs/ecommerce__product-search__*; do cp $d/scores.json /tmp/before-$(basename $d).json; done
for d in .irbench/runs/ecommerce__product-search__*; do uv run irbench score --run-dir $d >/dev/null; done   # use the CLI's actual re-score command (check `uv run irbench --help`)
for d in .irbench/runs/ecommerce__product-search__*; do python3 -c "import json,sys;a=json.load(open(sys.argv[1]));b=json.load(open(sys.argv[2]));assert a['relevance']==b['relevance'],sys.argv[2]" /tmp/before-$(basename $d).json $d/scores.json; done
```

If the CLI has no stand-alone re-score command, skip this step and note it in the report; the controller runs it in Task 7.

- [ ] **Step 5: Commit** — `git commit -m "feat: query groups, scored and reported separately"`

---

### Task 4: BEIR reader, CQADupStack and LoTTE

**Files:**
- Create: `harness/irbench/datasets/beir.py`, `harness/irbench/datasets/cqadupstack.py`, `harness/irbench/datasets/lotte.py`
- Modify: `harness/irbench/datasets/wands.py` (`NormalizedDataset.dropped`), `harness/irbench/datasets/__init__.py`, `harness/irbench/stages/prepare.py` (manifest counts)
- Create: `datasets/cqadupstack/dataset.yaml`, `datasets/lotte-lifestyle/dataset.yaml`
- Test: `harness/tests/test_beir.py` (new), `harness/tests/test_prepare.py`

**Interfaces:**
- Consumes: Task 1 (`FileSpec.member`), Task 3 (`group` in normalized queries).
- Produces:
  - `NormalizedDataset` gains `dropped: dict[str, int] = field(default_factory=dict)` (keys `docs`, `judgments`); `prepare` writes `counts.dropped_docs` and `counts.dropped_judgments` into `prepare.manifest.json` (0 for WANDS).
  - `beir.read_jsonl_rows(path) -> Iterator[dict]`, `beir.read_qrels_tsv(path) -> Qrels` (header skipped; negative grades become 0), `beir.write_normalized(out_dir, docs: Iterable[dict], queries: Iterable[dict], qrels: Qrels, text_fields: list[str]) -> NormalizedDataset` — drops docs whose `text_fields` are all empty, drops judgments to dropped or unknown docs, counts both, writes `docs.jsonl`, `queries.jsonl`, `qrels.tsv`.
  - `normalize_cqadupstack(raw: dict[str, Path], out_dir: Path) -> NormalizedDataset` with raw keys `<forum>_corpus`, `<forum>_queries`, `<forum>_qrels` for forums `webmasters`, `android`, `wordpress`.
  - `normalize_lotte(raw, out_dir)` with raw keys `collection`, `qas`.
  - `NORMALIZERS["cqadupstack"]`, `NORMALIZERS["lotte-lifestyle"]`.

- [ ] **Step 1: Failing tests** (`harness/tests/test_beir.py`)

```python
import json

from irbench.datasets.cqadupstack import FORUMS, normalize_cqadupstack
from irbench.datasets.lotte import normalize_lotte
from irbench.io import read_jsonl, read_qrels


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _cqa_raw(tmp_path):
    raw = {}
    for forum in FORUMS:
        d = tmp_path / forum
        d.mkdir()
        raw[f"{forum}_corpus"] = _jsonl(d / "corpus.jsonl", [
            {"_id": "1", "title": f"{forum} title", "text": "body", "metadata": {"tags": ["php", "seo"]}},
            {"_id": "2", "title": "", "text": "", "metadata": {"tags": []}},
        ])
        raw[f"{forum}_queries"] = _jsonl(d / "queries.jsonl", [
            {"_id": "9", "text": "how do I x?", "metadata": {"tags": ["x"], "123": {"body": "huge"}}},
            {"_id": "8", "text": "only empty", "metadata": {}},
        ])
        (d / "test.tsv").write_text("query-id\tcorpus-id\tscore\n9\t1\t1\n8\t2\t1\n")
        raw[f"{forum}_qrels"] = d / "test.tsv"
    return raw


def test_cqadupstack_prefixes_ids_and_keeps_forum_and_tags(tmp_path):
    out = normalize_cqadupstack(_cqa_raw(tmp_path), tmp_path / "out")
    docs = list(read_jsonl(out.docs))
    assert {d["id"] for d in docs} == {f"{f}-1" for f in FORUMS}
    doc = next(d for d in docs if d["id"] == "webmasters-1")
    assert doc == {"id": "webmasters-1", "title": "webmasters title", "text": "body",
                   "forum": "webmasters", "tags": ["php", "seo"]}
    queries = list(read_jsonl(out.queries))
    assert {q["qid"] for q in queries} == {f"{f}-{i}" for f in FORUMS for i in ("9", "8")}
    assert all(set(q) == {"qid", "text"} for q in queries)  # bulky metadata is not carried
    qrels = read_qrels(out.qrels)
    assert qrels["android-9"] == {"android-1": 1}
    assert "android-8" not in qrels  # its only judgment pointed at an empty doc
    assert out.dropped == {"docs": 3, "judgments": 3}


def test_lotte_reads_answers_and_groups_by_forum(tmp_path):
    coll = _jsonl(tmp_path / "c.jsonl", [
        {"doc_id": 0, "author": "a", "text": "Use a cast iron pan."},
        {"doc_id": 1, "author": "b", "text": "Visas are needed."},
        {"doc_id": 2, "author": "c", "text": "  "},
    ])
    qas = _jsonl(tmp_path / "q.jsonl", [
        {"qid": 0, "query": "best pan for steak?", "url": "https://cooking.stackexchange.com/questions/1/x",
         "answer_pids": [0, 2], "question_author": ""},
        {"qid": 1, "query": "do I need a visa?", "url": "https://travel.stackexchange.com/q/2",
         "answer_pids": [1], "question_author": ""},
    ])
    out = normalize_lotte({"collection": coll, "qas": qas}, tmp_path / "out")
    assert list(read_jsonl(out.docs))[0] == {"id": "0", "text": "Use a cast iron pan."}
    assert [(q["qid"], q["group"]) for q in read_jsonl(out.queries)] == [("0", "cooking"), ("1", "travel")]
    assert read_qrels(out.qrels) == {"0": {"0": 1}, "1": {"1": 1}}
    assert out.dropped == {"docs": 1, "judgments": 1}
```

Plus in `test_beir.py`: `read_qrels_tsv` turns a `-1` grade into `0` and skips the header.

Plus in `test_prepare.py` (Review Focus 5): preparing a dataset where one query's only judgment points at a dropped doc lists that query in `skipped_queries` and the manifest has `counts.dropped_docs` and `counts.dropped_judgments`; WANDS manifests have both at 0.

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement**

`datasets/wands.py`: `NormalizedDataset` gains `dropped: dict[str, int] = field(default_factory=dict)` (import `field`); `normalize_wands` returns `dropped={"docs": 0, "judgments": 0}`.

`datasets/beir.py`:

```python
"""Readers for BEIR-style files (corpus.jsonl, queries.jsonl, qrels TSV) and the shared
normalization rules of spec 2026-10-14-irbench-knowledge-base §3.4."""

import csv
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from irbench.datasets.wands import NormalizedDataset
from irbench.io import write_jsonl, write_qrels
from irbench.models import Qrels


def read_jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_qrels_tsv(path: Path) -> Qrels:
    qrels: Qrels = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            grade = max(0, int(row["score"]))
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = grade
    return qrels


def _has_text(doc: dict[str, Any], fields: list[str]) -> bool:
    return any(str(doc.get(f) or "").strip() for f in fields)


def write_normalized(
    out_dir: Path,
    docs: Iterable[dict[str, Any]],
    queries: Iterable[dict[str, Any]],
    qrels: Qrels,
    text_fields: list[str],
) -> NormalizedDataset:
    kept: set[str] = set()
    dropped_docs = 0

    def keep(rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        nonlocal dropped_docs
        for d in rows:
            if _has_text(d, text_fields):
                kept.add(d["id"])
                yield d
            else:
                dropped_docs += 1

    docs_path, queries_path, qrels_path = (
        out_dir / "docs.jsonl", out_dir / "queries.jsonl", out_dir / "qrels.tsv"
    )
    n_docs = write_jsonl(docs_path, keep(docs))
    n_queries = write_jsonl(queries_path, ({"qid": q["qid"], **{k: v for k, v in q.items() if k != "qid"}} for q in queries))
    dropped_judgments = 0
    clean: Qrels = {}
    for qid, grades in qrels.items():
        for doc_id, grade in grades.items():
            if doc_id in kept:
                clean.setdefault(qid, {})[doc_id] = grade
            else:
                dropped_judgments += 1
    write_qrels(qrels_path, clean)
    return NormalizedDataset(
        docs_path, queries_path, qrels_path, n_docs, n_queries,
        dropped={"docs": dropped_docs, "judgments": dropped_judgments},
    )
```

Simplify the `n_queries` line to `write_jsonl(queries_path, queries)` if queries are already built as plain dicts by the caller (they are).

`datasets/cqadupstack.py`:

```python
from pathlib import Path

from irbench.datasets.beir import read_jsonl_rows, read_qrels_tsv, write_normalized
from irbench.datasets.wands import NormalizedDataset
from irbench.models import Qrels

FORUMS = ("webmasters", "android", "wordpress")


def normalize_cqadupstack(raw: dict[str, Path], out_dir: Path) -> NormalizedDataset:
    """Three CQADupStack forums in one index. Post and query ids are only unique within a
    forum, so every id gets a `<forum>-` prefix."""

    def docs():
        for forum in FORUMS:
            for r in read_jsonl_rows(raw[f"{forum}_corpus"]):
                yield {
                    "id": f"{forum}-{r['_id']}",
                    "title": r.get("title") or "",
                    "text": r.get("text") or "",
                    "forum": forum,
                    "tags": list((r.get("metadata") or {}).get("tags") or []),
                }

    def queries():
        for forum in FORUMS:
            for r in read_jsonl_rows(raw[f"{forum}_queries"]):
                # BEIR's query metadata embeds whole post bodies; only the text is used.
                yield {"qid": f"{forum}-{r['_id']}", "text": " ".join(r["text"].split())}

    qrels: Qrels = {}
    for forum in FORUMS:
        for qid, grades in read_qrels_tsv(raw[f"{forum}_qrels"]).items():
            qrels[f"{forum}-{qid}"] = {f"{forum}-{d}": g for d, g in grades.items()}
    return write_normalized(out_dir, docs(), queries(), qrels, ["title", "text"])
```

Only queries that appear in the test qrels are kept: filter `queries()` to `qid in qrels` (BEIR's `queries.jsonl` may hold more than the test split).

`datasets/lotte.py`:

```python
from pathlib import Path
from urllib.parse import urlparse

from irbench.datasets.beir import read_jsonl_rows, write_normalized
from irbench.datasets.wands import NormalizedDataset
from irbench.models import Qrels


def _forum(url: str) -> str:
    """`https://cooking.stackexchange.com/...` -> `cooking`."""
    return urlparse(url).netloc.split(".")[0]


def normalize_lotte(raw: dict[str, Path], out_dir: Path) -> NormalizedDataset:
    docs = (
        {"id": str(r["doc_id"]), "text": r["text"]} for r in read_jsonl_rows(raw["collection"])
    )
    queries, qrels = [], {}
    for r in read_jsonl_rows(raw["qas"]):
        qid = str(r["qid"])
        queries.append({"qid": qid, "text": " ".join(r["query"].split()), "group": _forum(r["url"])})
        qrels[qid] = {str(pid): 1 for pid in r["answer_pids"]}
    return write_normalized(out_dir, docs, queries, qrels, ["text"])
```

`datasets/__init__.py`: register `"cqadupstack": normalize_cqadupstack`, `"lotte-lifestyle": normalize_lotte`.

`stages/prepare.py` manifest `counts`: add `"dropped_docs": normalized.dropped.get("docs", 0)` and `"dropped_judgments": normalized.dropped.get("judgments", 0)`.

`datasets/cqadupstack/dataset.yaml` — use the sha256 values from `/private/tmp/claude-501/-Users-quentindequelen/31ceb8e5-8b2a-46d2-abed-76ee88c1dc84/scratchpad/hashes.json` (key `cqadupstack.zip:<member>`, first element); the controller pastes them into the task dispatch if that file is not readable:

```yaml
id: cqadupstack
version: "beir-2021"
license: CC BY-SA 4.0 (StackExchange content, redistributed by BEIR)
source: https://github.com/beir-cellar/beir
files:
  webmasters_corpus:
    url: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip
    member: cqadupstack/webmasters/corpus.jsonl
    sha256: <hashes.json>
  webmasters_queries: {url: ..., member: cqadupstack/webmasters/queries.jsonl, sha256: ...}
  webmasters_qrels: {url: ..., member: cqadupstack/webmasters/qrels/test.tsv, sha256: ...}
  # same three entries for android and wordpress
```

`datasets/lotte-lifestyle/dataset.yaml`:

```yaml
id: lotte-lifestyle
version: "colbertv2-lotte@1a87807d+passages@3be7f857"
license: Apache-2.0 (Hugging Face card); StackExchange content, CC BY-SA
source: https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md
files:
  collection:
    url: https://huggingface.co/datasets/colbertv2/lotte_passages/resolve/3be7f857585299f1268d29d3591202d731ea84a1/lifestyle/test_collection.jsonl
    sha256: 6afca45a7a83ceaf5f1144631c57ab7796750037732970534511dab49a79ad9b
  qas:
    url: https://huggingface.co/datasets/colbertv2/lotte/resolve/1a87807da631e4197d77f7e720c38941abcf26d1/lifestyle/test_qas.search.jsonl
    sha256: 4215b80512270e328c0f267b5b4b60b4a8914e34d4b5182994822acc53cdd539
```

- [ ] **Step 4: Run tests, expect PASS** (full suite).

- [ ] **Step 5: Commit** — `git commit -m "feat: CQADupStack and LoTTE datasets on a shared BEIR reader"`

---

### Task 5: TREC-COVID with the CORD-19 metadata join

**Files:**
- Create: `harness/irbench/datasets/trec_covid.py`, `datasets/trec-covid/dataset.yaml`
- Modify: `harness/irbench/datasets/__init__.py`
- Test: `harness/tests/test_beir.py`

**Interfaces:**
- Consumes: Task 4 (`beir.read_jsonl_rows`, `read_qrels_tsv`, `write_normalized`), Task 1 (members).
- Produces: `normalize_trec_covid(raw, out_dir)` with raw keys `corpus`, `queries`, `qrels`, `metadata`; `publish_fields(publish_time: str) -> tuple[int | None, int | None]`; `NORMALIZERS["trec-covid"]`.

- [ ] **Step 1: Failing tests** (append to `test_beir.py`)

```python
import pytest

from irbench.datasets.trec_covid import normalize_trec_covid, publish_fields

METADATA_CSV = (
    "cord_uid,sha,source_x,title,doi,pmcid,pubmed_id,license,abstract,publish_time,authors,journal\n"
    "aaa,,Medline; PMC,T1,,,,cc-by,A1,2020-03-15,X,Lancet\n"
    "aaa,,WHO,T1 dup,,,,cc-by,A1,2019-01-01,X,Other\n"
    "bbb,,ArXiv,T2,,,,arxiv,A2,2020,Y,\n"
    "ccc,,PMC,T3,,,,cc,A3,,Z,BMJ\n"
)


@pytest.mark.parametrize("value, expected", [
    ("2020-03-15", (2020, 20200315)), ("2020", (2020, 20200101)), ("", (None, None)),
    ("2020-3-5", (2020, 20200305)),
])
def test_publish_fields(value, expected):
    assert publish_fields(value) == expected


def test_trec_covid_joins_metadata_and_splits_query_forms(tmp_path):
    corpus = _jsonl(tmp_path / "corpus.jsonl", [
        {"_id": "aaa", "title": "T1", "text": "A1", "metadata": {"url": "u", "pubmed_id": "1"}},
        {"_id": "bbb", "title": "T2", "text": "A2", "metadata": {}},
        {"_id": "ccc", "title": "T3", "text": "", "metadata": {}},
        {"_id": "zzz", "title": "", "text": "", "metadata": {}},
    ])
    queries = _jsonl(tmp_path / "queries.jsonl", [
        {"_id": "1", "text": "what is the origin of COVID-19",
         "metadata": {"query": "coronavirus origin", "narrative": "n"}},
    ])
    (tmp_path / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n1\taaa\t2\n1\tbbb\t-1\n1\tzzz\t1\n"
    )
    (tmp_path / "metadata.csv").write_text(METADATA_CSV)
    out = normalize_trec_covid(
        {"corpus": corpus, "queries": queries, "qrels": tmp_path / "test.tsv",
         "metadata": tmp_path / "metadata.csv"},
        tmp_path / "out",
    )
    docs = {d["id"]: d for d in read_jsonl(out.docs)}
    assert docs["aaa"] == {"id": "aaa", "title": "T1", "text": "A1", "publish_year": 2020,
                           "publish_date": 20200315, "source": ["Medline", "PMC"], "journal": "Lancet"}
    assert docs["bbb"]["publish_date"] == 20200101 and docs["bbb"]["journal"] is None
    assert docs["ccc"]["publish_year"] is None
    assert "zzz" not in docs
    qs = list(read_jsonl(out.queries))
    assert qs == [
        {"qid": "1-question", "text": "what is the origin of COVID-19", "group": "question"},
        {"qid": "1-keyword", "text": "coronavirus origin", "group": "keyword"},
    ]
    qrels = read_qrels(out.qrels)
    assert qrels["1-question"] == qrels["1-keyword"] == {"aaa": 2, "bbb": 0}
    assert out.dropped == {"docs": 1, "judgments": 2}  # zzz under both query forms
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** `harness/irbench/datasets/trec_covid.py`

```python
import csv
import re
from pathlib import Path
from typing import Any

from irbench.datasets.beir import read_jsonl_rows, read_qrels_tsv, write_normalized
from irbench.datasets.wands import NormalizedDataset

csv.field_size_limit(1 << 30)


def publish_fields(publish_time: str) -> tuple[int | None, int | None]:
    """CORD-19 `publish_time` (`2020-03-15`, `2020`, or empty) -> (year, YYYYMMDD).
    A year-only date sorts as January 1st."""
    m = re.match(r"\s*(\d{4})(?:-(\d{1,2})-(\d{1,2}))?", publish_time or "")
    if not m:
        return None, None
    year = int(m[1])
    month, day = (int(m[2]), int(m[3])) if m[2] else (1, 1)
    return year, year * 10000 + month * 100 + day


def _metadata(path: Path) -> dict[str, dict[str, Any]]:
    """First metadata row per cord_uid, in file order."""
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            uid = row["cord_uid"].strip()
            if uid in out:
                continue
            year, date = publish_fields(row.get("publish_time", ""))
            out[uid] = {
                "publish_year": year,
                "publish_date": date,
                "source": [s.strip() for s in (row.get("source_x") or "").split(";") if s.strip()],
                "journal": (row.get("journal") or "").strip() or None,
            }
    return out


EMPTY = {"publish_year": None, "publish_date": None, "source": [], "journal": None}


def normalize_trec_covid(raw: dict[str, Path], out_dir: Path) -> NormalizedDataset:
    meta = _metadata(raw["metadata"])
    docs = (
        {"id": r["_id"], "title": r.get("title") or "", "text": r.get("text") or "",
         **meta.get(r["_id"], EMPTY)}
        for r in read_jsonl_rows(raw["corpus"])
    )
    topic_qrels = read_qrels_tsv(raw["qrels"])
    queries, qrels = [], {}
    for r in read_jsonl_rows(raw["queries"]):
        topic = r["_id"]
        forms = {"question": r["text"], "keyword": (r.get("metadata") or {}).get("query")}
        for group, text in forms.items():
            if not text:
                continue
            qid = f"{topic}-{group}"
            queries.append({"qid": qid, "text": " ".join(text.split()), "group": group})
            qrels[qid] = dict(topic_qrels.get(topic, {}))
    return write_normalized(out_dir, docs, queries, qrels, ["title", "text"])
```

Register `"trec-covid": normalize_trec_covid`.

`datasets/trec-covid/dataset.yaml` (sha256 values from `hashes.json`, keys `trec-covid.zip:<member>` and `metadata.csv`):

```yaml
id: trec-covid
version: "beir-2021+cord19-2020-07-16"
license: CC BY-SA 4.0 (BEIR); CORD-19 papers carry their own licenses (metadata.csv `license`)
source: https://github.com/beir-cellar/beir and https://github.com/allenai/cord19
files:
  corpus: {url: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip, member: trec-covid/corpus.jsonl, sha256: <hashes.json>}
  queries: {url: ..., member: trec-covid/queries.jsonl, sha256: <hashes.json>}
  qrels: {url: ..., member: trec-covid/qrels/test.tsv, sha256: <hashes.json>}
  metadata:
    url: https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-07-16/metadata.csv
    sha256: <hashes.json>
```

- [ ] **Step 4: Run tests, expect PASS.**

- [ ] **Step 5: Commit** — `git commit -m "feat: TREC-COVID dataset joined with CORD-19 metadata"`

---

### Task 6: Scenarios, engine configs, CI and docs

**Files:**
- Create: `scenarios/knowledge-base/community-qa/scenario.yaml`, `scenarios/knowledge-base/covid-papers/scenario.yaml`, `scenarios/knowledge-base/web-questions/scenario.yaml`
- Create: `engines/{meilisearch,elasticsearch,qdrant,postgres,postgres-paradedb}/configs/knowledge-base/{community-qa,covid-papers,web-questions}.yaml` (15 files)
- Modify: `.github/workflows/ci.yml`; docs (`docs/` Mintlify pages and nav config), `docs/superpowers/specs/2026-10-09-irbench-design.md` ("Later" list)
- Test: `harness/tests/test_scenario.py` (load all repo scenarios), `uv run irbench lint`

**Interfaces:**
- Consumes: Tasks 1–5 (datasets prepared by `irbench prepare`).

- [ ] **Step 1: Failing test** (`test_scenario.py`)

```python
@pytest.mark.parametrize("scenario_id", [
    "knowledge-base/community-qa", "knowledge-base/covid-papers", "knowledge-base/web-questions",
])
def test_knowledge_base_scenarios_load(scenario_id):
    s = load_scenario(REPO_ROOT, scenario_id)
    assert s.synonyms is None and s.perturbations.kinds == ["typo", "prefix"]
    assert s.page_sizes == [20, 100] and s.top_k == 100
```

- [ ] **Step 2: Run, expect FAIL** (missing files).

- [ ] **Step 3: Write the scenarios.** Each file copies the WANDS scenario's comments style and these blocks verbatim from `scenarios/ecommerce/product-search/scenario.yaml`: `top_k`, `max_facet_values`, `facet_verify_limit`, `request_timeout_s`, `max_error_rate`, `perf`. Specific parts:

`community-qa`:
```yaml
id: knowledge-base/community-qa
dataset: cqadupstack
id_field: id
index_fields:
  searchable: [title, text]
  filterable: [forum, tags]
  sortable: []
  facets: [forum]
  numeric: []
modes: [keyword, semantic, hybrid, filtered, faceted]
structured_bases: [keyword, semantic]
embedding_fields: [title, text]
filter_templates:
  - {field: forum, op: eq}
  - {field: tags, op: eq}
sort_templates: []
index_timeout_s: 1800
perturbations:
  seed: 20261014
  kinds: [typo, prefix]
```

`covid-papers`:
```yaml
id: knowledge-base/covid-papers
dataset: trec-covid
id_field: id
index_fields:
  searchable: [title, text]
  filterable: [source, publish_year]
  sortable: [publish_date]
  facets: [source]
  numeric: [publish_year, publish_date]
modes: [keyword, semantic, hybrid, filtered, sorted, faceted]
structured_bases: [keyword, semantic]
embedding_fields: [title, text]
filter_templates:
  - {field: source, op: eq}
  - {field: publish_year, op: gte, values: [2020]}
sort_templates:
  - {field: publish_date, direction: desc}
index_timeout_s: 3600
perturbations:
  seed: 20261014
  kinds: [typo, prefix]
```

`web-questions`:
```yaml
id: knowledge-base/web-questions
dataset: lotte-lifestyle
id_field: id
index_fields:
  searchable: [text]
  filterable: []
  sortable: []
  facets: []
  numeric: []
modes: [keyword, semantic, hybrid]
structured_bases: []
embedding_fields: [text]
filter_templates: []
sort_templates: []
index_timeout_s: 1800
perturbations:
  seed: 20261014
  kinds: [typo, prefix]
```

Run `uv run pytest harness/tests/test_scenario.py -q` → PASS.

- [ ] **Step 4: Prepare the three scenarios** (real downloads, about 1 GB of range reads and files; run in the background if slow):

```bash
uv run irbench prepare --scenario knowledge-base/web-questions
uv run irbench prepare --scenario knowledge-base/community-qa
uv run irbench prepare --scenario knowledge-base/covid-papers
```

Record the printed counts (docs, queries, tasks, skipped, dropped) in the task report. Expected orders of magnitude: web-questions about 119k docs and 661 queries; community-qa about 89k docs and 1,746 queries; covid-papers 171,331 docs and 100 queries.

- [ ] **Step 5: Engine configs.** For each engine, copy the WANDS config `engines/<engine>/configs/ecommerce/product-search.yaml` to the three new paths and adapt:

- header comment: `# <Engine> on knowledge-base/<scenario>.`; drop mentions of `synonyms.txt` (these scenarios have none).
- **Meilisearch:**
  - `covid-papers` (has `sorted`): keep the WANDS `rankingRules` override with `sort` first and its justification, reworded to name `publish_date`.
  - `community-qa` and `web-questions` (no `sorted` mode): `default: {index: {}, search: {}, unsupported: {}}` — no ranking-rule change.
  - All three get the `tuned` block from WANDS (`matchingStrategy: frequency` on keyword and hybrid) with this justification: "Carries the WANDS finding (keyword nDCG@10 0.481 -> 0.508 with `frequency`) over to question-style queries, where `last` drops the final words of a question first. A documented, supported search parameter; no ranking-rule change. The tuned run measures whether the finding holds here."
- **Elasticsearch:** `default: {index: {}, search: {}, unsupported: {}}`.
- **Qdrant:** as WANDS, minus the `synonyms` prose (no synonyms to load); keep `unsupported.faceted` with the same reason in `community-qa` and `covid-papers` (both have `faceted`); `web-questions` has no faceted mode, so `unsupported: {}`.
- **Postgres and ParadeDB:** keep the `hnsw.ef_search`, `hnsw.iterative_scan` (and for ParadeDB `plan_cache_mode`) settings and justifications verbatim; drop ParadeDB's `synonyms` prose.

Then run `uv run irbench lint` → expect `0 findings` over 20 configs. Fix any finding in the config, not in the lint.

- [ ] **Step 6: CI.** In `.github/workflows/ci.yml`, after the WANDS prepare step, add one `uv run irbench prepare --scenario <id>` per new scenario (the dataset cache key already hashes every `datasets/**/dataset.yaml`).

- [ ] **Step 7: Docs.** Add one Mintlify page per scenario under the existing scenarios/datasets section of `docs/` (find it with `grep -rn "product-search" docs --include=*.mdx -l`; mirror the WANDS page's structure and add each page to the nav config file the WANDS page is listed in). Each page states: source and license; documents, queries, judgments (counts from Step 4); fields and which modes run; what was dropped and why (§3.4); perturbations (typo, prefix; no synonyms); for covid-papers, the two query forms and that 100 queries give wide confidence intervals; for web-questions, that queries are real Google questions and judgments are the accepted answers (sparse). Update `docs/superpowers/specs/2026-10-09-irbench-design.md` "Later → Datasets/scenarios" to mark knowledge-base as done with a pointer to the new spec.

- [ ] **Step 8: Run the full suite and lint; commit** — `git commit -m "feat: knowledge-base scenarios, engine configs, CI prepare and docs"`

---

### Task 7: Runs, report and Linear notes (controller)

Run by the controller, not a subagent (long-running, outward Linear writes already authorized by the user for this project).

- [ ] **Step 1: Embed** each scenario with bge-small (`uv run irbench embed --scenario <id> --model bge-small-en-v1.5`), covid-papers last (largest).
- [ ] **Step 2: Run** the 5 engines, default track, on each scenario (full pipeline, `local` profile), then Meilisearch `--track tuned` on each. Use a script modeled on the scratchpad `runall.sh`. Record any failed run as failed.
- [ ] **Step 3: WANDS regression:** re-score the 6 WANDS runs and compare `relevance` with the saved `scores.json` copies; they must be identical.
- [ ] **Step 4: Report:** `uv run irbench report --results .irbench/runs --out .irbench/site`; open it; check the three new scenario pages (matrix, N/A cells, group tables, perturbation columns without synonym).
- [ ] **Step 5: Linear:** one document per scenario in project "Information Retrieval Benches" (relevance by mode and query variant, groups, latency and throughput at 20 and 100), and update the project description to list the scenarios.
