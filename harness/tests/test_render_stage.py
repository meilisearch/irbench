import numpy as np
import pytest

from fake_engine import FAKE_SPEC, fake_root, set_scenario
from irbench.embed.vectors import read_vectors, write_vectors
from irbench.engines.base import Request
from irbench.io import read_json
from irbench.paths import Workspace
from irbench.runs import RunSpec
from irbench.stages.context import RunContext
from irbench.stages.render import render_stage


def test_run_id_names_every_axis():
    spec = RunSpec(
        scenario="ecommerce/product-search",
        engine="meilisearch",
        model="org/bge",
        track="tuned",
        hardware="aws-c7i-4xlarge",
    )
    assert spec.run_id == (
        "ecommerce__product-search__meilisearch__org__bge__tuned__aws-c7i-4xlarge"
    )


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
