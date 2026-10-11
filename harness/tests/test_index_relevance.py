import pytest

from fake_engine import (
    FAKE_SPEC,
    FakeRuntime,
    FakeServer,
    fake_root,
    perfect_answers,
    set_scenario,
)
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
