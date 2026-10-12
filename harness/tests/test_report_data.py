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
        ("bge-small-en-v1.5", "meilisearch"),
        ("bge-small-en-v1.5", "qdrant"),
        ("text-embedding-3-small", "meilisearch"),
    ]
    assert s.models == ["bge-small-en-v1.5", "text-embedding-3-small"]
    assert s.slug == "ecommerce__product-search__aws-c7i-4xlarge"


def test_profiles_are_never_mixed(tmp_path):
    write_bundle(tmp_path, "meilisearch", hardware="aws-c7i-4xlarge")
    write_bundle(tmp_path, "meilisearch", hardware="local")
    scenarios, _ = load_results(tmp_path)
    assert [(s.hardware, len(s.runs)) for s in scenarios] == [
        ("aws-c7i-4xlarge", 1),
        ("local", 1),
    ]


def test_lookups(tmp_path):
    write_bundle(
        tmp_path,
        "qdrant",
        track="tuned",
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
    groups = [
        "faceted/keyword",
        "hybrid/hybrid",
        "filtered/semantic",
        "keyword/keyword",
        "filtered/keyword",
        "semantic/semantic",
    ]
    assert sorted(groups, key=group_key) == [
        "keyword/keyword",
        "semantic/semantic",
        "hybrid/hybrid",
        "filtered/keyword",
        "filtered/semantic",
        "faceted/keyword",
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
