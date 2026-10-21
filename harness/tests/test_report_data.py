import json
import shutil

import pytest

from irbench.report.data import group_key, load_results
from report_fixtures import perf, relevance, write_bundle


def _edit_json(path, fn):
    path.write_text(json.dumps(fn(json.loads(path.read_text()))))


def test_runs_load_perf_per_page_size(tmp_path):
    write_bundle(
        tmp_path,
        "meilisearch",
        perf_reports={
            20: perf({"keyword/keyword": (90.0, 4)}),
            100: perf({"keyword/keyword": (50.0, 4)}),
        },
    )
    run = load_results(tmp_path)[0][0].runs[0]
    assert list(run.perf) == [20, 100]


def test_legacy_perf_json_loads_as_page_size_100(tmp_path):
    write_bundle(tmp_path, "meilisearch", perf_report=perf({"keyword/keyword": (50.0, 4)}))
    run = load_results(tmp_path)[0][0].runs[0]
    assert list(run.perf) == [100]


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
    assert run.label == "qdrant"  # the engine's only track: no track in the label
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


@pytest.mark.parametrize(
    "corrupt_fn,healthy_count",
    [
        (lambda d: (d / "scores.json").write_text("{}"), 1),
        (lambda d: (d / "scores.json").write_text("[]"), 1),
        (
            lambda d: (d / "scores.json").write_text('{"relevance": [], "not_applicable": {}}'),
            1,
        ),
        (
            lambda d: (d / "perf.json").write_text('{"limits": {}, "latency": {}}'),
            1,
        ),
        (
            lambda d: (d / "perf-p20.json").write_text('{"limits": {}, "latency": {}}'),
            1,
        ),
        (lambda d: (d / "index.json").write_text("[]"), 1),
    ],
    ids=[
        "scores_empty_dict",
        "scores_empty_list",
        "scores_missing_constraint_violations",
        "perf_missing_throughput",
        "perf_p20_missing_throughput",
        "index_wrong_type",
    ],
)
def test_malformed_files_are_unreadable(tmp_path, corrupt_fn, healthy_count):
    write_bundle(tmp_path, "meilisearch")
    corrupt = write_bundle(tmp_path, "qdrant")
    corrupt_fn(corrupt)
    scenarios, unreadable = load_results(tmp_path)
    assert sum(len(s.runs) for s in scenarios) == healthy_count
    assert len(unreadable) == 1
    assert unreadable[0].path == corrupt.name
    assert (
        "scores.json" in unreadable[0].error
        or "perf.json" in unreadable[0].error
        or "perf-p20.json" in unreadable[0].error
        or "index.json" in unreadable[0].error
    )


@pytest.mark.parametrize(
    "corrupt_fn",
    [
        lambda d: _edit_json(d / "manifest.json", lambda m: {**m, "stages": ["prepare"]}),
        lambda d: _edit_json(d / "manifest.json", lambda m: {**m, "config": None}),
        lambda d: _edit_json(
            d / "manifest.json",
            lambda m: {**m, "config": {**m["config"], "credits": [{"url": "https://x"}]}},
        ),
        lambda d: _edit_json(
            d / "manifest.json",
            lambda m: {
                **m,
                "config": {**m["config"], "credits": [{"name": "Jane", "url": 123}]},
            },
        ),
        lambda d: _edit_json(
            d / "manifest.json",
            lambda m: {**m, "config": {**m["config"], "synonyms": ["not a string"]}},
        ),
        lambda d: _edit_json(
            d / "scores.json",
            lambda s: {
                **s,
                "relevance": [
                    {k: v for k, v in s["relevance"][0].items() if k != "mode"},
                    *s["relevance"][1:],
                ],
            },
        ),
        lambda d: _edit_json(d / "index.json", lambda i: {**i, "seconds": "42.5"}),
        lambda d: _edit_json(
            d / "scores.json",
            lambda s: {
                **s,
                "not_applicable": {**s["not_applicable"], "renamedmode:keyword": "old bundle"},
            },
        ),
    ],
    ids=[
        "manifest_stages_is_a_list",
        "manifest_config_is_null",
        "credit_without_name",
        "credit_url_not_a_string",
        "synonyms_not_a_string",
        "relevance_item_missing_mode",
        "index_seconds_is_a_string",
        "not_applicable_key_has_unknown_mode",
    ],
)
def test_malformed_shapes_are_unreadable_not_fatal(tmp_path, corrupt_fn):
    write_bundle(tmp_path, "meilisearch")
    corrupt = write_bundle(tmp_path, "qdrant")
    corrupt_fn(corrupt)
    scenarios, unreadable = load_results(tmp_path)
    assert sum(len(s.runs) for s in scenarios) == 1
    assert len(unreadable) == 1
    assert unreadable[0].path == corrupt.name


def test_run_id_path_traversal_is_unreadable(tmp_path):
    d = write_bundle(tmp_path, "meilisearch")
    _edit_json(d / "manifest.json", lambda m: {**m, "run_id": "../../escaped"})
    scenarios, unreadable = load_results(tmp_path)
    assert sum(len(s.runs) for s in scenarios) == 0
    assert len(unreadable) == 1 and unreadable[0].path == d.name


def test_run_id_not_matching_directory_is_unreadable(tmp_path):
    d = write_bundle(tmp_path, "meilisearch")
    _edit_json(d / "manifest.json", lambda m: {**m, "run_id": "some-other-id"})
    scenarios, unreadable = load_results(tmp_path)
    assert sum(len(s.runs) for s in scenarios) == 0
    assert len(unreadable) == 1 and unreadable[0].path == d.name


def test_hardware_id_path_traversal_is_unreadable(tmp_path):
    d = write_bundle(tmp_path, "meilisearch")
    _edit_json(d / "manifest.json", lambda m: {**m, "hardware": {"id": "../escaped"}})
    scenarios, unreadable = load_results(tmp_path)
    assert sum(len(s.runs) for s in scenarios) == 0
    assert len(unreadable) == 1 and unreadable[0].path == d.name


def test_tmp_and_old_directories_are_ignored(tmp_path):
    d = write_bundle(tmp_path, "meilisearch")
    shutil.copytree(d, tmp_path / f"{d.name}.tmp")
    shutil.copytree(d, tmp_path / f"{d.name}.old")
    scenarios, unreadable = load_results(tmp_path)
    assert sum(len(s.runs) for s in scenarios) == 1
    assert unreadable == []


def _grouped_run(tmp_path, extra=None):
    rows = relevance({"keyword/keyword": 0.5})
    # A grouped row with the same mode/base/perturbation, placed FIRST to catch order bugs.
    rows.insert(
        0, {**rows[0], "group": "keyword", "metrics": {**rows[0]["metrics"], "ndcg@10": 0.99}}
    )
    bundle = write_bundle(tmp_path, "meilisearch", groups={"keyword/keyword": 0.5})
    _edit_json(bundle / "scores.json", lambda d: {**d, "relevance": rows + (extra or [])})
    return load_results(tmp_path)[0][0].runs[0]


def test_grouped_rows_never_replace_the_overall_score(tmp_path):
    run = _grouped_run(tmp_path)
    assert run.ndcg("keyword/keyword") == 0.5
    assert run.group_ndcg("keyword/keyword", "keyword") == 0.99
    assert run.group_ndcg("keyword/keyword", "other") is None
    assert run.query_groups == ["keyword"]
    assert run.groups == ["keyword/keyword"]


def test_relevance_group_must_be_a_string(tmp_path):
    bundle = write_bundle(tmp_path, "meilisearch", groups={"keyword/keyword": 0.5})

    def bad(d):
        d["relevance"][0]["group"] = 3
        return d

    _edit_json(bundle / "scores.json", bad)
    _, unreadable = load_results(tmp_path)
    assert unreadable and "group" in str(unreadable[0])


def test_label_names_the_track_when_an_engine_has_two(tmp_path):
    write_bundle(tmp_path, "qdrant", track="tuned")
    write_bundle(tmp_path, "qdrant")
    labels = sorted(r.label for r in load_results(tmp_path)[0][0].runs)
    assert labels == ["qdrant", "qdrant (tuned)"]
