import json

import pytest
from typer.testing import CliRunner

from irbench.cli import app
from irbench.publish import BUNDLE_FILES, publish_run

RUN_ID = "ecommerce__product-search__meilisearch__bge-small-en-v1.5__default__aws-c7i-4xlarge"


def make_run(
    root,
    run_id=RUN_ID,
    profile="aws-c7i-4xlarge",
    status="succeeded",
    files=("scores.json", "perf.json", "index.json"),
):
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


def test_malformed_manifest_missing_hardware_raises_valueerror(tmp_path):
    d = tmp_path / ".irbench" / "runs" / RUN_ID
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"run_id": RUN_ID, "status": "succeeded"}))
    with pytest.raises(ValueError, match="hardware.id"):
        publish_run(tmp_path, RUN_ID)
    assert not (tmp_path / "results").exists()


def test_malformed_manifest_missing_hardware_id_raises_valueerror(tmp_path):
    d = tmp_path / ".irbench" / "runs" / RUN_ID
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"run_id": RUN_ID, "status": "succeeded", "hardware": {}})
    )
    with pytest.raises(ValueError, match="hardware.id"):
        publish_run(tmp_path, RUN_ID)
    assert not (tmp_path / "results").exists()


def test_cli_continues_after_malformed_manifest(tmp_path):
    bad_id = RUN_ID.replace("aws-c7i-4xlarge", "bad")
    d = tmp_path / ".irbench" / "runs" / bad_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"run_id": bad_id, "status": "succeeded"}))
    make_run(tmp_path)
    result = CliRunner().invoke(app, ["publish", bad_id, RUN_ID, "--root", str(tmp_path)])
    assert result.exit_code == 1
    assert "hardware.id" in result.output
    assert f"results/{RUN_ID}" in result.output


def test_republishing_cleans_up_tmp_and_old_files(tmp_path):
    make_run(tmp_path)
    publish_run(tmp_path, RUN_ID)
    make_run(tmp_path, files=("scores.json",))
    (tmp_path / ".irbench/runs" / RUN_ID / "perf.json").unlink()
    (tmp_path / ".irbench/runs" / RUN_ID / "index.json").unlink()
    dest = publish_run(tmp_path, RUN_ID)
    # Verify no .tmp or .old sibling directories remain
    results_dir = tmp_path / "results"
    siblings = [p.name for p in results_dir.iterdir()]
    assert RUN_ID in siblings
    assert not any(p.endswith(".tmp") or p.endswith(".old") for p in siblings)
    assert sorted(p.name for p in dest.iterdir()) == ["manifest.json", "scores.json"]
