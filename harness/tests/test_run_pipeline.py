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
        root,
        FAKE_SPEC,
        list(STAGES),
        runtime_factory=lambda *a: runtime,
        client_factory=server.client_factory,
        prepare_client=fixture_client(wands_raw),
        embedder=FakeEmbedder(),
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
        root,
        FAKE_SPEC,
        ["up", "index", "render"],
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
        run_pipeline(
            fake_root(tmp_path, wands_raw),
            FAKE_SPEC,
            ["up", "nope"],
            runtime_factory=lambda *a: runtime,
        )
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
    args = ["run", "--scenario", SCENARIO, "--engine", "fake", "--model", "fake"]
    args += ["--stages", "up,index,render,relevance,score,down", "--root", str(root)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "keyword/keyword: nDCG@10=1.0000" in result.stdout
    assert "hybrid/hybrid: N/A (fake engine has no hybrid)" in result.stdout


def test_cli_run_rejects_unknown_stage(tmp_path, wands_raw):
    args = ["run", "--scenario", SCENARIO, "--engine", "fake", "--model", "fake"]
    args += ["--stages", "bogus", "--root", str(fake_root(tmp_path, wands_raw))]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert "bogus" in result.output
