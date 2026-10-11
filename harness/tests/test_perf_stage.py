import pytest

from conftest import REPO_ROOT
from fake_engine import FAKE_SPEC, SAMPLE_PERF, fake_loadgen, fake_root
from irbench.io import read_json
from irbench.runtime.hardware import load_hardware
from irbench.scenario import load_scenario
from irbench.stages.context import RunContext
from irbench.stages.perf import LOADGEN_PATH, loadgen_command, perf_stage, summarize_perf
from irbench.stages.render import render_stage

LOCAL = load_hardware(REPO_ROOT, "local")
REFERENCE = load_hardware(REPO_ROOT, "aws-c7i-4xlarge")


def test_scenario_declares_the_spec_perf_protocol():
    p = load_scenario(REPO_ROOT, "ecommerce/product-search").perf
    assert p.concurrency == [1, 2, 4, 8, 16, 32, 64]
    assert (p.p99_limit_ms, p.max_error_rate) == (100.0, 0.001)
    assert p.throughput_groups == ["keyword/keyword", "semantic/semantic", "hybrid/hybrid"]


def test_command_replays_the_rendered_dir_with_scenario_settings(tmp_path, wands_raw):
    ctx = RunContext(fake_root(tmp_path, wands_raw), FAKE_SPEC)
    cmd = loadgen_command(ctx, LOCAL, taskset=None)
    assert cmd[0] == str(ctx.root / LOADGEN_PATH)
    args = dict(zip(cmd[1::2], cmd[2::2]))
    assert args["--requests"] == str(ctx.rendered_dir)
    assert args["--base-url"] == "http://localhost:9999"
    assert args["--out"] == str(ctx.run_dir / "perf.json")
    assert args["--concurrency"] == "1,2,4,8,16,32,64"
    assert args["--groups"] == "keyword/keyword,semantic/semantic,hybrid/hybrid"
    assert (args["--p99-limit-ms"], args["--max-error-rate"], args["--timeout-s"]) == (
        "100.0",
        "0.001",
        "10.0",
    )


def test_reference_profile_pins_the_loadgen_cpuset(tmp_path, wands_raw):
    ctx = RunContext(fake_root(tmp_path, wands_raw), FAKE_SPEC)
    cmd = loadgen_command(ctx, REFERENCE, taskset="/usr/bin/taskset")
    assert cmd[:3] == ["/usr/bin/taskset", "-c", "8-11"]


def test_pinned_profile_without_taskset_fails(tmp_path, wands_raw):
    ctx = RunContext(fake_root(tmp_path, wands_raw), FAKE_SPEC)
    with pytest.raises(RuntimeError, match="taskset"):
        loadgen_command(ctx, REFERENCE, taskset=None)


def test_perf_stage_runs_the_binary_and_summarizes(tmp_path, wands_raw):
    ctx = RunContext(fake_root(tmp_path, wands_raw), FAKE_SPEC)
    render_stage(ctx)
    calls = []
    summary = perf_stage(ctx, LOCAL, runner=fake_loadgen(calls=calls))
    assert len(calls) == 1
    assert read_json(ctx.run_dir / "perf.json") == SAMPLE_PERF
    assert summary["keyword/keyword"] == {
        "p50_ms": 1.5,
        "p95_ms": 3.0,
        "p99_ms": 4.0,
        "mean_ms": 2.0,
        "errors": 0,
        "max_qps": 812.5,
        "max_qps_concurrency": 16,
    }
    assert summary["all"]["max_qps"] is None


def test_perf_needs_rendered_requests_and_the_binary(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    ctx = RunContext(root, FAKE_SPEC)
    with pytest.raises(FileNotFoundError, match="render"):
        perf_stage(ctx, LOCAL, runner=fake_loadgen())
    render_stage(ctx)
    (root / LOADGEN_PATH).unlink()
    with pytest.raises(FileNotFoundError, match="cargo build --release"):
        perf_stage(ctx, LOCAL, runner=fake_loadgen())


def test_summary_of_a_group_without_throughput():
    report = {**SAMPLE_PERF, "throughput": {}}
    assert summarize_perf(report)["keyword/keyword"]["max_qps"] is None
