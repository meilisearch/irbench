import pytest

from conftest import REPO_ROOT
from fake_engine import FAKE_SPEC, SAMPLE_PERF, fake_loadgen, fake_root
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
    cmd = loadgen_command(ctx, LOCAL, taskset=None, page_size=20)
    assert cmd[0] == str(ctx.root / LOADGEN_PATH)
    args = dict(zip(cmd[1::2], cmd[2::2]))
    assert args["--requests"] == str(ctx.rendered_dir_for(20))
    assert args["--base-url"] == "http://localhost:9999"
    assert args["--out"] == str(ctx.run_dir / "perf-p20.json")
    assert args["--concurrency"] == "1,2,4,8,16,32,64"
    assert args["--groups"] == "keyword/keyword,semantic/semantic,hybrid/hybrid"
    assert (args["--p99-limit-ms"], args["--max-error-rate"], args["--timeout-s"]) == (
        "100.0",
        "0.001",
        "10.0",
    )


def test_reference_profile_pins_the_loadgen_cpuset(tmp_path, wands_raw):
    ctx = RunContext(fake_root(tmp_path, wands_raw), FAKE_SPEC)
    cmd = loadgen_command(ctx, REFERENCE, taskset="/usr/bin/taskset", page_size=20)
    assert cmd[:3] == ["/usr/bin/taskset", "-c", "8-11"]


def test_pinned_profile_without_taskset_fails(tmp_path, wands_raw):
    ctx = RunContext(fake_root(tmp_path, wands_raw), FAKE_SPEC)
    with pytest.raises(RuntimeError, match="taskset"):
        loadgen_command(ctx, REFERENCE, taskset=None, page_size=20)


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


def test_perf_needs_rendered_requests_and_the_binary(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    ctx = RunContext(root, FAKE_SPEC)
    with pytest.raises(FileNotFoundError, match="render"):
        perf_stage(ctx, LOCAL, runner=fake_loadgen())
    render_stage(ctx)
    (root / LOADGEN_PATH).unlink()
    with pytest.raises(FileNotFoundError, match="loadgen build produced no binary"):
        perf_stage(ctx, LOCAL, runner=fake_loadgen())


def test_summary_of_a_group_without_throughput():
    report = {**SAMPLE_PERF, "throughput": {}}
    assert summarize_perf(report)["keyword/keyword"]["max_qps"] is None
