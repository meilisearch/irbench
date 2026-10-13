import shutil
from pathlib import Path
from typing import Any

from irbench.io import read_json
from irbench.perf_files import LEGACY_NAME, perf_path, perf_report_paths
from irbench.runtime.docker import Runner, run_command
from irbench.runtime.hardware import HardwareProfile
from irbench.stages.context import RunContext

LOADGEN_PATH = Path("loadgen/target/release/irbench-loadgen")
BUILD_HINT = "cargo build --release --manifest-path loadgen/Cargo.toml"


def loadgen_command(
    ctx: RunContext, profile: HardwareProfile, taskset: str | None, page_size: int
) -> list[str]:
    p = ctx.scenario.perf
    cmd = [
        str(ctx.root / LOADGEN_PATH),
        "--requests",
        str(ctx.rendered_dir_for(page_size)),
        "--base-url",
        ctx.base_url,
        "--out",
        str(perf_path(ctx.run_dir, page_size)),
        "--concurrency",
        ",".join(str(c) for c in p.concurrency),
        "--groups",
        ",".join(p.throughput_groups),
        "--step-seconds",
        str(p.step_seconds),
        "--p99-limit-ms",
        str(p.p99_limit_ms),
        "--max-error-rate",
        str(p.max_error_rate),
        "--timeout-s",
        str(ctx.scenario.request_timeout_s),
        "--warmup-concurrency",
        str(p.warmup_concurrency),
    ]
    cpuset = profile.loadgen.cpuset
    if cpuset:
        if taskset is None:
            raise RuntimeError(
                f"hardware profile {profile.id!r} pins the loadgen to CPUs {cpuset}, but taskset "
                "is not available (Linux only); refusing to measure unpinned"
            )
        cmd = [taskset, "-c", cpuset, *cmd]
    return cmd


def summarize_perf(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    # The loadgen's "all" mixes whichever modes the engine supports, so it is not
    # comparable across engines and stays out of the summary.
    summary = {}
    for group, lat in report["latency"].items():
        if group == "all":
            continue
        t = report["throughput"].get(group, {})
        summary[group] = {
            "p50_ms": lat["p50_ms"],
            "p95_ms": lat["p95_ms"],
            "p99_ms": lat["p99_ms"],
            "mean_ms": lat["mean_ms"],
            "errors": lat["errors"],
            "max_qps": t.get("max_qps"),
            "max_qps_concurrency": t.get("max_qps_concurrency"),
        }
    return summary


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
    # Build from this commit (a no-op when fresh) so a stale binary never measures.
    manifest = str(ctx.root / "loadgen" / "Cargo.toml")
    runner(["cargo", "build", "--release", "--quiet", "--manifest-path", manifest])
    binary = ctx.root / LOADGEN_PATH
    if not binary.exists():
        raise FileNotFoundError(
            f"{binary} missing: loadgen build produced no binary ({BUILD_HINT})"
        )
    # A pre-page-size report would sit next to the new ones and confuse readers; drop it.
    (ctx.run_dir / LEGACY_NAME).unlink(missing_ok=True)
    taskset = shutil.which("taskset")
    summary = {}
    try:
        for n in sizes:
            runner(loadgen_command(ctx, profile, taskset, n))
            summary[str(n)] = summarize_perf(read_json(perf_path(ctx.run_dir, n)))
    except BaseException:
        # All or nothing: a run is never scored or published with one size missing.
        for path in perf_report_paths(ctx.run_dir).values():
            path.unlink(missing_ok=True)
        raise
    return summary
