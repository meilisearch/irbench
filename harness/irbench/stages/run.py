import shutil
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from irbench.embed.base import Embedder
from irbench.io import read_json
from irbench.manifest import write_manifest
from irbench.runs import STAGES, RunSpec
from irbench.runtime.docker import DockerRuntime, Runner, Runtime, run_command
from irbench.runtime.hardware import HardwareProfile, load_hardware
from irbench.stages.context import ClientFactory, RunContext, default_client
from irbench.stages.embed import embed_scenario
from irbench.stages.index import index_stage
from irbench.stages.perf import perf_stage
from irbench.stages.prepare import prepare
from irbench.stages.relevance import relevance_stage
from irbench.stages.render import render_stage
from irbench.stages.score_run import score_stage

RuntimeFactory = Callable[[Path, str, HardwareProfile], Runtime]
GENERATED_KEYS = ("irbench_version", "git_sha", "created_at")
# Files each stage writes in the run dir; re-running a stage invalidates its own
# outputs and those of every later stage.
STAGE_OUTPUTS = {
    "index": ("index.json",),
    "render": ("rendered", "na.json"),
    "relevance": ("run.trec", "checks.jsonl", "errors.jsonl"),
    "perf": ("perf.json",),
    "score": ("scores.json",),
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def run_pipeline(
    root: Path,
    spec: RunSpec,
    stages: list[str],
    *,
    runtime_factory: RuntimeFactory | None = None,
    client_factory: ClientFactory | None = None,
    prepare_client: httpx.Client | None = None,
    embedder: Embedder | None = None,
    loadgen_runner: Runner | None = None,
) -> dict[str, Any]:
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; known: {', '.join(STAGES)}")
    ordered = [s for s in STAGES if s in stages]

    ctx = RunContext(root, spec, client_factory or default_client)
    profile = load_hardware(root, spec.hardware)
    runtime = (runtime_factory or DockerRuntime)(root, spec.engine, profile)

    manifest_path = ctx.run_dir / "manifest.json"
    manifest: dict[str, Any] = read_json(manifest_path) if manifest_path.exists() else {}
    for key in GENERATED_KEYS:
        manifest.pop(key, None)
    manifest.update(
        run_id=spec.run_id,
        scenario=spec.scenario,
        engine=spec.engine,
        model=spec.model,
        track=spec.track,
        hardware=profile.model_dump(),
    )
    log: dict[str, Any] = manifest.setdefault("stages", {})
    _invalidate_from(ctx.run_dir, log, ordered[0] if ordered else None)

    def execute(stage: str) -> Any:
        if stage == "prepare":
            return {"tasks": prepare(root, spec.scenario, client=prepare_client).n_tasks}
        if stage == "embed":
            result = embed_scenario(root, spec.scenario, spec.model, embedder)
            return {"computed": result.n_computed}
        if stage == "up":
            runtime.up()
            manifest["engine_image"] = runtime.image_info()
            return None
        if stage == "index":
            return asdict(index_stage(ctx, runtime))
        if stage == "render":
            return asdict(render_stage(ctx))
        if stage == "relevance":
            result = relevance_stage(ctx)
            rate = result.n_errors / result.n_tasks if result.n_tasks else 0.0
            limit = ctx.scenario.max_error_rate
            if rate > limit:
                raise RuntimeError(
                    f"error rate {rate:.2%} ({result.n_errors}/{result.n_tasks}) exceeds "
                    f"{limit:.2%}: see errors.jsonl"
                )
            return asdict(result)
        if stage == "perf":
            return perf_stage(ctx, profile, runner=loadgen_runner or run_command)
        if stage == "score":
            report = score_stage(ctx)
            return {"errors": report["errors"], "not_applicable": len(report["not_applicable"])}
        raise AssertionError(stage)

    def record(stage: str, fn: Callable[[], Any]) -> bool:
        started, t0 = _now(), time.perf_counter()
        entry: dict[str, Any] = {"started_at": started}
        try:
            result = fn()
        except Exception as e:  # spec §9: a failed stage is recorded, never a crash
            entry.update(status="failed", error=f"{type(e).__name__}: {e}")
            ok = False
        else:
            entry["status"] = "succeeded"
            if result is not None:
                entry["result"] = result
            ok = True
        entry["seconds"] = round(time.perf_counter() - t0, 3)
        log[stage] = entry
        return ok

    status, started_up, clean = "succeeded", False, False
    try:
        for stage in ordered:
            if stage == "down":
                continue
            # A failing `up` may already have created the container: always tear down.
            started_up = started_up or stage == "up"
            if not record(stage, lambda s=stage: execute(s)):
                status = "failed"
                break
        clean = status == "succeeded"
    finally:
        if "down" in ordered or (started_up and not clean):
            if not record("down", runtime.down):
                status = "failed"

    manifest["status"] = status
    _describe_inputs(ctx, manifest)
    write_manifest(manifest_path, manifest)
    return manifest


def _invalidate_from(run_dir: Path, log: dict[str, Any], first: str | None) -> None:
    if first is None:
        return
    for stage in STAGES[STAGES.index(first) :]:
        log.pop(stage, None)
        for name in STAGE_OUTPUTS.get(stage, ()):
            path = run_dir / name
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)


def _describe_inputs(ctx: RunContext, manifest: dict[str, Any]) -> None:
    try:
        manifest["config"] = {
            "path": str(ctx.config.path.relative_to(ctx.root)),
            "hash": ctx.config.hash(ctx.spec.track),
            "diff_vs_default": ctx.config.diff(ctx.spec.track),
        }
    except (FileNotFoundError, ValueError) as e:
        manifest["config"] = {"error": str(e)}
    prepared = ctx.ws.scenario_dir(ctx.spec.scenario) / "prepare.manifest.json"
    if prepared.exists():
        info = read_json(prepared)
        manifest["dataset"] = info["dataset"]
        manifest["seed"] = info["seed"]
