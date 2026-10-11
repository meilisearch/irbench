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
from irbench.runtime.docker import DockerRuntime, Runtime
from irbench.runtime.hardware import HardwareProfile, load_hardware
from irbench.stages.context import ClientFactory, RunContext, default_client
from irbench.stages.embed import embed_scenario
from irbench.stages.index import index_stage
from irbench.stages.prepare import prepare
from irbench.stages.relevance import relevance_stage
from irbench.stages.render import render_stage
from irbench.stages.score_run import score_stage

RuntimeFactory = Callable[[Path, str, HardwareProfile], Runtime]
GENERATED_KEYS = ("irbench_version", "git_sha", "created_at")


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
            return asdict(relevance_stage(ctx))
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
            log.pop(stage, None)
            if not record(stage, lambda s=stage: execute(s)):
                status = "failed"
                break
            started_up = started_up or stage == "up"
        clean = status == "succeeded"
    finally:
        if "down" in ordered or (started_up and not clean):
            if not record("down", runtime.down):
                status = "failed"

    manifest["status"] = status
    _describe_inputs(ctx, manifest)
    write_manifest(manifest_path, manifest)
    return manifest


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
