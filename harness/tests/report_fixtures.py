"""Realistic result bundles for report tests (shapes copied from real Plan 3 runs)."""

import json
from pathlib import Path

PERTURBATIONS = ("all", "original", "typo", "prefix", "synonym")


def relevance(groups: dict[str, float]) -> list[dict]:
    """{"keyword/keyword": 0.5} -> score rows for every perturbation (typo -0.1, prefix -0.2)."""
    rows = []
    for group, ndcg in groups.items():
        mode, base = group.split("/")
        drop = {"all": 0.0, "original": 0.05, "typo": -0.1, "prefix": -0.2, "synonym": -0.05}
        for p in PERTURBATIONS:
            value = round(ndcg + drop[p], 4)
            rows.append(
                {
                    "mode": mode,
                    "base": base,
                    "perturbation": p,
                    "n": 100,
                    "metrics": {
                        "ndcg@10": value,
                        "recall@10": value / 2,
                        "recall@100": value,
                        "mrr@10": value,
                        "success@1": value / 3,
                    },
                }
            )
    return rows


def steps(qps_per_client: float, valid_until: int) -> list[dict]:
    out = []
    for c in (1, 2, 4, 8):
        valid = c <= valid_until
        out.append(
            {
                "concurrency": c,
                "seconds": 20.0,
                "requests": int(qps_per_client * c * 20),
                "errors": 0,
                "qps": qps_per_client * c,
                "error_rate": 0.0,
                "p50_ms": 2.0,
                "p95_ms": 5.0,
                "p99_ms": 20.0 if valid else 150.0,
                "valid": valid,
            }
        )
        if not valid:
            break
    return out


def perf(throughput: dict[str, tuple[float, int]], skipped=(), latency_groups=None) -> dict:
    groups = latency_groups or list(throughput)
    latency = {
        "all": {
            "count": 1,
            "errors": 0,
            "mean_ms": 1.0,
            "p50_ms": 1.0,
            "p95_ms": 1.0,
            "p99_ms": 1.0,
            "max_ms": 1.0,
        }
    }
    for g in groups:
        latency[g] = {
            "count": 100,
            "errors": 0,
            "mean_ms": 3.0,
            "p50_ms": 2.5,
            "p95_ms": 7.0,
            "p99_ms": 12.25,
            "max_ms": 30.0,
        }
    tp = {}
    for g, (qps, valid_until) in throughput.items():
        s = steps(qps, valid_until)
        valid = [x for x in s if x["valid"]]
        best = max(valid, key=lambda x: x["qps"]) if valid else None
        tp[g] = {
            "steps": s,
            "max_qps": best and best["qps"],
            "max_qps_concurrency": best and best["concurrency"],
        }
    return {
        "base_url": "http://localhost:1",
        "requests": 100,
        "limits": {"p99_ms": 100.0, "max_error_rate": 0.001},
        "warmup": {"requests": 100, "errors": 0, "seconds": 1.0},
        "latency": latency,
        "throughput": tp,
        "skipped_groups": list(skipped),
    }


def write_bundle(
    results: Path,
    engine: str,
    *,
    model="bge-small-en-v1.5",
    track="default",
    hardware="aws-c7i-4xlarge",
    scenario="ecommerce/product-search",
    status="succeeded",
    groups=None,
    na=None,
    violations=None,
    perf_report=None,
    index=True,
    failed_stage=None,
    config_diff=None,
    credits=None,
    synonyms=None,
    checks=None,
) -> Path:
    run_id = "__".join([scenario.replace("/", "__"), engine, model, track, hardware])
    d = results / run_id
    d.mkdir(parents=True)
    stages = {"prepare": {"status": "succeeded", "seconds": 1.0}}
    if failed_stage:
        stages[failed_stage] = {"status": "failed", "seconds": 2.0, "error": "RuntimeError: boom"}
    manifest = {
        "run_id": run_id,
        "scenario": scenario,
        "engine": engine,
        "model": model,
        "track": track,
        "hardware": {"id": hardware},
        "status": status,
        "stages": stages,
        "git_sha": "abc123",
        "irbench_version": "0.1.0",
        "engine_image": {"image": f"{engine}:1", "digest": f"{engine}@sha256:00"},
        "config": {
            "path": f"engines/{engine}/configs/{scenario}.yaml",
            "hash": "f" * 64,
            "diff_vs_default": config_diff or {},
            "credits": credits or [],
            "synonyms": synonyms,
        },
        "dataset": {"id": "wands", "version": "3b74dcf4"},
        "seed": 20261009,
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    if status == "succeeded":
        scores = {
            "run_id": run_id,
            "scenario": scenario,
            "engine": engine,
            "model": model,
            "track": track,
            "relevance": relevance(groups or {"keyword/keyword": 0.5}),
            "checks": checks or [],
            "constraint_violations": violations or {},
            "not_applicable": na or {},
            "errors": 0,
            "perf": None,
        }
        (d / "scores.json").write_text(json.dumps(scores))
        if perf_report is not None:
            (d / "perf.json").write_text(json.dumps(perf_report))
        if index:
            (d / "index.json").write_text(
                json.dumps(
                    {
                        "seconds": 42.5,
                        "memory_peak_bytes": 1566838784,
                        "disk_bytes": 727293952,
                        "n_docs": 42994,
                    }
                )
            )
    return d
