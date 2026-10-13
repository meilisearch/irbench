"""Load run bundles (results/<run-id>/ or .irbench/runs/<run-id>/) for the report."""

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

from irbench.io import read_json
from irbench.models import BASE_MODES, Mode
from irbench.paths import slug
from irbench.perf_files import perf_report_paths, read_perf_reports

MODES: tuple[str, ...] = get_args(Mode)
PERTURBATIONS = ("original", "typo", "prefix", "synonym")
REQUIRED_KEYS = ("run_id", "scenario", "engine", "model", "track", "hardware", "status")
REQUIRED_RELEVANCE_KEYS = ("mode", "base", "perturbation", "n", "metrics")
# run_id and hardware.id become path components (runs/{run_id}.html, the scenario slug);
# this also excludes "." and "..".
SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


def group_key(group: str) -> tuple[int, int]:
    mode, _, base = group.partition("/")
    return MODES.index(mode), BASE_MODES.index(base)


@dataclass(frozen=True)
class RunResult:
    run_id: str
    scenario: str
    engine: str
    model: str
    track: str
    hardware: str
    status: str
    failure: str | None
    manifest: dict[str, Any]
    scores: dict[str, Any] | None = None
    perf: dict[int, dict[str, Any]] | None = None
    index: dict[str, Any] | None = None

    @property
    def label(self) -> str:
        return self.engine if self.track == "default" else f"{self.engine} ({self.track})"

    @property
    def groups(self) -> list[str]:
        if self.scores is None:
            return []
        found = {f"{g['mode']}/{g['base']}" for g in self.scores["relevance"]}
        found |= {key.replace(":", "/") for key in self.scores["not_applicable"]}
        return sorted(found, key=group_key)

    def ndcg(self, group: str, perturbation: str = "all") -> float | None:
        if self.scores is None:
            return None
        mode, _, base = group.partition("/")
        for g in self.scores["relevance"]:
            if (g["mode"], g["base"], g["perturbation"]) == (mode, base, perturbation):
                return g["metrics"]["ndcg@10"]
        return None

    def na_reason(self, group: str) -> str | None:
        if self.scores is None:
            return None
        return self.scores["not_applicable"].get(group.replace("/", ":"))

    def violation(self, group: str) -> str | None:
        if self.scores is None:
            return None
        return self.scores["constraint_violations"].get(group.replace("/", ":"))


@dataclass(frozen=True)
class ScenarioResults:
    scenario: str
    hardware: str
    runs: list[RunResult]

    @property
    def slug(self) -> str:
        return f"{slug(self.scenario)}__{self.hardware}"

    @property
    def models(self) -> list[str]:
        return sorted({r.model for r in self.runs})

    @property
    def groups(self) -> list[str]:
        return sorted({g for r in self.runs for g in r.groups}, key=group_key)


@dataclass(frozen=True)
class Unreadable:
    path: str
    error: str


def load_results(results_dir: Path) -> tuple[list[ScenarioResults], list[Unreadable]]:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"{results_dir} is not a directory")
    runs: list[RunResult] = []
    unreadable: list[Unreadable] = []
    candidates = (
        p
        for p in results_dir.iterdir()
        if (p / "manifest.json").is_file() and not p.name.endswith((".tmp", ".old"))
    )
    for d in sorted(candidates):
        try:
            runs.append(_load_run(d))
        except (ValueError, KeyError, TypeError) as e:  # JSONDecodeError is a ValueError
            unreadable.append(Unreadable(d.name, f"{type(e).__name__}: {e}"))

    grouped: dict[tuple[str, str], list[RunResult]] = defaultdict(list)
    for run in runs:
        grouped[(run.scenario, run.hardware)].append(run)
    scenarios = [
        ScenarioResults(scenario, hardware, sorted(rs, key=lambda r: (r.model, r.engine, r.track)))
        for (scenario, hardware), rs in sorted(grouped.items())
    ]
    return scenarios, unreadable


def _load_run(d: Path) -> RunResult:
    manifest = read_json(d / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError(f"{d}/manifest.json must be a dict")
    missing = [k for k in REQUIRED_KEYS if k not in manifest]
    if missing:
        raise ValueError(f"manifest.json lacks {', '.join(missing)}")
    _validate_manifest(d, manifest)
    files = {
        name: read_json(d / f"{name}.json") if (d / f"{name}.json").exists() else None
        for name in ("scores", "index")
    }
    files["perf"] = read_perf_reports(d) or None
    _validate_files(d, files)
    status, failure = manifest["status"], _failure(manifest)
    if status == "succeeded" and files["scores"] is None:
        status, failure = "incomplete", "the run succeeded but the bundle has no scores.json"
    return RunResult(
        run_id=manifest["run_id"],
        scenario=manifest["scenario"],
        engine=manifest["engine"],
        model=manifest["model"],
        track=manifest["track"],
        hardware=manifest["hardware"]["id"],
        status=status,
        failure=failure,
        manifest=manifest,
        **files,
    )


def _failure(manifest: dict[str, Any]) -> str | None:
    if manifest["status"] == "succeeded":
        return None
    for stage, info in manifest.get("stages", {}).items():
        if info.get("status") == "failed":
            return f"{stage}: {info.get('error', 'no error recorded')}"
    return f"run status {manifest['status']!r}"


def _safe_path_component(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(SAFE_PATH_COMPONENT.match(value))
        and value
        not in (
            ".",
            "..",
        )
    )


def _validate_manifest(d: Path, manifest: dict[str, Any]) -> None:
    """Validate the shapes the renderer reads from manifest.json, including run_id and
    hardware.id, which become path components (runs/{run_id}.html, the scenario slug)."""
    p = f"{d}/manifest.json"
    run_id = manifest["run_id"]
    if not _safe_path_component(run_id):
        raise ValueError(f"{p}: run_id {run_id!r} is not a safe path component")
    if run_id != d.name:
        raise ValueError(f"{p}: run_id {run_id!r} does not match its directory {d.name!r}")

    hardware = manifest["hardware"]
    if not isinstance(hardware, dict) or not _safe_path_component(hardware.get("id")):
        raise ValueError(f"{p}: hardware.id must be a safe path component")

    stages = manifest.get("stages", {})
    if not isinstance(stages, dict):
        raise ValueError(f"{p}: stages must be a dict")
    for stage, info in stages.items():
        if not isinstance(info, dict):
            raise ValueError(f"{p}: stages.{stage} must be a dict")

    if "config" in manifest:
        config = manifest["config"]
        if not isinstance(config, dict):
            raise ValueError(f"{p}: config must be a dict")
        synonyms = config.get("synonyms")
        if synonyms is not None and not isinstance(synonyms, str):
            raise ValueError(f"{p}: config.synonyms must be a string or null")
        credits = config.get("credits", [])
        if not isinstance(credits, list):
            raise ValueError(f"{p}: config.credits must be a list")
        for c in credits:
            if not isinstance(c, dict) or not isinstance(c.get("name"), str):
                raise ValueError(f"{p}: config.credits entries need a string name")
            url = c.get("url")
            if url is not None and not isinstance(url, str):
                raise ValueError(f"{p}: config.credits.url must be a string or null")


def _validate_relevance_item(p: str, item: Any) -> None:
    if not isinstance(item, dict):
        raise ValueError(f"{p}: relevance item must be a dict")
    missing = [k for k in REQUIRED_RELEVANCE_KEYS if k not in item]
    if missing:
        raise ValueError(f"{p}: relevance item lacks {', '.join(missing)}")
    if item["mode"] not in MODES or item["base"] not in BASE_MODES:
        raise ValueError(
            f"{p}: relevance item has unknown mode/base {item['mode']!r}/{item['base']!r}"
        )


def _validate_group_key(p: str, field: str, key: str) -> None:
    mode, sep, base = key.partition(":")
    if not sep or mode not in MODES or base not in BASE_MODES:
        raise ValueError(
            f"{p}: {field} key {key!r} must be <mode>:<base> with a known mode and base"
        )


def _validate_number(p: str, key: str, value: Any) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{p}: {key} must be a number")


def _validate_files(d: Path, files: dict[str, Any]) -> None:
    """Validate structure of loaded JSON files; raise ValueError if malformed."""
    if files["scores"] is not None:
        p = f"{d}/scores.json"
        scores = files["scores"]
        if not isinstance(scores, dict):
            raise ValueError(f"{p} must be a dict")
        if "relevance" not in scores:
            raise ValueError(f"{p} lacks relevance")
        if not isinstance(scores["relevance"], list):
            raise ValueError(f"{p} relevance must be a list")
        for item in scores["relevance"]:
            _validate_relevance_item(p, item)
        if "not_applicable" not in scores:
            raise ValueError(f"{p} lacks not_applicable")
        if not isinstance(scores["not_applicable"], dict):
            raise ValueError(f"{p} not_applicable must be a dict")
        for key in scores["not_applicable"]:
            _validate_group_key(p, "not_applicable", key)
        if "constraint_violations" not in scores:
            raise ValueError(f"{p} lacks constraint_violations")
        if not isinstance(scores["constraint_violations"], dict):
            raise ValueError(f"{p} constraint_violations must be a dict")
        for key in scores["constraint_violations"]:
            _validate_group_key(p, "constraint_violations", key)

    if files["perf"] is not None:
        paths = perf_report_paths(d)
        for size, report in files["perf"].items():
            p = str(paths[size])
            if not isinstance(report, dict):
                raise ValueError(f"{p} must be a dict")
            required_perf = ("limits", "latency", "throughput", "skipped_groups")
            missing_perf = [k for k in required_perf if k not in report]
            if missing_perf:
                raise ValueError(f"{p} lacks {', '.join(missing_perf)}")

    if files["index"] is not None:
        p = f"{d}/index.json"
        index = files["index"]
        if not isinstance(index, dict):
            raise ValueError(f"{p} must be a dict")
        if "seconds" not in index:
            raise ValueError(f"{p} lacks seconds")
        if "n_docs" not in index:
            raise ValueError(f"{p} lacks n_docs")
        for key in ("seconds", "n_docs", "memory_peak_bytes", "disk_bytes"):
            if key in index and index[key] is not None:
                _validate_number(p, key, index[key])
