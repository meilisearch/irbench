"""Load run bundles (results/<run-id>/ or .irbench/runs/<run-id>/) for the report."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

from irbench.io import read_json
from irbench.models import BASE_MODES, Mode
from irbench.paths import slug

MODES: tuple[str, ...] = get_args(Mode)
PERTURBATIONS = ("original", "typo", "prefix", "synonym")
REQUIRED_KEYS = ("run_id", "scenario", "engine", "model", "track", "hardware", "status")


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
    perf: dict[str, Any] | None = None
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
    for d in sorted(p for p in results_dir.iterdir() if (p / "manifest.json").is_file()):
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
    missing = [k for k in REQUIRED_KEYS if k not in manifest]
    if missing:
        raise ValueError(f"manifest.json lacks {', '.join(missing)}")
    files = {
        name: read_json(d / f"{name}.json") if (d / f"{name}.json").exists() else None
        for name in ("scores", "perf", "index")
    }
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


def _validate_files(d: Path, files: dict[str, Any]) -> None:
    """Validate structure of loaded JSON files; raise ValueError if malformed."""
    if files["scores"] is not None:
        if not isinstance(files["scores"], dict):
            raise ValueError("scores.json must be a dict")
        if "relevance" not in files["scores"]:
            raise ValueError("scores.json lacks relevance")
        if not isinstance(files["scores"]["relevance"], list):
            raise ValueError("scores.json relevance must be a list")
        if "not_applicable" not in files["scores"]:
            raise ValueError("scores.json lacks not_applicable")
        if not isinstance(files["scores"]["not_applicable"], dict):
            raise ValueError("scores.json not_applicable must be a dict")
        if "constraint_violations" not in files["scores"]:
            raise ValueError("scores.json lacks constraint_violations")
        if not isinstance(files["scores"]["constraint_violations"], dict):
            raise ValueError("scores.json constraint_violations must be a dict")

    if files["perf"] is not None:
        if not isinstance(files["perf"], dict):
            raise ValueError("perf.json must be a dict")
        required_perf = ("limits", "latency", "throughput", "skipped_groups")
        missing_perf = [k for k in required_perf if k not in files["perf"]]
        if missing_perf:
            raise ValueError(f"perf.json lacks {', '.join(missing_perf)}")

    if files["index"] is not None:
        if not isinstance(files["index"], dict):
            raise ValueError("index.json must be a dict")
        if "seconds" not in files["index"]:
            raise ValueError("index.json lacks seconds")
        if "n_docs" not in files["index"]:
            raise ValueError("index.json lacks n_docs")
