"""Copy a run's outputs into results/<run-id>/ (spec §4). Only reference-machine runs are
published (spec §7); failed runs are published too, so the report shows the failure (§9)."""

import shutil
from pathlib import Path

import yaml

from irbench.io import read_json
from irbench.paths import Workspace
from irbench.scenario import load_scenario

# Small, self-contained files. rendered/ and run.trec are reproducible from the manifest.
BUNDLE_FILES = ("manifest.json", "scores.json", "perf.json", "index.json")
UNPUBLISHABLE_PROFILES = ("local",)


def publish_run(root: Path, run_id: str, results_dir: Path | None = None) -> Path:
    src = Workspace(root).run_dir(run_id)
    if not (src / "manifest.json").exists():
        raise FileNotFoundError(f"{src} has no manifest.json: not a run directory")
    manifest = read_json(src / "manifest.json")
    try:
        profile = manifest["hardware"]["id"]
        if not isinstance(profile, str):
            raise ValueError(f"{src}/manifest.json has no hardware.id: not a complete run manifest")
    except (KeyError, TypeError) as e:
        raise ValueError(
            f"{src}/manifest.json has no hardware.id: not a complete run manifest"
        ) from e
    if profile in UNPUBLISHABLE_PROFILES:
        raise ValueError(
            f"{run_id} ran on the {profile!r} profile; only reference-machine results are "
            "published. Preview it with `irbench report --results .irbench/runs`."
        )
    _check_publishable(root, run_id, manifest.get("scenario"))
    dest = (results_dir or root / "results") / run_id
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    names = [n for n in BUNDLE_FILES if (src / n).exists()]
    names += sorted(p.name for p in src.glob("perf-p*.json"))
    for name in names:
        shutil.copy2(src / name, tmp / name)
    # Atomic replacement: keep old bundle safe until new one is in place.
    old = dest.with_name(dest.name + ".old")
    if dest.exists():
        shutil.rmtree(old, ignore_errors=True)
        dest.rename(old)
    tmp.replace(dest)
    shutil.rmtree(old, ignore_errors=True)
    return dest


def _check_publishable(root: Path, run_id: str, scenario_id: str | None) -> None:
    """Refuse runs of a scenario marked `publishable: false` (data without publishing terms).
    Manifests that name no scenario predate the field and are not checked."""
    if scenario_id is None:
        return
    try:
        scenario = load_scenario(root, scenario_id)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        raise ValueError(
            f"{run_id}: cannot read scenario {scenario_id} to check it may be published: {e}"
        ) from e
    if not scenario.publishable:
        raise ValueError(
            f"{run_id}: scenario {scenario_id} is internal (publishable: false), its data has "
            "no publishing terms; see the Chinese title search spec §7. Preview it with "
            "`irbench report --results .irbench/runs`."
        )
