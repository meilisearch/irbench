"""Internal-only datasets (`distribution: internal` in dataset.yaml): their data may not be
redistributed, so nothing derived from them is ever published or uploaded."""

from pathlib import Path
from typing import Any

from irbench.datasets.registry import DatasetSpec, load_dataset_spec
from irbench.scenario import load_scenario


class InternalDataError(ValueError):
    """Raised when a publish-like operation is asked to ship internal-only data."""


def manifest_is_internal(manifest: dict[str, Any]) -> bool:
    return manifest.get("dataset", {}).get("distribution") == "internal"


def _spec(root: Path, scenario_id: str) -> DatasetSpec | None:
    """The scenario's dataset spec; None when the scenario or dataset is gone from the repo."""
    try:
        return load_dataset_spec(root, load_scenario(root, scenario_id).dataset)
    except FileNotFoundError:
        return None


def is_internal(root: Path, scenario_id: str) -> bool:
    spec = _spec(root, scenario_id)
    return spec is not None and spec.distribution == "internal"


def refuse_if_internal(
    root: Path, scenario_id: str | None, manifest: dict[str, Any] | None, action: str
) -> None:
    """Raise InternalDataError when the scenario's dataset spec or the given manifest says
    the data is internal. The message names the dataset id and its license."""
    spec = _spec(root, scenario_id) if scenario_id else None
    by_spec = spec is not None and spec.distribution == "internal"
    if not (by_spec or (manifest is not None and manifest_is_internal(manifest))):
        return
    dataset_id = spec.id if spec else (manifest or {}).get("dataset", {}).get("id", "unknown")
    license_text = spec.license if spec else "see the dataset's dataset.yaml"
    raise InternalDataError(
        f"dataset {dataset_id!r} is internal-only ({license_text}); cannot {action}"
    )
