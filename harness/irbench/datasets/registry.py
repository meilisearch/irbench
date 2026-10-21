from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import yaml
from pydantic import BaseModel

from irbench.datasets.fetch import fetch_file, fetch_zip_member


class FileSpec(BaseModel):
    url: str
    sha256: str
    # A path inside the zip at `url`; only that member is fetched (range requests).
    member: str | None = None


class DatasetSpec(BaseModel):
    id: str
    version: str
    license: str
    source: str
    files: dict[str, FileSpec]
    # Extra keyword arguments passed to the normalizer (e.g. `corpus: full|pool`); a loader
    # parameter baked into the dataset's definition, not a run setting.
    params: dict[str, Any] = {}


def load_dataset_spec(root: Path, dataset_id: str) -> DatasetSpec:
    path = root / "datasets" / dataset_id / "dataset.yaml"
    spec = DatasetSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if spec.id != dataset_id:
        raise ValueError(f"{path}: id {spec.id!r} does not match directory {dataset_id!r}")
    return spec


def fetch_dataset(
    spec: DatasetSpec, raw_dir: Path, client: httpx.Client | None = None
) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for name, f in spec.files.items():
        if f.member:
            out[name] = fetch_zip_member(f.url, f.member, raw_dir / f.member, f.sha256, client)
        else:
            dest = raw_dir / Path(urlparse(f.url).path).name
            out[name] = fetch_file(f.url, dest, f.sha256, client)
    return out
