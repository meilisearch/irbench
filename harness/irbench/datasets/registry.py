from pathlib import Path
from urllib.parse import urlparse

import httpx
import yaml
from pydantic import BaseModel

from irbench.datasets.fetch import fetch_file


class FileSpec(BaseModel):
    url: str
    sha256: str


class DatasetSpec(BaseModel):
    id: str
    version: str
    license: str
    source: str
    files: dict[str, FileSpec]


def load_dataset_spec(root: Path, dataset_id: str) -> DatasetSpec:
    path = root / "datasets" / dataset_id / "dataset.yaml"
    spec = DatasetSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if spec.id != dataset_id:
        raise ValueError(f"{path}: id {spec.id!r} does not match directory {dataset_id!r}")
    return spec


def fetch_dataset(
    spec: DatasetSpec, raw_dir: Path, client: httpx.Client | None = None
) -> dict[str, Path]:
    return {
        name: fetch_file(f.url, raw_dir / Path(urlparse(f.url).path).name, f.sha256, client)
        for name, f in spec.files.items()
    }
