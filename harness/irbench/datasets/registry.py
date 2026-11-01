from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
import yaml
from pydantic import BaseModel, model_validator

from irbench.datasets.fetch import (
    ChecksumError,
    ManualFileMissing,
    fetch_file,
    fetch_zip_member,
    sha256_file,
)


class FileSpec(BaseModel):
    # Exactly one of `url` (downloaded) and `manual` (placed in the raw directory by the
    # runner, e.g. files behind a login; the text says how to get them).
    url: str | None = None
    sha256: str
    # A path inside the zip at `url`; only that member is fetched (range requests).
    member: str | None = None
    manual: str | None = None
    # The name a manual file must have in the raw directory.
    filename: str | None = None

    @model_validator(mode="after")
    def _check_source(self) -> "FileSpec":
        if (self.url is None) == (self.manual is None):
            raise ValueError("exactly one of `url` and `manual` is required")
        if self.manual is not None and not self.filename:
            raise ValueError("`manual` requires `filename`")
        if self.member and self.url is None:
            raise ValueError("`member` requires `url`")
        return self


class DatasetSpec(BaseModel):
    id: str
    version: str
    license: str
    source: str
    files: dict[str, FileSpec]
    # Extra keyword arguments passed to the normalizer (e.g. `corpus: full|pool`); a loader
    # parameter baked into the dataset's definition, not a run setting.
    params: dict[str, Any] = {}
    # A dataset id whose raw directory is reused, so datasets cut from the same source files
    # (e.g. ESCI locales) download them once.
    raw_from: str | None = None
    # "internal" data may not be redistributed: runs on it are never published and its
    # vectors are never packed or downloaded (irbench.internal).
    distribution: Literal["public", "internal"] = "public"


def load_dataset_spec(root: Path, dataset_id: str) -> DatasetSpec:
    path = root / "datasets" / dataset_id / "dataset.yaml"
    spec = DatasetSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if spec.id != dataset_id:
        raise ValueError(f"{path}: id {spec.id!r} does not match directory {dataset_id!r}")
    return spec


def fetch_dataset(
    spec: DatasetSpec, raw_dir: Path, client: httpx.Client | None = None
) -> dict[str, Path]:
    # Manual files first: a runner who has not downloaded them fails before any download.
    manual: dict[str, Path] = {}
    instructions: list[str] = []
    for name, f in spec.files.items():
        if f.manual is not None and f.filename is not None:
            manual[name] = raw_dir / f.filename
            if f.manual not in instructions:
                instructions.append(f.manual)
    missing = [p for p in manual.values() if not p.exists()]
    if missing:
        raise ManualFileMissing(missing, "\n".join(instructions))
    for name, path in manual.items():
        actual = sha256_file(path)
        if actual != spec.files[name].sha256:
            raise ChecksumError(
                f"{path}: expected sha256 {spec.files[name].sha256}, got {actual}; "
                "download the file again (the file was left in place)"
            )

    out: dict[str, Path] = {}
    for name, f in spec.files.items():
        if name in manual:
            out[name] = manual[name]
        elif f.url is None:  # unreachable: FileSpec requires url or manual
            raise AssertionError(name)
        elif f.member:
            out[name] = fetch_zip_member(f.url, f.member, raw_dir / f.member, f.sha256, client)
        else:
            dest = raw_dir / Path(urlparse(f.url).path).name
            out[name] = fetch_file(f.url, dest, f.sha256, client)
    return out
