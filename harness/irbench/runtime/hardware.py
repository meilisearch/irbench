from pathlib import Path

import yaml
from pydantic import BaseModel


class EngineLimits(BaseModel):
    cpuset: str | None = None
    memory: str | None = None


class LoadgenLimits(BaseModel):
    cpuset: str | None = None


class HardwareProfile(BaseModel):
    id: str
    description: str = ""
    engine: EngineLimits = EngineLimits()
    loadgen: LoadgenLimits = LoadgenLimits()


def load_hardware(root: Path, profile_id: str) -> HardwareProfile:
    path = root / "hardware" / f"{profile_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: unknown hardware profile {profile_id!r}")
    profile = HardwareProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if profile.id != profile_id:
        raise ValueError(f"{path}: id {profile.id!r} does not match {profile_id!r}")
    return profile
