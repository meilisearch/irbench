from dataclasses import dataclass
from pathlib import Path


def _slug(value: str) -> str:
    return value.replace("/", "__")


@dataclass(frozen=True)
class Workspace:
    root: Path

    @property
    def cache(self) -> Path:
        return self.root / ".irbench"

    def dataset_dir(self, dataset_id: str) -> Path:
        return self.cache / "data" / dataset_id

    def raw_dir(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "raw"

    def scenario_dir(self, scenario_id: str) -> Path:
        return self.cache / "scenarios" / _slug(scenario_id)

    def embeddings_dir(self, dataset_id: str, model_id: str) -> Path:
        return self.cache / "embeddings" / dataset_id / _slug(model_id)
