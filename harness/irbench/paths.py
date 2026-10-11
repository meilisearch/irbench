from dataclasses import dataclass
from pathlib import Path


def slug(value: str) -> str:
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
        return self.cache / "scenarios" / slug(scenario_id)

    def embeddings_dir(self, dataset_id: str, model_id: str) -> Path:
        return self.cache / "embeddings" / dataset_id / slug(model_id)

    def doc_vectors_path(self, dataset_id: str, model_id: str) -> Path:
        return self.embeddings_dir(dataset_id, model_id) / "docs.parquet"

    def query_vectors_path(self, scenario_id: str, model_id: str) -> Path:
        return self.scenario_dir(scenario_id) / "embeddings" / slug(model_id) / "queries.parquet"

    def run_dir(self, run_id: str) -> Path:
        return self.cache / "runs" / run_id
