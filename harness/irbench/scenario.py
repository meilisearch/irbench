from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, PrivateAttr, model_validator

from irbench.models import BaseMode, Mode, Sort


class IndexFields(BaseModel):
    searchable: list[str]
    filterable: list[str]
    sortable: list[str]
    facets: list[str]


class FilterTemplate(BaseModel):
    field: str
    op: Literal["eq", "gte", "lte"]
    values: list[float] | None = None


class PerturbationSpec(BaseModel):
    seed: int
    kinds: list[Literal["typo", "prefix", "synonym"]]
    vocabulary: str


class Scenario(BaseModel):
    id: str
    dataset: str
    id_field: str = "id"
    index_fields: IndexFields
    modes: list[Mode]
    structured_bases: list[BaseMode]
    embedding_fields: list[str]
    filter_templates: list[FilterTemplate]
    sort_templates: list[Sort]
    top_k: int = 100
    index_timeout_s: int
    perturbations: PerturbationSpec
    synonyms: str

    _directory: Path = PrivateAttr()

    @model_validator(mode="after")
    def _check_consistency(self) -> "Scenario":
        f = self.index_fields
        for t in self.filter_templates:
            if t.field not in f.filterable:
                raise ValueError(f"filter template field {t.field!r} is not filterable")
            if (t.op == "eq") != (t.values is None):
                raise ValueError(
                    f"filter template {t.field!r}: eq takes no values, ranges need values"
                )
        for s in self.sort_templates:
            if s.field not in f.sortable:
                raise ValueError(f"sort template field {s.field!r} is not sortable")
        if "faceted" in self.modes and not f.facets:
            raise ValueError("faceted mode needs at least one facet field")
        return self

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def vocabulary_path(self) -> Path:
        return self._directory / self.perturbations.vocabulary

    @property
    def synonyms_path(self) -> Path:
        return self._directory / self.synonyms


def load_scenario(root: Path, scenario_id: str) -> Scenario:
    directory = root / "scenarios" / scenario_id
    path = directory / "scenario.yaml"
    scenario = Scenario.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if scenario.id != scenario_id:
        raise ValueError(f"{path}: id {scenario.id!r} does not match {scenario_id!r}")
    scenario._directory = directory
    return scenario
