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
    numeric: list[str] = []


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
    max_facet_values: int = 1000
    facet_verify_limit: int = 1000
    request_timeout_s: float = 10.0
    max_error_rate: float = 0.001
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
            if t.op != "eq" and t.field not in f.numeric:
                raise ValueError(
                    f"range filter template field {t.field!r} must be listed in index_fields.numeric"
                )
        for name in f.sortable:
            if name not in f.numeric:
                raise ValueError(
                    f"sortable field {name!r} must be numeric (v0.1 sorts numbers only)"
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
