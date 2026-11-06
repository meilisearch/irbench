from typing import Literal

from pydantic import BaseModel, ConfigDict

Perturbation = Literal["original", "typo", "prefix", "synonym", "accents", "corrected"]
# Variants a dataset supplies that are not perturbations of what users type.
REFERENCE_PERTURBATIONS: frozenset[str] = frozenset({"corrected"})
Mode = Literal["keyword", "semantic", "hybrid", "filtered", "sorted", "faceted"]
BaseMode = Literal["keyword", "semantic", "hybrid"]
BASE_MODES: tuple[BaseMode, ...] = ("keyword", "semantic", "hybrid")

Qrels = dict[str, dict[str, int]]
RunResults = dict[str, list[str]]


class Query(BaseModel):
    model_config = ConfigDict(frozen=True)

    qid: str
    text: str
    variant_of: str
    perturbation: Perturbation
    group: str | None = None
    corrected: str | None = None  # set on originals only; the source of the `corrected` variant


class Filter(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str
    op: Literal["eq", "gte", "lte"]
    value: str | float


class Sort(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str
    direction: Literal["asc", "desc"]


class QueryTask(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    qid: str
    variant_of: str
    perturbation: Perturbation
    text: str
    mode: Mode
    base: BaseMode
    filter: Filter | None = None
    sort: Sort | None = None
    facets: list[str] = []
    group: str | None = None
