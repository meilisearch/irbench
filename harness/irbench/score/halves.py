"""Tune and test halves of a scenario's queries.

Settings are chosen on the `tune` half (sweeps) and scores are published on the `test` half,
so no published number comes from the queries the settings were picked on. The split is by
source query: every variant of a query (typo, prefix, filtered, sorted...) falls in the same
half as its original. It depends only on the scenario name and the query id, so it is the same
for every engine, track and run, and can be recomputed from the repository.
"""

import hashlib
from collections.abc import Iterable
from typing import Protocol

HALVES = ("tune", "test")
PUBLISHED_HALF = "test"


def _key(scenario: str, qid: str) -> str:
    return hashlib.sha256(f"irbench-half:{scenario}:{qid}".encode()).hexdigest()


def assign_halves(scenario: str, source_qids: Iterable[str]) -> dict[str, str]:
    """Source query id -> "tune" or "test". Queries are ordered by a hash of their id and the
    first half (rounded up) is `tune`, so the halves differ in size by at most one."""
    ordered = sorted(set(source_qids), key=lambda q: (_key(scenario, q), q))
    cut = (len(ordered) + 1) // 2
    return {q: ("tune" if i < cut else "test") for i, q in enumerate(ordered)}


class HasSource(Protocol):
    task_id: str
    variant_of: str
    group: str | None


def topic_of(source_qid: str, group: str | None) -> str:
    """The unit that is split. Scenarios with query groups give each form of a topic its own id
    (`1-keyword`, `1-question`); both forms of a topic go to the same half."""
    suffix = f"-{group}" if group else ""
    return source_qid[: -len(suffix)] if suffix and source_qid.endswith(suffix) else source_qid


def task_halves(scenario: str, tasks: Iterable["HasSource"]) -> dict[str, str]:
    """Task id -> half, for every task of a scenario."""
    tasks = list(tasks)
    topics = {t.task_id: topic_of(t.variant_of, t.group) for t in tasks}
    halves = assign_halves(scenario, topics.values())
    return {task_id: halves[topic] for task_id, topic in topics.items()}
