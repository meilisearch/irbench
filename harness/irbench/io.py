import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from irbench.models import Qrels, RunResults


def _check_id(value: str) -> str:
    if not value or any(c.isspace() for c in value):
        raise ValueError(f"id must be non-empty and contain no whitespace: {value!r}")
    return value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_qrels(path: Path, qrels: Qrels) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid in sorted(qrels):
            for doc_id in sorted(qrels[qid]):
                f.write(f"{_check_id(qid)} 0 {_check_id(doc_id)} {qrels[qid][doc_id]}\n")


def read_qrels(path: Path) -> Qrels:
    qrels: Qrels = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qid, _, doc_id, grade = line.split()
            qrels.setdefault(qid, {})[doc_id] = int(grade)
    return qrels


def write_run(path: Path, results: RunResults, tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for task_id in sorted(results):
            ids = results[task_id]
            for rank, doc_id in enumerate(ids, start=1):
                score = len(ids) - rank + 1
                f.write(f"{_check_id(task_id)} Q0 {_check_id(doc_id)} {rank} {score} {tag}\n")


def read_run(path: Path) -> RunResults:
    ranked: dict[str, list[tuple[int, str]]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            task_id, _, doc_id, rank, _score, _tag = line.split()
            ranked.setdefault(task_id, []).append((int(rank), doc_id))
    return {tid: [doc for _, doc in sorted(pairs)] for tid, pairs in ranked.items()}
