"""Readers for BEIR-style files (corpus.jsonl, queries.jsonl, qrels TSV) and the shared
normalization rules of spec 2026-10-14-irbench-knowledge-base §3.4."""

import csv
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from irbench.datasets.wands import NormalizedDataset
from irbench.io import write_jsonl, write_qrels
from irbench.models import Qrels


def read_jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_qrels_tsv(path: Path) -> Qrels:
    """BEIR qrels: tab-separated with a header row; negative grades count as 0."""
    qrels: Qrels = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            grade = max(0, int(row["score"]))
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = grade
    return qrels


def _has_text(doc: dict[str, Any], fields: list[str]) -> bool:
    return any(str(doc.get(f) or "").strip() for f in fields)


def write_normalized(
    out_dir: Path,
    docs: Iterable[dict[str, Any]],
    queries: Iterable[dict[str, Any]],
    qrels: Qrels,
    text_fields: list[str],
) -> NormalizedDataset:
    """Write the normalized files. Docs whose `text_fields` are all empty are dropped, and so
    are judgments pointing at dropped or unknown docs; both counts are reported."""
    kept: set[str] = set()
    dropped_docs = 0

    def keep(rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        nonlocal dropped_docs
        for d in rows:
            if _has_text(d, text_fields):
                kept.add(d["id"])
                yield d
            else:
                dropped_docs += 1

    docs_path = out_dir / "docs.jsonl"
    queries_path = out_dir / "queries.jsonl"
    qrels_path = out_dir / "qrels.tsv"
    n_docs = write_jsonl(docs_path, keep(docs))
    n_queries = write_jsonl(queries_path, queries)

    dropped_judgments = 0
    clean: Qrels = {}
    for qid, grades in qrels.items():
        for doc_id, grade in grades.items():
            if doc_id in kept:
                clean.setdefault(qid, {})[doc_id] = grade
            else:
                dropped_judgments += 1
    write_qrels(qrels_path, clean)
    return NormalizedDataset(
        docs_path,
        queries_path,
        qrels_path,
        n_docs,
        n_queries,
        dropped={"docs": dropped_docs, "judgments": dropped_judgments},
    )
