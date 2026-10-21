import csv
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from irbench.io import write_jsonl, write_qrels
from irbench.models import Qrels

csv.field_size_limit(1 << 30)

GRADES = {"Exact": 2, "Partial": 1, "Irrelevant": 0}


@dataclass(frozen=True)
class NormalizedDataset:
    docs: Path
    queries: Path
    qrels: Path
    n_docs: int
    n_queries: int
    # Rows removed while normalizing: keys `docs` and `judgments`.
    dropped: dict[str, int] = field(default_factory=dict)
    # Extra query sets a normalizer writes beyond `queries`/`qrels`, keyed by set name
    # (e.g. "prefixes"), mapping to (queries path, qrels path).
    query_sets: dict[str, tuple[Path, Path]] = field(default_factory=dict)


def _read_tsv(path: Path) -> Iterator[dict[str, str]]:
    # WANDS is tab-separated with unescaped quotes inside text: disable quoting.
    with path.open(encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)


def _split(value: str, sep: str) -> list[str]:
    return [part.strip() for part in value.split(sep) if part.strip()]


def _float_or_none(value: str) -> float | None:
    value = value.strip()
    return float(value) if value else None


def _product_to_doc(row: dict[str, str]) -> dict[str, Any]:
    return {
        "id": row["product_id"].strip(),
        "name": row["product_name"].strip(),
        "product_class": _split(row["product_class"], "|"),
        "category": _split(row["category hierarchy"], " / "),
        "description": row["product_description"].strip(),
        "features": _split(row["product_features"], "|"),
        "rating_count": _float_or_none(row["rating_count"]),
        "average_rating": _float_or_none(row["average_rating"]),
        "review_count": _float_or_none(row["review_count"]),
    }


def normalize_wands(raw: dict[str, Path], out_dir: Path) -> NormalizedDataset:
    docs_path = out_dir / "docs.jsonl"
    queries_path = out_dir / "queries.jsonl"
    qrels_path = out_dir / "qrels.tsv"

    n_docs = write_jsonl(docs_path, (_product_to_doc(r) for r in _read_tsv(raw["product"])))
    n_queries = write_jsonl(
        queries_path,
        (
            {
                "qid": r["query_id"].strip(),
                "text": " ".join(r["query"].split()),
                "query_class": r["query_class"].strip(),
            }
            for r in _read_tsv(raw["query"])
        ),
    )

    qrels: Qrels = {}
    for r in _read_tsv(raw["label"]):
        label = r["label"].strip()
        if label not in GRADES:
            raise ValueError(f"unknown WANDS label {label!r} (row id {r['id']})")
        qrels.setdefault(r["query_id"].strip(), {})[r["product_id"].strip()] = GRADES[label]
    write_qrels(qrels_path, qrels)

    return NormalizedDataset(
        docs_path,
        queries_path,
        qrels_path,
        n_docs,
        n_queries,
        dropped={"docs": 0, "judgments": 0},
    )
