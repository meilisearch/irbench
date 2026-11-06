import csv
import re
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

from irbench.datasets.beir import read_jsonl_rows, read_qrels_tsv, write_normalized
from irbench.datasets.wands import NormalizedDataset

csv.field_size_limit(1 << 30)

EPOCH = date(1970, 1, 1)


def publish_fields(publish_time: str) -> tuple[int | None, int | None]:
    """CORD-19 `publish_time` (`2020-03-15`, `2020`, or empty) -> (year, days since 1970-01-01).
    A year-only or invalid date sorts as January 1st. Days, not YYYYMMDD: engines that sort on
    float32 (Elasticsearch `float`, Qdrant formula scores) are exact only up to 2**24."""
    m = re.match(r"\s*(\d{4})(?:-(\d{1,2})-(\d{1,2}))?", publish_time or "")
    if not m:
        return None, None
    year = int(m[1])
    try:
        day = date(year, int(m[2]), int(m[3])) if m[2] else date(year, 1, 1)
    except ValueError:
        day = date(year, 1, 1)
    return year, (day - EPOCH).days


def _metadata(path: Path) -> dict[str, dict[str, Any]]:
    """First metadata row per cord_uid, in file order."""
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            uid = row["cord_uid"].strip()
            if uid in out:
                continue
            year, date = publish_fields(row.get("publish_time", ""))
            out[uid] = {
                "publish_year": year,
                "publish_date": date,
                "source": [s.strip() for s in (row.get("source_x") or "").split(";") if s.strip()],
                "journal": (row.get("journal") or "").strip() or None,
            }
    return out


EMPTY: dict[str, Any] = {"publish_year": None, "publish_date": None, "source": [], "journal": None}


def normalize_trec_covid(
    raw: dict[str, Path],
    out_dir: Path,
    min_abstract_chars: int = 0,
    relevant_only: bool = False,
) -> NormalizedDataset:
    """BEIR TREC-COVID, joined with CORD-19 metadata when `raw` has a `metadata` file.

    `min_abstract_chars` and `relevant_only` reproduce the relevancy-bench files: papers whose
    abstract (stripped) is shorter are dropped, and only judgments with a grade above 0 kept."""
    meta = _metadata(raw["metadata"]) if "metadata" in raw else None
    short = 0

    def docs() -> Iterator[dict[str, Any]]:
        nonlocal short
        for r in read_jsonl_rows(raw["corpus"]):
            text = r.get("text") or ""
            if len(text.strip()) < min_abstract_chars:
                short += 1
                continue
            doc = {"id": r["_id"], "title": r.get("title") or "", "text": text}
            if meta is not None:
                doc.update(meta.get(r["_id"], EMPTY))
            yield doc

    topic_qrels = read_qrels_tsv(raw["qrels"])
    queries: list[dict[str, Any]] = []
    qrels: dict[str, dict[str, int]] = {}
    for r in read_jsonl_rows(raw["queries"]):
        topic = r["_id"]
        forms = {"question": r["text"], "keyword": (r.get("metadata") or {}).get("query")}
        for group, text in forms.items():
            if not text:
                continue
            qid = f"{topic}-{group}"
            queries.append({"qid": qid, "text": " ".join(text.split()), "group": group})
            grades = topic_qrels.get(topic, {})
            qrels[qid] = {d: g for d, g in grades.items() if g > 0 or not relevant_only}
    out = write_normalized(out_dir, docs(), queries, qrels, ["title", "text"])
    out.dropped["docs"] += short
    return out
