import csv
import re
from pathlib import Path
from typing import Any

from irbench.datasets.beir import read_jsonl_rows, read_qrels_tsv, write_normalized
from irbench.datasets.wands import NormalizedDataset

csv.field_size_limit(1 << 30)


def publish_fields(publish_time: str) -> tuple[int | None, int | None]:
    """CORD-19 `publish_time` (`2020-03-15`, `2020`, or empty) -> (year, YYYYMMDD).
    A year-only date sorts as January 1st."""
    m = re.match(r"\s*(\d{4})(?:-(\d{1,2})-(\d{1,2}))?", publish_time or "")
    if not m:
        return None, None
    year = int(m[1])
    month, day = (int(m[2]), int(m[3])) if m[2] else (1, 1)
    return year, year * 10000 + month * 100 + day


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


def normalize_trec_covid(raw: dict[str, Path], out_dir: Path) -> NormalizedDataset:
    meta = _metadata(raw["metadata"])
    docs = (
        {
            "id": r["_id"],
            "title": r.get("title") or "",
            "text": r.get("text") or "",
            **meta.get(r["_id"], EMPTY),
        }
        for r in read_jsonl_rows(raw["corpus"])
    )
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
            qrels[qid] = dict(topic_qrels.get(topic, {}))
    return write_normalized(out_dir, docs, queries, qrels, ["title", "text"])
