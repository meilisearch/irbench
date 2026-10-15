from pathlib import Path

from irbench.datasets.beir import read_jsonl_rows, read_qrels_tsv, write_normalized
from irbench.datasets.wands import NormalizedDataset
from irbench.models import Qrels

FORUMS = ("webmasters", "android", "wordpress")


def normalize_cqadupstack(raw: dict[str, Path], out_dir: Path) -> NormalizedDataset:
    """Three CQADupStack forums in one index. Post and query ids are only unique within a
    forum, so every id gets a `<forum>-` prefix."""
    qrels: Qrels = {}
    for forum in FORUMS:
        for qid, grades in read_qrels_tsv(raw[f"{forum}_qrels"]).items():
            qrels[f"{forum}-{qid}"] = {f"{forum}-{d}": g for d, g in grades.items()}

    def docs():
        for forum in FORUMS:
            for r in read_jsonl_rows(raw[f"{forum}_corpus"]):
                yield {
                    "id": f"{forum}-{r['_id']}",
                    "title": r.get("title") or "",
                    "text": r.get("text") or "",
                    "forum": forum,
                    "tags": list((r.get("metadata") or {}).get("tags") or []),
                }

    def queries():
        for forum in FORUMS:
            for r in read_jsonl_rows(raw[f"{forum}_queries"]):
                qid = f"{forum}-{r['_id']}"
                # BEIR's queries.jsonl may hold more than the test split, and its metadata
                # embeds whole post bodies; keep test queries and only their text.
                if qid in qrels:
                    yield {"qid": qid, "text": " ".join(r["text"].split())}

    return write_normalized(out_dir, docs(), queries(), qrels, ["title", "text"])
