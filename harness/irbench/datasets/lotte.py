from pathlib import Path
from urllib.parse import urlparse

from irbench.datasets.beir import read_jsonl_rows, write_normalized
from irbench.datasets.wands import NormalizedDataset
from irbench.models import Qrels


def _forum(url: str) -> str:
    """`https://cooking.stackexchange.com/...` -> `cooking`."""
    return urlparse(url).netloc.split(".")[0]


def normalize_lotte(raw: dict[str, Path], out_dir: Path) -> NormalizedDataset:
    docs = ({"id": str(r["doc_id"]), "text": r["text"]} for r in read_jsonl_rows(raw["collection"]))
    queries: list[dict[str, str]] = []
    qrels: Qrels = {}
    for r in read_jsonl_rows(raw["qas"]):
        qid = str(r["qid"])
        queries.append(
            {"qid": qid, "text": " ".join(r["query"].split()), "group": _forum(r["url"])}
        )
        qrels[qid] = {str(pid): 1 for pid in r["answer_pids"]}
    return write_normalized(out_dir, docs, queries, qrels, ["text"])
