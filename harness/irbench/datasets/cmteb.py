"""`cmteb-ecom` / `cmteb-video`: C-MTEB EcomRetrieval and VideoRetrieval titles, their 1,000
dev queries, and a sample of Multi-CPR train queries whose judged title is in the same corpus
(spec 2026-10-31-irbench-chinese-title-search §3)."""

import random
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq

from irbench.datasets.beir import write_normalized
from irbench.datasets.wands import NormalizedDataset
from irbench.models import Qrels

# Loader constants: part of the dataset's definition, not a run setting (spec §3.3).
SEED = 20261031
SAMPLE_SIZE = 3000


def _tsv(path: Path) -> Iterator[list[str]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line.rstrip("\n").split("\t")


def _text(value: object) -> str:
    return " ".join(str(value).split())


def normalize_cmteb(
    raw: dict[str, Path], out: Path, domain: Literal["ecom", "video"]
) -> NormalizedDataset:
    # --- Titles, exact duplicates merged into the smallest id ----------------------------
    corpus = pq.read_table(raw["corpus"], columns=["id", "text"]).to_pylist()
    keeper: dict[str, str] = {}  # title text -> kept id
    alias: dict[str, str] = {}  # merged id -> kept id
    for row in sorted(corpus, key=lambda r: int(r["id"])):
        doc_id, title = str(row["id"]), str(row["text"]).strip()
        if title in keeper:
            alias[doc_id] = keeper[title]
        else:
            keeper[title] = doc_id
    docs = [{"id": doc_id, "title": title} for title, doc_id in keeper.items()]
    docs.sort(key=lambda d: int(d["id"]))
    known = set(alias) | set(keeper.values())

    moved = 0

    def judged(pid: str) -> str:
        nonlocal moved
        if pid in alias:
            moved += 1
            return alias[pid]
        return pid

    queries: list[dict[str, str]] = []
    qrels: Qrels = {}

    # --- Dev queries (C-MTEB), all kept --------------------------------------------------
    dev_text = {
        str(r["id"]): r["text"]
        for r in pq.read_table(raw["queries"], columns=["id", "text"]).to_pylist()
    }
    dev_qrels = pq.read_table(raw["qrels"], columns=["qid", "pid"]).to_pylist()
    dev_judged: dict[str, str] = {str(r["qid"]): str(r["pid"]) for r in dev_qrels}
    for qid in sorted(dev_text, key=int):
        if qid not in dev_judged:
            continue
        query_id = f"{domain}-dev-{qid}"
        queries.append({"qid": query_id, "text": _text(dev_text[qid]), "group": "dev"})
        qrels[query_id] = {judged(dev_judged[qid]): 1}

    # --- Train queries (Multi-CPR) whose judged title is in this corpus, sampled --------
    train_text = {row[0]: row[1] for row in _tsv(raw["train_queries"])}
    train_judged = {row[0]: row[2] for row in _tsv(raw["train_qrels"])}
    eligible = sorted(
        (q for q, pid in train_judged.items() if pid in known and q in train_text), key=int
    )
    sample = random.Random(SEED).sample(eligible, min(SAMPLE_SIZE, len(eligible)))
    for qid in sorted(sample, key=int):
        query_id = f"{domain}-train-{qid}"
        queries.append({"qid": query_id, "text": _text(train_text[qid]), "group": "train"})
        qrels[query_id] = {judged(train_judged[qid]): 1}

    ds = write_normalized(out, docs, queries, qrels, ["title"])
    ds.dropped["docs"] = ds.dropped.get("docs", 0) + len(alias)
    ds.dropped["moved_judgments"] = moved
    return ds
