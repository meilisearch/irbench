"""`amazon-esci`: Amazon Shopping Queries (ESCI) products and judgments, joined with
AmazonQAC typed-prefix sessions (spec 2026-10-20-irbench-amazon-esci §2, §3)."""

import csv
import html
import random
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq

from irbench.datasets.wands import NormalizedDataset
from irbench.io import write_jsonl, write_qrels
from irbench.models import Qrels

csv.field_size_limit(1 << 30)

# Loader constants: part of the dataset's definition, not a run setting (spec §3.1).
SEED = 20261020
SAMPLE_SIZE = 2000

GRADES = {"E": 2, "S": 1, "C": 0, "I": 0}

LE25, G25_50, G50_75, G75_100, COMPLETE, DIVERGES = (
    "≤25%",
    "25–50%",
    "50–75%",
    "75–<100%",
    "complete",
    "diverges",
)
GROUP_ORDER = [LE25, G25_50, G50_75, G75_100, COMPLETE, DIVERGES]
# Groups built from a session's typed prefixes; `complete` is added separately, once per
# query, from the ESCI query text itself (spec §3.2).
PREFIX_GROUPS = [LE25, G25_50, G50_75, G75_100, DIVERGES]
QAC_COLUMNS = ["prefixes", "final_search_term"]
QAC_BATCH_SIZE = 500_000


def _normalize_key(text: str) -> str:
    return " ".join(text.lower().split())


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _strip_html(value: Any) -> str | None:
    """Strip tags, unescape entities, collapse whitespace; empty -> None (spec §3.1)."""
    if value is None:
        return None
    raw = str(value)
    if not raw.strip():
        return None
    parser = _TextExtractor()
    parser.feed(raw)
    parser.close()
    text = " ".join(html.unescape("".join(parser.parts)).split())
    return text or None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _read_table(path: Path, columns: list[str]) -> Iterator[dict[str, Any]]:
    table = pq.read_table(path, columns=columns)
    yield from table.to_pylist()


def _read_sources(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[str(row["query_id"]).strip()] = row["source"].strip()
    return out


def _product_doc(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["product_id"]),
        "title": _str_or_none(row.get("product_title")),
        "brand": _str_or_none(row.get("product_brand")),
        "color": _str_or_none(row.get("product_color")),
        "bullet_points": _str_or_none(row.get("product_bullet_point")),
        "description": _strip_html(row.get("product_description")),
    }


def _group_for(prefix: str, final: str) -> str | None:
    """Spec §3.2 completion group, `None` for the literal-complete prefix (handled separately,
    once per query, from the ESCI query text itself)."""
    if not final.startswith(prefix.lower()):
        return DIVERGES
    if prefix.strip().lower() == final:
        return None
    ratio = len(prefix) / len(final)
    if ratio <= 0.25:
        return LE25
    if ratio <= 0.5:
        return G25_50
    if ratio <= 0.75:
        return G50_75
    return G75_100


def normalize_amazon_esci(
    raw: dict[str, Path], out: Path, corpus: Literal["full", "pool"] = "full"
) -> NormalizedDataset:
    # --- Products -> docs -------------------------------------------------------------
    products = [
        row
        for row in _read_table(
            raw["products"],
            [
                "product_id",
                "product_title",
                "product_description",
                "product_bullet_point",
                "product_brand",
                "product_color",
                "product_locale",
            ],
        )
        if row.get("product_locale") == "us"
    ]

    # --- Examples (judgments) ----------------------------------------------------------
    examples = [
        row
        for row in _read_table(
            raw["examples"],
            [
                "query_id",
                "query",
                "product_id",
                "product_locale",
                "esci_label",
                "small_version",
                "split",
            ],
        )
        if row.get("product_locale") == "us"
    ]

    queries_by_id: dict[str, dict[str, Any]] = {}
    judgments: dict[str, list[tuple[str, str]]] = {}
    pool_product_ids: set[str] = set()
    for row in examples:
        qid = str(row["query_id"])
        pid = str(row["product_id"])
        info = queries_by_id.setdefault(
            qid,
            {
                "text": row["query"],
                "small_version": row.get("small_version"),
                "split": row.get("split"),
                "labels": set(),
            },
        )
        info["labels"].add(row["esci_label"])
        judgments.setdefault(qid, []).append((pid, row["esci_label"]))
        if row.get("small_version") == 1 and row.get("split") == "test":
            pool_product_ids.add(pid)

    # --- Docs: title-drop, then corpus:full|pool ----------------------------------------
    kept_doc_ids: set[str] = set()
    dropped_docs = 0

    def docs() -> Iterator[dict[str, Any]]:
        nonlocal dropped_docs
        for row in products:
            doc = _product_doc(row)
            if doc["title"] is None:
                dropped_docs += 1
                continue
            if corpus == "pool" and doc["id"] not in pool_product_ids:
                continue
            kept_doc_ids.add(doc["id"])
            yield doc

    docs_path = out / "docs.jsonl"
    n_docs = write_jsonl(docs_path, docs())

    # --- Query pool: small_version==1, split==test, >=1 E/S, deduped by normalized text --
    candidates = [
        qid
        for qid, info in queries_by_id.items()
        if info["small_version"] == 1 and info["split"] == "test" and ({"E", "S"} & info["labels"])
    ]

    def qid_sort_key(qid: str) -> tuple[int, str]:
        return (0, qid.zfill(20)) if qid.isdigit() else (1, qid)

    best_by_key: dict[str, str] = {}
    for qid in sorted(candidates, key=qid_sort_key):
        key = _normalize_key(queries_by_id[qid]["text"])
        best_by_key.setdefault(key, qid)
    deduped = set(best_by_key.values())

    pool_keys = {_normalize_key(queries_by_id[qid]["text"]) for qid in deduped}

    # --- AmazonQAC: stream, keep sessions whose final term matches the pool -------------
    sessions_by_key: dict[str, list[list[str]]] = {}
    pf = pq.ParquetFile(raw["qac"])
    for batch in pf.iter_batches(columns=QAC_COLUMNS, batch_size=QAC_BATCH_SIZE):
        for row in batch.to_pylist():
            key = _normalize_key(row["final_search_term"] or "")
            if key in pool_keys:
                sessions_by_key.setdefault(key, []).append(list(row["prefixes"] or []))

    eligible = [qid for qid in deduped if sessions_by_key.get(_normalize_key(queries_by_id[qid]["text"]))]

    rng = random.Random(SEED)
    sorted_eligible = sorted(eligible, key=qid_sort_key)
    if len(sorted_eligible) > SAMPLE_SIZE:
        sampled = rng.sample(sorted_eligible, SAMPLE_SIZE)
    else:
        sampled = sorted_eligible
    sampled_sorted = sorted(sampled, key=qid_sort_key)

    sources = _read_sources(raw["sources"])

    # --- Main qrels for the sampled queries, filtered against kept docs -----------------
    dropped_judgments = 0
    main_qrels: Qrels = {}
    for qid in sampled_sorted:
        grades: dict[str, int] = {}
        for pid, label in judgments.get(qid, []):
            if pid not in kept_doc_ids:
                dropped_judgments += 1
                continue
            grades[pid] = GRADES[label]
        main_qrels[f"esci-{qid}"] = grades

    query_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    prefix_qrels: Qrels = {}

    for qid in sampled_sorted:
        final = _normalize_key(queries_by_id[qid]["text"])
        esci_qid = f"esci-{qid}"
        query_rows.append({"qid": esci_qid, "text": final, "group": sources.get(qid, "")})

        # complete, from the ESCI query text itself, once per query.
        complete_id = f"{esci_qid}-{COMPLETE}"
        prefix_rows.append({"qid": complete_id, "text": final, "group": COMPLETE})
        prefix_qrels[complete_id] = main_qrels.get(esci_qid, {})

        sessions = sessions_by_key.get(final, [])
        if not sessions:
            continue
        session = rng.choice(sessions)

        by_group: dict[str, list[str]] = {}
        for prefix in session:
            group = _group_for(prefix, final)
            if group is None:
                continue
            by_group.setdefault(group, []).append(prefix)

        for group in PREFIX_GROUPS:
            candidates_for_group = by_group.get(group)
            if not candidates_for_group:
                continue
            chosen = rng.choice(candidates_for_group)
            row_id = f"{esci_qid}-{group}"
            prefix_rows.append({"qid": row_id, "text": chosen, "group": group})
            prefix_qrels[row_id] = main_qrels.get(esci_qid, {})

    queries_path = out / "queries.jsonl"
    qrels_path = out / "qrels.tsv"
    n_queries = write_jsonl(queries_path, query_rows)
    write_qrels(qrels_path, main_qrels)

    prefixes_queries_path = out / "queries.prefixes.jsonl"
    prefixes_qrels_path = out / "qrels.prefixes.tsv"
    write_jsonl(prefixes_queries_path, prefix_rows)
    write_qrels(prefixes_qrels_path, prefix_qrels)

    return NormalizedDataset(
        docs_path,
        queries_path,
        qrels_path,
        n_docs,
        n_queries,
        dropped={"docs": dropped_docs, "judgments": dropped_judgments},
        query_sets={"prefixes": (prefixes_queries_path, prefixes_qrels_path)},
    )
