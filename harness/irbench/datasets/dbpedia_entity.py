"""`dbpedia-entity`: DBpedia-Entity v2 (the `mteb/dbpedia` copy), cut to a seeded sample of the
corpus: every judged entity plus uniform random distractors (spec
2026-10-31-irbench-dbpedia-entity-design §3.3), written through the BEIR helpers with hashed
document ids and the paper's four query groups (§3.4)."""

import hashlib
import random
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from irbench.datasets.beir import read_jsonl_rows, read_qrels_tsv, write_normalized
from irbench.datasets.wands import NormalizedDataset
from irbench.models import Qrels

# Loader constants: part of the dataset's definition, not a run setting (spec §3.3).
SEED = 20261031

# Query `_id` prefix (text before the last `-`) -> the paper's query group (spec §3.2).
GROUPS = {
    "SemSearch_ES": "SemSearch ES",
    "INEX_LD": "INEX-LD",
    "INEX_XER": "ListSearch",
    "SemSearch_LS": "ListSearch",
    "TREC_Entity": "ListSearch",
    "QALD2_tr": "QALD2",
    "QALD2_te": "QALD2",
}


def entity_id(uri: str) -> str:
    """Document id: a hash of the DBpedia URI, which is not a safe id (`<dbpedia:Zürich>`)."""
    return "dbp-" + hashlib.sha1(uri.encode("utf-8")).hexdigest()[:16]


def _query_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for r in read_jsonl_rows(path):
        prefix = r["_id"].rsplit("-", 1)[0]
        if prefix not in GROUPS:
            raise ValueError(f"query {r['_id']!r}: unknown group prefix {prefix!r}")
        rows.append({"qid": r["_id"], "text": " ".join(r["text"].split()), "group": GROUPS[prefix]})
    return rows


def normalize_dbpedia_entity(
    raw: dict[str, Path], out: Path, corpus_size: int = 1_000_000
) -> NormalizedDataset:
    queries = _query_rows(raw["queries"])
    uri_qrels: Qrels = {}
    for key in ("qrels_test", "qrels_dev"):
        for qid, grades in read_qrels_tsv(raw[key]).items():
            uri_qrels.setdefault(qid, {}).update(grades)
    judged = {uri for grades in uri_qrels.values() for uri in grades}

    # Pass 1: size the corpus and check that every judged entity is in it.
    source_docs = unjudged = 0
    seen: set[str] = set()
    for row in read_jsonl_rows(raw["corpus"]):
        source_docs += 1
        if row["_id"] in judged:
            seen.add(row["_id"])
        else:
            unjudged += 1
    if missing := judged - seen:
        raise ValueError(
            f"{len(missing)} judged entities are not in the corpus, e.g. {min(missing)!r}"
        )
    if corpus_size < len(judged):
        raise ValueError(
            f"corpus_size {corpus_size} is below the {len(judged)} judged entities, "
            "which must all be kept"
        )
    if corpus_size > source_docs:
        raise ValueError(f"corpus_size {corpus_size} is above the {source_docs} entities available")

    n_distractors = corpus_size - len(judged)
    chosen = set(random.Random(SEED).sample(range(unjudged), n_distractors))

    # Pass 2: stream the sample out in corpus file order; only the kept ids are held in memory.
    uri_by_id: dict[str, str] = {}

    def docs() -> Iterator[dict[str, Any]]:
        number = 0
        for row in read_jsonl_rows(raw["corpus"]):
            uri = row["_id"]
            if uri not in judged:
                number += 1
                if number - 1 not in chosen:
                    continue
            doc_id = entity_id(uri)
            if doc_id in uri_by_id:
                raise ValueError(
                    f"ids collide: {uri!r} and {uri_by_id[doc_id]!r} both hash to {doc_id}"
                )
            uri_by_id[doc_id] = uri
            yield {"id": doc_id, "title": row["title"], "text": row["text"], "uri": uri}

    qrels: Qrels = {
        qid: {entity_id(uri): grade for uri, grade in grades.items()}
        for qid, grades in uri_qrels.items()
    }
    result = write_normalized(out, docs(), queries, qrels, ["title", "text"])
    return replace(
        result,
        stats={
            "corpus_size": corpus_size,
            "corpus_seed": SEED,
            "judged_docs": len(judged),
            "distractor_docs": n_distractors,
            "source_docs": source_docs,
        },
    )
