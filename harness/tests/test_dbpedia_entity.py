import hashlib
import json
import random

import pytest

import irbench.datasets.dbpedia_entity as dbpedia_entity
from irbench.datasets.dbpedia_entity import SEED, entity_id, normalize_dbpedia_entity
from irbench.io import read_jsonl, read_qrels

ZURICH = "<dbpedia:Zürich>"
ENTITIES = [f"<dbpedia:E{i:02d}>" for i in range(30)] + [ZURICH]
QUERIES = [
    ("SemSearch_ES-1", "glenn frey"),
    ("INEX_LD-2", "vietnam war movie"),
    ("INEX_XER-3", "Airports in Germany"),
    ("SemSearch_LS-4", "Neil Gaiman novels"),
    ("TREC_Entity-5", "provinces of Canada"),
    ("QALD2_tr-6", "July, 1850  president  died"),
    ("QALD2_te-7", "Who designed the Brooklyn Bridge?"),
]
# Judged in test.tsv: E00-E05 (grades 0, 1, 2). Judged in dev.tsv only: E06 and Zürich.
TEST_QRELS = [
    ("SemSearch_ES-1", "<dbpedia:E00>", 0),
    ("SemSearch_ES-1", "<dbpedia:E01>", 1),
    ("INEX_LD-2", "<dbpedia:E02>", 2),
    ("INEX_XER-3", "<dbpedia:E03>", 1),
    ("SemSearch_LS-4", "<dbpedia:E04>", 0),
    ("TREC_Entity-5", "<dbpedia:E05>", 2),
]
DEV_QRELS = [("QALD2_te-7", "<dbpedia:E06>", 1), ("QALD2_te-7", ZURICH, 2)]
JUDGED = {uri for _, uri, _ in TEST_QRELS + DEV_QRELS}


def _qrels_text(rows):
    return "query-id\tcorpus-id\tscore\n" + "".join(f"{q}\t{d}\t{g}\n" for q, d, g in rows)


def make_raw(tmp_path, queries=QUERIES, test_qrels=TEST_QRELS, dev_qrels=DEV_QRELS):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    corpus = raw_dir / "corpus.jsonl"
    corpus.write_text(
        "".join(
            json.dumps({"_id": uri, "title": f"Title {uri}", "text": f"Abstract of {uri}."}) + "\n"
            for uri in ENTITIES
        )
    )
    qpath = raw_dir / "queries.jsonl"
    qpath.write_text("".join(json.dumps({"_id": q, "text": t}) + "\n" for q, t in queries))
    test = raw_dir / "test.tsv"
    test.write_text(_qrels_text(test_qrels))
    dev = raw_dir / "dev.tsv"
    dev.write_text(_qrels_text(dev_qrels))
    return {"corpus": corpus, "queries": qpath, "qrels_test": test, "qrels_dev": dev}


def run(tmp_path, corpus_size, **raw_overrides):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return normalize_dbpedia_entity(make_raw(tmp_path, **raw_overrides), out, corpus_size)


def test_keeps_every_judged_entity_and_fills_to_size(tmp_path):
    result = run(tmp_path, 12)
    docs = list(read_jsonl(result.docs))
    assert len(docs) == result.n_docs == 12
    assert {entity_id(u) for u in JUDGED} <= {d["id"] for d in docs}
    assert len(JUDGED) == 8
    assert result.stats == {
        "corpus_size": 12,
        "corpus_seed": 20261031,
        "judged_docs": 8,
        "distractor_docs": 4,
        "source_docs": 31,
    }


def test_sample_is_seeded_and_in_file_order(tmp_path):
    first = run(tmp_path, 12)
    a = first.docs.read_bytes()
    assert run(tmp_path, 12).docs.read_bytes() == a

    docs = list(read_jsonl(first.docs))
    positions = [ENTITIES.index(d["uri"]) for d in docs]
    assert positions == sorted(positions)

    unjudged = [u for u in ENTITIES if u not in JUDGED]
    assert len(unjudged) == 23
    expected = {unjudged[i] for i in random.Random(SEED).sample(range(23), 4)}
    assert {d["uri"] for d in docs if d["uri"] not in JUDGED} == expected


def test_dev_only_judgment_is_judged(tmp_path):
    dev_only = "<dbpedia:E06>"
    unjudged = [u for u in ENTITIES if u not in JUDGED]
    distractors = {unjudged[i] for i in random.Random(SEED).sample(range(23), 4)}
    for _ in range(2):
        docs = list(read_jsonl(run(tmp_path, 12).docs))
        uris = [d["uri"] for d in docs]
        assert dev_only in uris
        assert dev_only not in distractors
        assert {u for u in uris if u not in JUDGED} == distractors


def test_ids_hash_the_uri(tmp_path):
    assert entity_id(ZURICH) == "dbp-" + hashlib.sha1(ZURICH.encode()).hexdigest()[:16]
    result = run(tmp_path, 12)
    doc = next(d for d in read_jsonl(result.docs) if d["uri"] == ZURICH)
    assert doc["id"] == entity_id(ZURICH)
    assert set(doc) == {"id", "title", "text", "uri"}
    assert doc["id"] in read_qrels(result.qrels)["QALD2_te-7"]


def test_id_collision_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(dbpedia_entity, "entity_id", lambda uri: "dbp-same")
    with pytest.raises(ValueError, match="collide"):
        run(tmp_path, 12)


def test_groups_and_text(tmp_path):
    result = run(tmp_path, 12)
    rows = list(read_jsonl(result.queries))
    assert result.n_queries == len(rows) == 7
    assert [r["group"] for r in rows] == [
        "SemSearch ES",
        "INEX-LD",
        "ListSearch",
        "ListSearch",
        "ListSearch",
        "QALD2",
        "QALD2",
    ]
    assert [r["qid"] for r in rows] == [q for q, _ in QUERIES]
    assert rows[5]["text"] == "July, 1850 president died"


def test_unknown_group_prefix_fails(tmp_path):
    with pytest.raises(ValueError, match="Foo_Bar"):
        run(tmp_path, 12, queries=[*QUERIES, ("Foo_Bar-1", "x")])


def test_qrels_merge_test_and_dev(tmp_path):
    result = run(tmp_path, 12)
    expected: dict[str, dict[str, int]] = {}
    for q, uri, g in TEST_QRELS + DEV_QRELS:
        expected.setdefault(q, {})[entity_id(uri)] = g
    assert read_qrels(result.qrels) == expected
    assert {g for grades in expected.values() for g in grades.values()} == {0, 1, 2}


def test_missing_judged_entity_fails(tmp_path):
    with pytest.raises(ValueError, match="<dbpedia:Absent>"):
        run(tmp_path, 12, test_qrels=[*TEST_QRELS, ("INEX_LD-2", "<dbpedia:Absent>", 1)])


def test_size_below_judged_fails(tmp_path):
    with pytest.raises(ValueError, match=r"(?s)(?=.*\b5\b)(?=.*\b8\b)"):
        run(tmp_path, 5)


def test_size_bounds(tmp_path):
    result = run(tmp_path, 31)
    assert {d["uri"] for d in read_jsonl(result.docs)} == set(ENTITIES)
    assert result.stats["distractor_docs"] == 23
    with pytest.raises(ValueError, match="32"):
        run(tmp_path, 32)
