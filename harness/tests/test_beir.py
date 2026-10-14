import json

from irbench.datasets.beir import read_qrels_tsv
from irbench.datasets.cqadupstack import FORUMS, normalize_cqadupstack
from irbench.datasets.lotte import normalize_lotte
from irbench.io import read_jsonl, read_qrels


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _cqa_raw(tmp_path):
    raw = {}
    for forum in FORUMS:
        d = tmp_path / forum
        d.mkdir()
        raw[f"{forum}_corpus"] = _jsonl(
            d / "corpus.jsonl",
            [
                {
                    "_id": "1",
                    "title": f"{forum} title",
                    "text": "body",
                    "metadata": {"tags": ["php", "seo"]},
                },
                {"_id": "2", "title": "", "text": "", "metadata": {"tags": []}},
            ],
        )
        raw[f"{forum}_queries"] = _jsonl(
            d / "queries.jsonl",
            [
                {
                    "_id": "9",
                    "text": "how do I x?",
                    "metadata": {"tags": ["x"], "123": {"body": "huge"}},
                },
                {"_id": "8", "text": "only empty", "metadata": {}},
            ],
        )
        (d / "test.tsv").write_text("query-id\tcorpus-id\tscore\n9\t1\t1\n8\t2\t1\n")
        raw[f"{forum}_qrels"] = d / "test.tsv"
    return raw


def test_cqadupstack_prefixes_ids_and_keeps_forum_and_tags(tmp_path):
    out = normalize_cqadupstack(_cqa_raw(tmp_path), tmp_path / "out")
    docs = list(read_jsonl(out.docs))
    assert {d["id"] for d in docs} == {f"{f}-1" for f in FORUMS}
    doc = next(d for d in docs if d["id"] == "webmasters-1")
    assert doc == {
        "id": "webmasters-1",
        "title": "webmasters title",
        "text": "body",
        "forum": "webmasters",
        "tags": ["php", "seo"],
    }
    queries = list(read_jsonl(out.queries))
    assert {q["qid"] for q in queries} == {f"{f}-{i}" for f in FORUMS for i in ("9", "8")}
    assert all(set(q) == {"qid", "text"} for q in queries)  # bulky metadata is not carried
    qrels = read_qrels(out.qrels)
    assert qrels["android-9"] == {"android-1": 1}
    assert "android-8" not in qrels  # its only judgment pointed at an empty doc
    assert out.dropped == {"docs": 3, "judgments": 3}


def test_cqadupstack_keeps_only_queries_in_test_qrels(tmp_path):
    raw = _cqa_raw(tmp_path)
    _jsonl(
        raw["android_queries"],
        [{"_id": "9", "text": "kept"}, {"_id": "7", "text": "not in the test split"}],
    )
    out = normalize_cqadupstack(raw, tmp_path / "out")
    android = {q["qid"] for q in read_jsonl(out.queries) if q["qid"].startswith("android-")}
    assert android == {"android-9"}


def test_lotte_reads_answers_and_groups_by_forum(tmp_path):
    coll = _jsonl(
        tmp_path / "c.jsonl",
        [
            {"doc_id": 0, "author": "a", "text": "Use a cast iron pan."},
            {"doc_id": 1, "author": "b", "text": "Visas are needed."},
            {"doc_id": 2, "author": "c", "text": "  "},
        ],
    )
    qas = _jsonl(
        tmp_path / "q.jsonl",
        [
            {
                "qid": 0,
                "query": "best pan for steak?",
                "url": "https://cooking.stackexchange.com/questions/1/x",
                "answer_pids": [0, 2],
                "question_author": "",
            },
            {
                "qid": 1,
                "query": "do I need a visa?",
                "url": "https://travel.stackexchange.com/q/2",
                "answer_pids": [1],
                "question_author": "",
            },
        ],
    )
    out = normalize_lotte({"collection": coll, "qas": qas}, tmp_path / "out")
    assert list(read_jsonl(out.docs))[0] == {"id": "0", "text": "Use a cast iron pan."}
    assert [(q["qid"], q["group"]) for q in read_jsonl(out.queries)] == [
        ("0", "cooking"),
        ("1", "travel"),
    ]
    assert read_qrels(out.qrels) == {"0": {"0": 1}, "1": {"1": 1}}
    assert out.dropped == {"docs": 1, "judgments": 1}


def test_read_qrels_tsv_skips_header_and_clamps_negative_grades(tmp_path):
    path = tmp_path / "test.tsv"
    path.write_text("query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t-1\n")
    assert read_qrels_tsv(path) == {"q1": {"d1": 2, "d2": 0}}
