"""cmteb-ecom / cmteb-video loader (spec 2026-10-31-irbench-chinese-title-search §3.3)."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from irbench.datasets.cmteb import SAMPLE_SIZE, normalize_cmteb
from irbench.io import read_jsonl, read_qrels


def _raw(
    tmp_path: Path,
    corpus: dict[str, str],
    dev: dict[str, str],
    dev_qrels: dict[str, str],
    train: dict[str, str],
    train_qrels: dict[str, str],
) -> dict[str, Path]:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    pq.write_table(
        pa.table({"id": list(corpus), "text": list(corpus.values())}), raw / "corpus.parquet"
    )
    pq.write_table(pa.table({"id": list(dev), "text": list(dev.values())}), raw / "q.parquet")
    pq.write_table(
        pa.table(
            {
                "qid": list(dev_qrels),
                "pid": list(dev_qrels.values()),
                "score": [1] * len(dev_qrels),
            }
        ),
        raw / "qrels.parquet",
    )
    (raw / "train.query.txt").write_text(
        "".join(f"{q}\t{t}\n" for q, t in train.items()), encoding="utf-8"
    )
    (raw / "qrels.train.tsv").write_text(
        "".join(f"{q}\t0\t{p}\t1\n" for q, p in train_qrels.items()), encoding="utf-8"
    )
    return {
        "corpus": raw / "corpus.parquet",
        "queries": raw / "q.parquet",
        "qrels": raw / "qrels.parquet",
        "train_queries": raw / "train.query.txt",
        "train_qrels": raw / "qrels.train.tsv",
    }


CORPUS = {"1": "纯山楂酱", "2": "工作站机箱", "3": "老肉片"}


def _run(tmp_path, corpus=CORPUS, dev=None, dev_qrels=None, train=None, train_qrels=None):
    raw = _raw(
        tmp_path,
        corpus,
        dev if dev is not None else {"200000": "山楂酱", "200001": "机箱", "200002": "老肉"},
        dev_qrels if dev_qrels is not None else {"200000": "1", "200001": "2", "200002": "3"},
        train or {},
        train_qrels or {},
    )
    out = tmp_path / "out"
    return normalize_cmteb(raw, out, domain="ecom")


def test_dev_queries_all_kept(tmp_path):
    ds = _run(tmp_path)
    rows = list(read_jsonl(ds.queries))
    assert [r["qid"] for r in rows] == ["ecom-dev-200000", "ecom-dev-200001", "ecom-dev-200002"]
    assert {r["group"] for r in rows} == {"dev"}
    assert read_qrels(ds.qrels)["ecom-dev-200001"] == {"2": 1}
    assert [d for d in read_jsonl(ds.docs)][0] == {"id": "1", "title": "纯山楂酱"}


def test_train_sample_only_eligible_and_stable(tmp_path):
    train = {"7": "山楂", "8": "机箱", "9": "不在库"}
    train_qrels = {"7": "1", "8": "2", "9": "999"}
    a = _run(tmp_path / "a", train=train, train_qrels=train_qrels)
    b = _run(tmp_path / "b", train=train, train_qrels=train_qrels)
    rows_a = [r for r in read_jsonl(a.queries) if r["group"] == "train"]
    rows_b = [r for r in read_jsonl(b.queries) if r["group"] == "train"]
    assert rows_a == rows_b
    assert sorted(r["qid"] for r in rows_a) == ["ecom-train-7", "ecom-train-8"]
    assert read_qrels(a.qrels)["ecom-train-7"] == {"1": 1}


def test_train_sample_is_capped(tmp_path):
    corpus = {str(i): f"标题{i}" for i in range(SAMPLE_SIZE + 50)}
    train = {str(10_000 + i): f"查询{i}" for i in range(SAMPLE_SIZE + 50)}
    train_qrels = {str(10_000 + i): str(i) for i in range(SAMPLE_SIZE + 50)}
    ds = _run(tmp_path, corpus=corpus, dev={}, dev_qrels={}, train=train, train_qrels=train_qrels)
    assert ds.n_queries == SAMPLE_SIZE


def test_duplicate_merge_keeps_one_judgment(tmp_path):
    corpus = {"1": "甲", "5": "甲", "6": "乙"}
    dev = {"200000": "甲", "200001": "甲乙"}
    dev_qrels = {"200000": "5", "200001": "1"}
    ds = _run(tmp_path, corpus=corpus, dev=dev, dev_qrels=dev_qrels)
    assert [d["id"] for d in read_jsonl(ds.docs)] == ["1", "6"]
    qrels = read_qrels(ds.qrels)
    assert qrels["ecom-dev-200000"] == {"1": 1}
    assert qrels["ecom-dev-200001"] == {"1": 1}
    assert ds.dropped["docs"] == 1
    assert ds.dropped["moved_judgments"] == 1


def test_text_kept_as_written(tmp_path):
    corpus = {"1": "ＡＢＣ小米WiFi"}
    ds = _run(tmp_path, corpus=corpus, dev={"200000": " 老肉， "}, dev_qrels={"200000": "1"})
    assert next(iter(read_jsonl(ds.docs)))["title"] == "ＡＢＣ小米WiFi"
    assert next(iter(read_jsonl(ds.queries)))["text"] == "老肉，"
