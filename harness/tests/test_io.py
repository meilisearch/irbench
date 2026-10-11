import pytest

from irbench.io import read_jsonl, read_qrels, read_run, write_jsonl, write_qrels, write_run
from irbench.models import Filter, QueryTask


def test_jsonl_roundtrip_keeps_unicode(tmp_path):
    rows = [{"id": "1", "name": "café table"}, {"id": "2", "name": "sofa"}]
    assert write_jsonl(tmp_path / "a" / "x.jsonl", rows) == 2
    assert list(read_jsonl(tmp_path / "a" / "x.jsonl")) == rows
    assert "café" in (tmp_path / "a" / "x.jsonl").read_text(encoding="utf-8")


def test_qrels_roundtrip_trec_format(tmp_path):
    qrels = {"q2": {"d9": 0}, "q1": {"d2": 1, "d1": 2}}
    write_qrels(tmp_path / "q.tsv", qrels)
    assert (tmp_path / "q.tsv").read_text().splitlines() == [
        "q1 0 d1 2",
        "q1 0 d2 1",
        "q2 0 d9 0",
    ]
    assert read_qrels(tmp_path / "q.tsv") == qrels


def test_run_roundtrip_preserves_rank_order(tmp_path):
    results = {"t1": ["d3", "d1", "d2"], "t0": ["d5"]}
    write_run(tmp_path / "r.trec", results, tag="meili")
    lines = (tmp_path / "r.trec").read_text().splitlines()
    assert lines[0] == "t0 Q0 d5 1 1 meili"
    assert lines[1] == "t1 Q0 d3 1 3 meili"
    assert read_run(tmp_path / "r.trec") == results


def test_ids_with_whitespace_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="whitespace"):
        write_qrels(tmp_path / "q.tsv", {"q 1": {"d1": 1}})
    with pytest.raises(ValueError, match="whitespace"):
        write_run(tmp_path / "r.trec", {"t1": ["d 1"]}, tag="x")


def test_query_task_json_roundtrip():
    task = QueryTask(
        task_id="1-typo:filtered:keyword",
        qid="1-typo",
        variant_of="1",
        perturbation="typo",
        text="platfrom bed",
        mode="filtered",
        base="keyword",
        filter=Filter(field="average_rating", op="gte", value=4.0),
    )
    assert QueryTask.model_validate(task.model_dump(mode="json")) == task


def test_filter_keeps_string_values_as_strings():
    assert Filter(field="product_class", op="eq", value="Beds").value == "Beds"
