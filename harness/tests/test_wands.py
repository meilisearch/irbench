import pytest

from irbench.datasets import NORMALIZERS
from irbench.io import read_jsonl, read_qrels


def test_normalize_wands_docs(wands_raw, tmp_path):
    result = NORMALIZERS["wands"](wands_raw, tmp_path / "out")
    docs = {d["id"]: d for d in read_jsonl(result.docs)}
    assert result.n_docs == 4
    assert docs["0"] == {
        "id": "0",
        "name": "solid wood platform bed",
        "product_class": ["Beds"],
        "category": ["Furniture", "Bedroom Furniture", "Beds"],
        "description": 'a "sturdy" bed',
        "features": ["color : brown", "material : wood"],
        "rating_count": 15.0,
        "average_rating": 4.5,
        "review_count": 15.0,
    }
    assert docs["1"]["product_class"] == ["Sofas", "Sectionals"]
    assert docs["1"]["average_rating"] is None
    assert docs["3"]["features"] == []


def test_normalize_wands_queries_and_qrels(wands_raw, tmp_path):
    result = NORMALIZERS["wands"](wands_raw, tmp_path / "out")
    queries = list(read_jsonl(result.queries))
    assert queries[0] == {"qid": "0", "text": "platform bed", "query_class": "Beds"}
    assert result.n_queries == 3
    assert read_qrels(result.qrels) == {
        "0": {"0": 2, "3": 1},
        "1": {"1": 2, "2": 0},
        "2": {"2": 0},
    }


def test_unknown_label_is_an_error(wands_raw, tmp_path):
    wands_raw["label"].write_text("id\tquery_id\tproduct_id\tlabel\n0\t0\t0\tMaybe\n")
    with pytest.raises(ValueError, match="Maybe"):
        NORMALIZERS["wands"](wands_raw, tmp_path / "out")
