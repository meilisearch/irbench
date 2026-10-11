import pytest

from irbench.models import Filter, Sort
from irbench.score.constraints import (
    expected_facets,
    facet_accuracy,
    filter_violations,
    order_violations,
)

DOCS = {
    "a": {"product_class": ["Beds"], "average_rating": 4.5},
    "b": {"product_class": ["Sofas", "Sectionals"], "average_rating": 3.0},
    "c": {"product_class": ["Beds"], "average_rating": None},
    "d": {"product_class": [], "average_rating": 4.0},
}


def test_filter_violations_count_non_matching_and_unknown():
    f = Filter(field="product_class", op="eq", value="Beds")
    assert filter_violations(["a", "c"], f, DOCS) == 0
    assert filter_violations(["a", "b", "zzz"], f, DOCS) == 2


def test_order_violations_desc_and_asc():
    desc = Sort(field="average_rating", direction="desc")
    assert order_violations(["a", "d", "b"], desc, DOCS) == 0
    assert order_violations(["b", "a", "d"], desc, DOCS) == 1
    asc = Sort(field="average_rating", direction="asc")
    assert order_violations(["b", "d", "a"], asc, DOCS) == 0


def test_order_missing_values_must_be_last():
    desc = Sort(field="average_rating", direction="desc")
    assert order_violations(["a", "b", "c"], desc, DOCS) == 0
    assert order_violations(["c", "a", "b"], desc, DOCS) == 1


def test_expected_facets_counts_list_values():
    assert expected_facets(["a", "b", "c", "d"], "product_class", DOCS) == {
        "Beds": 2,
        "Sofas": 1,
        "Sectionals": 1,
    }


def test_facet_accuracy():
    expected = {"Beds": 2, "Sofas": 1}
    assert facet_accuracy({"Beds": 2, "Sofas": 1}, expected) == 1.0
    assert facet_accuracy({"Beds": 2, "Sofas": 3}, expected) == pytest.approx(0.5)
    assert facet_accuracy({"Beds": 2, "Sofas": 1, "Rugs": 1}, expected) == pytest.approx(2 / 3)
    assert facet_accuracy({}, {}) == 1.0
