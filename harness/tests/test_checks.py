import pytest

from irbench.engines.base import SearchResult
from irbench.models import Filter, QueryTask, Sort
from irbench.score.checks import TaskCheck, check_task, summarize_checks

DOCS = {
    "a": {"product_class": ["Beds"], "average_rating": 4.5},
    "b": {"product_class": ["Sofas"], "average_rating": 3.0},
    "c": {"product_class": ["Beds"], "average_rating": None},
}


def _task(mode, **extra):
    return QueryTask(
        task_id=f"1:{mode}:keyword",
        qid="1",
        variant_of="1",
        perturbation="original",
        text="x",
        mode=mode,
        base="keyword",
        **extra,
    )


def test_plain_task_has_no_checks():
    c = check_task(_task("keyword"), SearchResult(["a"]), None, DOCS, 10)
    assert (c.filter_violations, c.order_violations, c.facet_accuracy, c.facet_verifiable) == (
        None,
        None,
        None,
        None,
    )


def test_filter_and_order_violations():
    f = _task("filtered", filter=Filter(field="product_class", op="eq", value="Beds"))
    assert check_task(f, SearchResult(["a", "b", "zzz"]), None, DOCS, 10).filter_violations == 2
    s = _task("sorted", sort=Sort(field="average_rating", direction="desc"))
    assert check_task(s, SearchResult(["a", "b", "c"]), None, DOCS, 10).order_violations == 0
    assert check_task(s, SearchResult(["c", "a"]), None, DOCS, 10).order_violations == 1


def test_facet_accuracy_against_verification_fetch():
    t = _task("faceted", facets=["product_class"])
    good = SearchResult(["a"], facets={"product_class": {"Beds": 2, "Sofas": 1}})
    c = check_task(t, good, ["a", "b", "c"], DOCS, 10)
    assert (c.facet_verifiable, c.facet_accuracy) == (True, 1.0)
    wrong = SearchResult(["a"], facets={"product_class": {"Beds": 5, "Sofas": 1}})
    assert check_task(t, wrong, ["a", "b", "c"], DOCS, 10).facet_accuracy == pytest.approx(0.5)
    missing = SearchResult(["a"], facets=None)
    assert check_task(t, missing, ["a", "b", "c"], DOCS, 10).facet_accuracy == 0.0


def test_facet_check_at_limit_is_unverifiable():
    t = _task("faceted", facets=["product_class"])
    c = check_task(t, SearchResult(["a"], facets={}), ["a", "b", "c"], DOCS, verify_limit=3)
    assert (c.facet_verifiable, c.facet_accuracy) == (False, None)
    failed_fetch = check_task(t, SearchResult(["a"], facets={}), None, DOCS, 10)
    assert failed_fetch.facet_verifiable is False


def test_duplicate_check_ids_count_once():
    t = _task("faceted", facets=["product_class"])
    result = SearchResult(["a"], facets={"product_class": {"Beds": 1}})
    assert check_task(t, result, ["a", "a", "a"], DOCS, 10).facet_accuracy == 1.0


def test_summaries_by_mode_and_base():
    def c(tid, mode, **kw):
        return TaskCheck(task_id=tid, mode=mode, base="keyword", **kw)

    checks = [
        c("1", "filtered", filter_violations=0),
        c("2", "filtered", filter_violations=3),
        c("3", "filtered", error="boom"),
        c("4", "faceted", facet_verifiable=True, facet_accuracy=1.0),
        c("5", "faceted", facet_verifiable=True, facet_accuracy=0.5),
        c("6", "faceted", facet_verifiable=False),
        c("7", "keyword"),
    ]
    by_key = {(s.mode, s.base): s for s in summarize_checks(checks)}
    filtered = by_key[("filtered", "keyword")]
    assert (filtered.n, filtered.errors, filtered.filter_violations) == (3, 1, 3)
    assert filtered.order_violations is None
    faceted = by_key[("faceted", "keyword")]
    assert (faceted.facet_verified, faceted.facet_unverifiable) == (2, 1)
    assert faceted.facet_accuracy == pytest.approx(0.75)
    plain = by_key[("keyword", "keyword")]
    assert (plain.filter_violations, plain.facet_accuracy) == (None, None)


def test_failed_check_fetches_are_counted_apart_from_unverifiable():
    def c(tid, **kw):
        return TaskCheck(task_id=tid, mode="faceted", base="keyword", **kw)

    checks = [
        c("1", facet_verifiable=False, check_error="EngineError: 500"),
        c("2", facet_verifiable=False),
        c("3", facet_verifiable=True, facet_accuracy=1.0),
    ]
    (s,) = summarize_checks(checks)
    assert (s.facet_check_errors, s.facet_unverifiable, s.facet_verified) == (1, 1, 1)
