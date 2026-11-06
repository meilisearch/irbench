from irbench.models import Query
from irbench.perturb.variants import make_variants

VOCAB = [["sofa", "couch"]]


def _q(qid: str, text: str) -> Query:
    return Query(qid=qid, text=text, variant_of=qid, perturbation="original")


def test_variants_are_linked_to_source():
    out = make_variants([_q("7", "velvet sofa")], 1, ["typo", "prefix", "synonym"], VOCAB)
    assert [q.qid for q in out] == ["7", "7-typo", "7-prefix", "7-synonym"]
    assert all(q.variant_of == "7" for q in out)
    assert out[3].text == "velvet couch"


def test_variants_are_byte_identical_across_runs():
    queries = [_q(str(i), t) for i, t in enumerate(["platform bed", "grey velvet sofa", "rug"])]
    a = make_variants(queries, 42, ["typo", "prefix", "synonym"], VOCAB)
    b = make_variants(queries, 42, ["typo", "prefix", "synonym"], VOCAB)
    assert [q.model_dump() for q in a] == [q.model_dump() for q in b]


def test_inapplicable_perturbations_produce_no_variant():
    out = make_variants([_q("1", "tv")], 1, ["typo", "prefix", "synonym"], VOCAB)
    assert [q.qid for q in out] == ["1"]


def test_seed_changes_output():
    q = [_q("1", "upholstered platform nightstand")]
    texts = {make_variants(q, s, ["typo"], VOCAB)[1].text for s in range(20)}
    assert len(texts) > 1


def test_variants_keep_the_query_group():
    q = Query(
        qid="1", text="coronavirus origin", variant_of="1", perturbation="original", group="keyword"
    )
    variants = make_variants([q], 7, ["typo", "prefix"], [])
    assert len(variants) > 1
    assert {v.group for v in variants} == {"keyword"}


def test_corrected_variant():
    q = Query(
        qid="9",
        text="kohlr toilt seat",
        variant_of="9",
        perturbation="original",
        group="misspelled",
        corrected="kohler toilet seat",
    )
    out = make_variants([q], 1, ["corrected"], [])
    assert [v.qid for v in out] == ["9", "9-corrected"]
    v = out[1]
    assert v.text == "kohler toilet seat"
    assert v.perturbation == "corrected"
    assert v.group == "misspelled"
    assert v.variant_of == "9"
    assert v.corrected is None


def test_corrected_absent():
    out = make_variants([_q("9", "kohler toilet seat")], 1, ["corrected"], [])
    assert [v.qid for v in out] == ["9"]


def test_corrected_equal_after_norm_dropped():
    q = Query(
        qid="9",
        text="Kohler  toilet seat",
        variant_of="9",
        perturbation="original",
        corrected=" kohler toilet  SEAT ",
    )
    assert [v.qid for v in make_variants([q], 1, ["corrected"], [])] == ["9"]


def test_other_kinds_unchanged():
    # Pinned from the output of the code before `corrected` existed.
    texts = ["platform bed", "grey velvet sofa", "upholstered platform nightstand", "rug"]
    queries = [_q(str(i), t) for i, t in enumerate(texts)]
    vocab = [["sofa", "couch"], ["bed", "cot"]]
    out = make_variants(queries, 42, ["typo", "prefix", "synonym"], vocab)
    assert [(v.qid, v.text) for v in out] == [
        ("0", "platform bed"),
        ("0-typo", "latofrm bed"),
        ("0-prefix", "platform be"),
        ("0-synonym", "platform cot"),
        ("1", "grey velvet sofa"),
        ("1-typo", "grey velvet ofa"),
        ("1-prefix", "grey velvet so"),
        ("1-synonym", "grey velvet couch"),
        ("2", "upholstered platform nightstand"),
        ("2-typo", "uphoksttered platform nightstand"),
        ("2-prefix", "upholstered platform nightst"),
        ("3", "rug"),
        ("3-prefix", "ru"),
    ]
    # Adding `corrected` to the kinds must not move the other kinds' outputs.
    with_corrected = make_variants(queries, 42, ["typo", "corrected", "prefix", "synonym"], vocab)
    assert [(v.qid, v.text) for v in with_corrected] == [(v.qid, v.text) for v in out]
