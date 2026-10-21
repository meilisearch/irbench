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
