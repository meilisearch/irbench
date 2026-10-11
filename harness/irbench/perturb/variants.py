import random

from irbench.models import Query
from irbench.perturb.prefix import prefix
from irbench.perturb.synonym import synonym
from irbench.perturb.typo import typo


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _apply(kind: str, text: str, rng: random.Random, vocabulary: list[list[str]]) -> str | None:
    if kind == "typo":
        return typo(text, rng)
    if kind == "prefix":
        return prefix(text, rng)
    if kind == "synonym":
        return synonym(text, rng, vocabulary)
    raise ValueError(f"unknown perturbation kind {kind!r}")


def make_variants(
    queries: list[Query], seed: int, kinds: list[str], vocabulary: list[list[str]]
) -> list[Query]:
    out: list[Query] = []
    for query in queries:
        out.append(query)
        for kind in kinds:
            rng = random.Random(f"{seed}:{query.qid}:{kind}")
            text = _apply(kind, query.text, rng, vocabulary)
            if text is None or _norm(text) == _norm(query.text):
                continue
            out.append(
                Query(qid=f"{query.qid}-{kind}", text=text, variant_of=query.qid, perturbation=kind)
            )
    return out
