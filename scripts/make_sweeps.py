"""Write engines/<engine>/sweeps/<scenario>.yaml for every engine and scenario.

Each sweep file lists variants merged over the engine's tuned track (run as `tuned+<variant>`,
on the tune half of the queries only; see irbench.score.halves). The same budget applies to
every engine: `base` (the tuned track as committed) plus at most MAX_VARIANTS alternatives.

Design, one factor at a time:
- index variants: language analysis (stop words, stemming) where the engine ships it and it is
  off in the tuned track, BM25 parameters, typo thresholds, the ranking rule order, the
  hybrid score distribution;
- search variants: a keyword alternative and a hybrid alternative in the same run. Keyword and
  hybrid requests are scored separately, so one run measures one alternative of each.

Selection rule (applied by hand from `irbench sweep-summary`, recorded in each config's
justification): for each retrieval base, keep the variant with the best tune-half nDCG@10 over
all query variants; an index setting must not lower the other base by more than its gain;
a variant within 0.003 of the best that is closer to the engine's defaults wins the tie.

Run: uv run python scripts/make_sweeps.py
"""

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = (
    "ecommerce/product-search",
    "knowledge-base/community-qa",
    "knowledge-base/covid-papers",
    "knowledge-base/covid-papers-abstracts",
    "knowledge-base/web-questions",
)
MAX_VARIANTS = 7
LANG_BOTH = {"stop_words": "english", "stemming": "english"}

# Meilisearch semantic score ((1 + cosine) / 2) of the top 20 documents for the tune-half
# original queries, bge-small-en-v1.5 (scripts/meili_distribution.py).
MEILI_DISTRIBUTION = {
    "ecommerce/product-search": {"mean": 0.876, "sigma": 0.034},
    "knowledge-base/community-qa": {"mean": 0.877, "sigma": 0.024},
    "knowledge-base/covid-papers": {"mean": 0.911, "sigma": 0.016},
    "knowledge-base/covid-papers-abstracts": {"mean": 0.908, "sigma": 0.013},
    "knowledge-base/web-questions": {"mean": 0.868, "sigma": 0.029},
}


def tuned(engine: str, scenario: str) -> dict[str, Any]:
    path = ROOT / "engines" / engine / "configs" / f"{scenario}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")).get("tuned") or {}


def search(t: dict[str, Any], base: str) -> dict[str, Any]:
    return ((t.get("search") or {}).get(base)) or {}


def pair(keyword: dict[str, Any] | None, hybrid: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if keyword:
        out["keyword"] = keyword
    if hybrid:
        out["hybrid"] = hybrid
    return {"search": out}


def elasticsearch(s: str) -> dict[str, Any]:
    t = tuned("elasticsearch", s)
    kw = search(t, "keyword").get("multi_match", {})
    hy = search(t, "hybrid")
    fuzzy = kw.get("fuzziness") == "AUTO" or hy.get("multi_match", {}).get("fuzziness") == "AUTO"
    ktype = kw.get("type", "best_fields")
    # cross_fields rejects fuzziness, so it is only an alternative when fuzziness is off.
    types = ["best_fields", "most_fields"] + ([] if fuzzy else ["cross_fields"])
    alts = [x for x in types if x != ktype]
    fusion = hy.get("fusion") or {}
    fusions = [
        f
        for f in (
            {},
            {"weights": [1, 2]},
            {"weights": [2, 1]},
            {"method": "linear", "weights": [1, 1.5]},
        )
        if f != fusion
    ]
    toggled = "0" if fuzzy else "AUTO"
    if not fuzzy:
        toggle_type = "most_fields" if ktype == "cross_fields" else ktype
    else:
        toggle_type = ktype
    return {
        "lang": {"index": LANG_BOTH},
        "stem": {"index": {"stemming": "english"}},
        "stop": {"index": {"stop_words": "english"}},
        "lang-alt": {"index": LANG_BOTH, **pair({"multi_match": {"type": alts[0]}}, None)},
        "alt-a": pair({"multi_match": {"type": alts[0]}}, {"fusion": fusions[0]}),
        "alt-b": pair({"multi_match": {"type": alts[-1]}}, {"fusion": fusions[1]}),
        "fuzz": pair(
            {"multi_match": {"type": toggle_type, "fuzziness": toggled}},
            {
                "multi_match": {
                    "type": toggle_type if hy.get("multi_match") else "best_fields",
                    "fuzziness": toggled,
                }
            },
        ),
    }


def meilisearch(s: str) -> dict[str, Any]:
    t = tuned("meilisearch", s)
    ratio = (search(t, "hybrid").get("hybrid") or {}).get("semanticRatio", 0.5)
    lo, hi = round(max(0.05, ratio - 0.15), 2), round(min(0.95, ratio + 0.15), 2)
    if hi == ratio:
        hi = round(ratio - 0.3, 2)
    shipped = ["sort", "words", "typo", "proximity", "attributeRank", "wordPosition", "exactness"]
    moved = ["sort", "words", "attributeRank", "typo", "proximity", "wordPosition", "exactness"]
    rules = (t.get("index") or {}).get("rankingRules")
    alt_rules = shipped if rules == moved else moved
    dist = {"embedders": {"default": {"distribution": MEILI_DISTRIBUTION[s]}}}
    return {
        "last": pair({"matchingStrategy": "last"}, {"hybrid": {"semanticRatio": lo}}),
        "all": pair({"matchingStrategy": "all"}, {"hybrid": {"semanticRatio": hi}}),
        "rules": {"index": {"rankingRules": alt_rules}},
        # Elasticsearch's AUTO: one typo from 3 letters, two from 6 (Meilisearch: 5 and 9).
        "typo-3-6": {
            "index": {"typoTolerance": {"minWordSizeForTypos": {"oneTypo": 3, "twoTypos": 6}}}
        },
        "typo-4-8": {
            "index": {"typoTolerance": {"minWordSizeForTypos": {"oneTypo": 4, "twoTypos": 8}}}
        },
        "dist": {"index": dist},
        "dist-ratio": {"index": dist, **pair(None, {"hybrid": {"semanticRatio": lo}})},
    }


def typesense(s: str) -> dict[str, Any]:
    t = tuned("typesense", s)
    kw, hy = search(t, "keyword"), search(t, "hybrid")
    threshold = {250: 10, 10: 250}.get(kw.get("drop_tokens_threshold", 1), 250)
    match_type = {"max_weight": "max_score"}.get(
        kw.get("text_match_type", "max_score"), "max_weight"
    )
    alpha = (hy.get("vector_query_options") or {}).get("alpha", 0.3)
    a_lo, a_hi = round(max(0.05, alpha - 0.2), 2), round(min(0.95, alpha + 0.2), 2)
    if a_hi == alpha:
        a_hi = round(alpha - 0.4, 2)
    mode = "both_sides:3" if kw.get("drop_tokens_mode") == "left_to_right" else "left_to_right"
    hy_type = {"sum_score": "max_weight"}.get(hy.get("text_match_type", "max_score"), "sum_score")
    return {
        "lang": {"index": LANG_BOTH},
        "stem": {"index": {"stemming": "english"}},
        "stop": {"index": {"stop_words": "english"}},
        "alt-a": pair(
            {"drop_tokens_threshold": threshold}, {"vector_query_options": {"alpha": a_lo}}
        ),
        "alt-b": pair({"text_match_type": match_type}, {"vector_query_options": {"alpha": a_hi}}),
        "alt-c": pair(
            {"drop_tokens_mode": mode},
            {"rerank_hybrid_matches": not hy.get("rerank_hybrid_matches", False)},
        ),
        "lang-alt": {
            "index": LANG_BOTH,
            **pair({"drop_tokens_threshold": threshold}, {"text_match_type": hy_type}),
        },
    }


def qdrant(s: str) -> dict[str, Any]:
    t = tuned("qdrant", s)
    hy = search(t, "hybrid")
    fusion = hy.get("fusion") or {"fusion": "rrf"}
    fusions = [
        f
        for f in (
            {"fusion": "rrf"},
            {"fusion": "dbsf"},
            {"rrf": {"k": 60, "weights": [1, 3]}},
            {"rrf": {"k": 60, "weights": [1, 2]}},
        )
        if f != fusion
    ]
    corpus = (t.get("index") or {}).get("bm25_avg_len") == "corpus"
    # Qdrant language analysis (English stop words and stemmer) is on by default in qdrant/bm25.
    return {
        "avg-len": {"index": {"bm25_avg_len": "shipped" if corpus else "corpus"}},
        "k-1.2-b-0.75": {"index": {"bm25_options": {"k": 1.2, "b": 0.75}}},
        "k-1.5": {"index": {"bm25_options": {"k": 1.5, "b": 0.75}}},
        "b-0.5": {"index": {"bm25_options": {"k": 1.2, "b": 0.5}}},
        "k-0.9-b-0.4": {"index": {"bm25_options": {"k": 0.9, "b": 0.4}}},
        "fusion-a": pair(None, {"fusion": fusions[0]}),
        "fusion-b": pair({"keyword_pool": 100}, {"fusion": fusions[1]}),
    }


def postgres(s: str) -> dict[str, Any]:
    t = tuned("postgres", s)
    kw, hy = search(t, "keyword"), search(t, "hybrid")
    match = {"any": "all"}.get(kw.get("match", "all"), "any")
    weights = (hy.get("rrf") or {}).get("weights", [1, 1])
    ws = [w for w in ([1, 1], [1, 2], [2, 1], [1, 3]) if w != weights]
    # english text search configuration: stop words and Snowball stemming are always on.
    return {
        "match": pair({"match": match}, {"rrf": {"weights": ws[0]}}),
        "cd": pair({"rank": "ts_rank_cd"}, {"rrf": {"weights": ws[1]}}),
        "norm-1": pair({"normalization": 1}, {"rrf": {"weights": ws[2]}}),
        "norm-2": pair({"normalization": 2}, {"rrf": {"k": 20}}),
        "norm-32": pair(
            {"normalization": 32}, {"match": {"any": "all"}.get(hy.get("match", "all"), "any")}
        ),
    }


def paradedb(s: str) -> dict[str, Any]:
    t = tuned("postgres-paradedb", s)
    kw, hy = search(t, "keyword"), search(t, "hybrid")
    fuzzy = 0 if kw.get("fuzzy") else 1
    weights = (hy.get("rrf") or {}).get("weights", [1, 1])
    ws = [w for w in ([1, 1], [1, 2], [2, 1], [1, 3]) if w != weights]
    return {
        "lang": {"index": LANG_BOTH},
        "stem": {"index": {"stemming": "english"}},
        "stop": {"index": {"stop_words": "english"}},
        "fuzzy": pair({"fuzzy": fuzzy}, {"fuzzy": fuzzy, "rrf": {"weights": ws[0]}}),
        "match-all": pair({"match": "all"}, {"rrf": {"weights": ws[1]}}),
        "lang-fuzzy": {"index": LANG_BOTH, **pair({"fuzzy": fuzzy}, {"rrf": {"k": 20}})},
        "alt-c": pair(None, {"rrf": {"weights": ws[2]}}),
    }


ENGINES = {
    "elasticsearch": elasticsearch,
    "meilisearch": meilisearch,
    "typesense": typesense,
    "qdrant": qdrant,
    "postgres": postgres,
    "postgres-paradedb": paradedb,
}


def main() -> None:
    for engine, make in ENGINES.items():
        for scenario in SCENARIOS:
            variants = make(scenario)
            assert len(variants) <= MAX_VARIANTS, (engine, scenario, len(variants))
            path = ROOT / "engines" / engine / "sweeps" / f"{scenario}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            header = (
                f"# Sweep for {engine} on {scenario}, generated by scripts/make_sweeps.py.\n"
                "# Each variant is merged over the tuned track and runs as `tuned+<variant>` on the\n"
                "# tune half of the queries; the report publishes the test half.\n"
            )
            body = yaml.safe_dump({"variants": {"base": {}, **variants}}, sort_keys=False)
            path.write_text(header + body, encoding="utf-8")
            print(path.relative_to(ROOT), len(variants) + 1)


if __name__ == "__main__":
    main()
