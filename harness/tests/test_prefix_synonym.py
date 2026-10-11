import random

from irbench.perturb.prefix import prefix
from irbench.perturb.synonym import synonym

GROUPS = [["sofa", "couch", "settee"], ["coffee table", "cocktail table"], ["rug", "carpet"]]


def test_prefix_truncates_last_word_within_bounds():
    for seed in range(300):
        out = prefix("solid wood nightstand", random.Random(seed))
        head, last = out.rsplit(" ", 1)
        assert head == "solid wood"
        assert "nightstand".startswith(last)
        assert round(10 * 0.4) <= len(last) <= round(10 * 0.7)


def test_prefix_min_two_chars_and_short_word_none():
    assert prefix("big bed", random.Random(3)) == "big be"
    assert prefix("a tv", random.Random(3)) is None


def test_synonym_replaces_whole_words_only():
    out = synonym("rugged sofa", random.Random(1), GROUPS)
    assert out in {"rugged couch", "rugged settee"}


def test_synonym_prefers_longest_phrase():
    for seed in range(20):
        assert synonym("oak coffee table", random.Random(seed), GROUPS) == "oak cocktail table"


def test_synonym_no_match_returns_none():
    assert synonym("platform bed", random.Random(1), GROUPS) is None


def test_synonym_is_deterministic():
    a = synonym("sofa with rug", random.Random("x"), GROUPS)
    b = synonym("sofa with rug", random.Random("x"), GROUPS)
    assert a == b
