import random

from irbench.perturb.keyboard import QWERTY_NEIGHBORS
from irbench.perturb.typo import typo


def _edit_distance(a: str, b: str) -> int:
    # Optimal string alignment distance (counts a transposition as 1 edit).
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


def test_neighbors_follow_keyboard_layout():
    assert QWERTY_NEIGHBORS["f"] == ["c", "d", "g", "r", "t", "v"]
    assert QWERTY_NEIGHBORS["a"] == ["q", "s", "w", "z"]


def test_same_seed_same_output():
    outs = {typo("solid wood platform bed", random.Random("s:1:typo")) for _ in range(5)}
    assert len(outs) == 1


def test_only_one_word_changes_and_short_words_untouched():
    for seed in range(200):
        out = typo("a tv platform bed", random.Random(seed))
        words = out.split()
        assert words[0] == "a" and words[1] == "tv"
        changed = [i for i, (x, y) in enumerate(zip(words, "a tv platform bed".split())) if x != y]
        assert len(changed) == 1


def test_edit_budget_by_word_length():
    for seed in range(300):
        short = typo("table", random.Random(seed))  # 5 chars -> 1 edit
        assert 1 <= _edit_distance(short, "table") <= 1
        long = typo("nightstand", random.Random(seed))  # 10 chars -> 2 edits
        assert 1 <= _edit_distance(long, "nightstand") <= 2
        assert long != "nightstand"


def test_no_eligible_word_returns_none():
    assert typo("tv 3d", random.Random(1)) is None
