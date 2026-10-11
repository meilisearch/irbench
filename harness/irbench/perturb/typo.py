import random

from irbench.perturb.keyboard import QWERTY_NEIGHBORS

EDIT_KINDS = ("substitute", "delete", "duplicate", "transpose")
MIN_WORD_LEN = 4
TWO_EDITS_FROM_LEN = 8
MAX_ATTEMPTS = 10


def _edit(word: str, rng: random.Random) -> str:
    i = rng.randrange(len(word))
    kind = rng.choice(EDIT_KINDS)
    if kind == "substitute":
        neighbors = QWERTY_NEIGHBORS.get(word[i].lower())
        if neighbors:
            return word[:i] + rng.choice(neighbors) + word[i + 1 :]
        kind = "delete"
    if kind == "delete":
        return word[:i] + word[i + 1 :]
    if kind == "duplicate":
        return word[:i] + word[i] + word[i:]
    if i == len(word) - 1:
        i -= 1
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def typo(text: str, rng: random.Random) -> str | None:
    words = text.split()
    eligible = [i for i, w in enumerate(words) if len(w) >= MIN_WORD_LEN and w.isalpha()]
    if not eligible:
        return None
    index = rng.choice(eligible)
    word = words[index]
    edits = 1 if len(word) < TWO_EDITS_FROM_LEN else 2
    for _ in range(MAX_ATTEMPTS):
        candidate = word
        for _ in range(edits):
            candidate = _edit(candidate, rng)
        if candidate != word:
            words[index] = candidate
            return " ".join(words)
    return None
