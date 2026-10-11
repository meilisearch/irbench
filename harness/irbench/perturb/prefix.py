import random

MIN_KEEP = 2
MIN_WORD_LEN = 3


def prefix(text: str, rng: random.Random) -> str | None:
    words = text.split()
    if not words or len(words[-1]) < MIN_WORD_LEN:
        return None
    last = words[-1]
    keep = max(MIN_KEEP, round(len(last) * rng.uniform(0.4, 0.7)))
    if keep >= len(last):
        return None
    words[-1] = last[:keep]
    return " ".join(words)
