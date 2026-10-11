import random
import re


def _pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")


def synonym(text: str, rng: random.Random, groups: list[list[str]]) -> str | None:
    lowered = " ".join(text.lower().split())
    matches = [
        (term, group) for group in groups for term in group if _pattern(term).search(lowered)
    ]
    if not matches:
        return None
    longest = max(len(term) for term, _ in matches)
    candidates = sorted((m for m in matches if len(m[0]) == longest), key=lambda m: m[0])
    term, group = rng.choice(candidates)
    replacement = rng.choice([t for t in group if t != term])
    return _pattern(term).sub(replacement, lowered, count=1)
