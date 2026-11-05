"""Engine-neutral language-analysis keys of a config's `index` block.

`stop_words: english` and `stemming: english` turn on an engine's own stop word removal and
stemming (parity rules: every feature an engine ships may be used on the tuned track). Each
adapter maps them onto its native settings; an engine without the feature rejects the key.
Engines that take a user-supplied stop word list get ENGLISH_STOP_WORDS.
"""

from dataclasses import dataclass
from typing import Any

# Lucene's EnglishAnalyzer default stop set, the list Elasticsearch's `_english_` uses.
ENGLISH_STOP_WORDS = (
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is",
    "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there",
    "these", "they", "this", "to", "was", "will", "with",
)  # fmt: skip
LANGUAGE_KEYS = ("stop_words", "stemming")
LANGUAGES = ("english",)


@dataclass(frozen=True)
class Language:
    stop_words: bool = False
    stemming: bool = False


def split_language(index: dict[str, Any]) -> tuple[dict[str, Any], Language]:
    """The index config without the language keys, and the language features they turn on."""
    rest = {k: v for k, v in index.items() if k not in LANGUAGE_KEYS}
    on: dict[str, bool] = {}
    for key in LANGUAGE_KEYS:
        value = index.get(key)
        if value in (None, False):
            on[key] = False
        elif value in LANGUAGES:
            on[key] = True
        else:
            raise ValueError(f"index.{key} must be one of {LANGUAGES} or false, got {value!r}")
    return rest, Language(**on)
