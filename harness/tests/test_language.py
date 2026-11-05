import pytest

from irbench.engines.language import ENGLISH_STOP_WORDS, Language, split_language


def test_language_keys_are_split_from_the_rest():
    rest, lang = split_language({"stop_words": "english", "stemming": False, "k": 1})
    assert rest == {"k": 1} and lang == Language(stop_words=True, stemming=False)
    assert split_language({}) == ({}, Language())


def test_unknown_language_is_rejected():
    with pytest.raises(ValueError, match="stemming"):
        split_language({"stemming": "french"})


def test_stop_list_is_lucenes_english_set():
    assert len(ENGLISH_STOP_WORDS) == 33 == len(set(ENGLISH_STOP_WORDS))
    assert {"the", "and", "with"} <= set(ENGLISH_STOP_WORDS) and "what" not in ENGLISH_STOP_WORDS
