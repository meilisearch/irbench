import random
import unicodedata


def accents(text: str, rng: random.Random) -> str | None:
    """Drop diacritics ("eléctrico" -> "electrico"); None when the text has none.

    `rng` is unused: the transformation is deterministic. It keeps the `_apply` signature.
    """
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    out = unicodedata.normalize("NFC", stripped)
    return None if out == text else out
