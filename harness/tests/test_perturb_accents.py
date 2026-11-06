import random
import unicodedata

from irbench.models import Query
from irbench.perturb.accents import accents
from irbench.perturb.prefix import prefix
from irbench.perturb.variants import make_variants


def test_accents_strips_diacritics():
    rng = random.Random(0)
    assert accents("calefactor eléctrico", rng) == "calefactor electrico"
    assert accents("niño", rng) == "nino"
    assert accents("pingüino", rng) == "pinguino"


def test_accents_without_diacritics_is_none():
    assert accents("filtro cafetera", random.Random(0)) is None


def test_accents_is_nfd_minus_combining_marks():
    decomposed = unicodedata.normalize("NFD", "eléctrico")
    assert decomposed != "eléctrico"
    assert accents(decomposed, random.Random(0)) == "electrico"


def test_accents_output_is_nfc():
    out = accents("canción ñandú", random.Random(0))
    assert out == unicodedata.normalize("NFC", out)


def test_prefix_on_unspaced_japanese():
    text = "楽しくない仕事はなぜ楽しくないのか"
    for seed in range(50):
        out = prefix(text, random.Random(seed))
        assert out and out != text and text.startswith(out)
    assert prefix("時計", random.Random(1)) is None


def test_make_variants_applies_accents():
    q = Query(qid="1", text="calefactor eléctrico", variant_of="1", perturbation="original")
    out = make_variants([q], 1, ["accents"], [])
    assert [(v.qid, v.text, v.perturbation) for v in out[1:]] == [
        ("1-accents", "calefactor electrico", "accents")
    ]
    plain = Query(qid="2", text="filtro cafetera", variant_of="2", perturbation="original")
    assert [v.qid for v in make_variants([plain], 1, ["accents"], [])] == ["2"]
