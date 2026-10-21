from collections.abc import Callable

from irbench.datasets.amazon_esci import normalize_amazon_esci
from irbench.datasets.cqadupstack import normalize_cqadupstack
from irbench.datasets.lotte import normalize_lotte
from irbench.datasets.trec_covid import normalize_trec_covid
from irbench.datasets.wands import NormalizedDataset, normalize_wands

NORMALIZERS: dict[str, Callable[..., NormalizedDataset]] = {
    "wands": normalize_wands,
    "cqadupstack": normalize_cqadupstack,
    "lotte-lifestyle": normalize_lotte,
    "trec-covid": normalize_trec_covid,
    "amazon-esci": normalize_amazon_esci,
}
