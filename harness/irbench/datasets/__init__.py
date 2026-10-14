from collections.abc import Callable
from pathlib import Path

from irbench.datasets.cqadupstack import normalize_cqadupstack
from irbench.datasets.lotte import normalize_lotte
from irbench.datasets.wands import NormalizedDataset, normalize_wands

NORMALIZERS: dict[str, Callable[[dict[str, Path], Path], NormalizedDataset]] = {
    "wands": normalize_wands,
    "cqadupstack": normalize_cqadupstack,
    "lotte-lifestyle": normalize_lotte,
}
