from collections.abc import Callable
from pathlib import Path

from irbench.datasets.wands import NormalizedDataset, normalize_wands

NORMALIZERS: dict[str, Callable[[dict[str, Path], Path], NormalizedDataset]] = {
    "wands": normalize_wands,
}
