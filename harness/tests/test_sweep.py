"""Sweep significance rule (spec 2026-10-31 Chinese title search §5.2)."""

import numpy as np
import pytest

from irbench.report.stats import holm
from irbench.sweep import keep_variant

N = 400


def _scores(mean: float, seed: int, noise: float = 0.05) -> dict[str, float]:
    values = mean + np.random.default_rng(seed).normal(0, noise, N)
    return {f"t{i}": float(v) for i, v in enumerate(values)}


def test_holm_known_values():
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx(
        {"a": 0.03, "b": 0.06, "c": 0.06}
    )
    assert holm({"a": 0.9, "b": 0.8}) == pytest.approx({"a": 1.0, "b": 1.0})
    assert holm({}) == {}


def test_no_variants_keeps_default():
    kept, adjusted = keep_variant(_scores(0.5, 0), {})
    assert kept is None and adjusted == {}


def test_identical_variant_keeps_default():
    default = _scores(0.5, 0)
    kept, adjusted = keep_variant(default, {"same": dict(default)})
    assert kept is None and adjusted["same"] == 1.0


def test_worse_significant_variant_not_kept():
    default = _scores(0.5, 0)
    worse = {t: v - 0.2 for t, v in default.items()}
    kept, adjusted = keep_variant(default, {"worse": worse})
    assert kept is None and adjusted["worse"] < 0.05


def test_spec_worked_case():
    """standard beats cjk significantly; smartcn scores higher than standard. smartcn is kept
    only if it is itself significantly above cjk (the default), otherwise standard is."""
    cjk = _scores(0.4, 1)
    standard = {t: v + 0.10 for t, v in cjk.items()}
    smartcn = {t: v + 0.12 for t, v in cjk.items()}
    kept, adjusted = keep_variant(cjk, {"standard": standard, "smartcn": smartcn})
    assert kept == "smartcn" and adjusted["smartcn"] < 0.05

    # Same mean gain (+0.12), but ±3 swings per task: not significant against cjk.
    noisy = {t: v + 0.12 + (3.0 if i % 2 else -3.0) for i, (t, v) in enumerate(cjk.items())}
    kept, adjusted = keep_variant(cjk, {"standard": standard, "smartcn": noisy})
    assert adjusted["smartcn"] >= 0.05
    assert kept == "standard"
