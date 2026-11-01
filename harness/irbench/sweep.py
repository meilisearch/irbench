"""Sweep significance rule (spec 2026-10-31 Chinese title search §5.2): a setting's variant
replaces the default only when it is significantly better, its p-value Holm-corrected
together with the setting's other variants."""

from irbench.report.stats import ALPHA, holm, paired_randomization_p


def _mean(scores: dict[str, float]) -> float:
    return sum(scores.values()) / len(scores) if scores else 0.0


def keep_variant(
    default: dict[str, float], variants: dict[str, dict[str, float]], alpha: float = ALPHA
) -> tuple[str | None, dict[str, float]]:
    """The variant to keep (None: keep the default) and each variant's Holm-adjusted p.

    Every variant is compared with the default, never with another variant, by the paired
    randomization test on per-task scores. Eligible: a higher mean and adjusted p < alpha.
    The eligible variant with the highest mean is kept."""
    raw = {}
    for name, scores in variants.items():
        p = paired_randomization_p(default, scores)
        raw[name] = 1.0 if p is None else p
    adjusted = holm(raw)
    base = _mean(default)
    eligible = [n for n in variants if _mean(variants[n]) > base and adjusted[n] < alpha]
    if not eligible:
        return None, adjusted
    return max(eligible, key=lambda n: _mean(variants[n])), adjusted
