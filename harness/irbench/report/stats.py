"""Paired significance test between two runs of the same scenario (no JavaScript, computed
when the report is built)."""

import numpy as np

ROUNDS = 10_000
ALPHA = 0.05
CHUNK = 1_000


def paired_randomization_p(
    a: dict[str, float], b: dict[str, float], rounds: int = ROUNDS, seed: int = 0
) -> float | None:
    """Two-sided p-value of a paired randomization (sign-flip) test on the mean difference,
    over the tasks both runs scored. None when they share no task.

    Each round swaps the two runs' scores on a random subset of tasks; p is the share of
    rounds whose mean difference is at least as large as the observed one."""
    shared = sorted(a.keys() & b.keys())
    if not shared:
        return None
    diff = np.array([a[t] - b[t] for t in shared])
    observed = abs(diff.mean())
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, rounds, CHUNK):
        size = min(CHUNK, rounds - start)
        signs = rng.choice((-1.0, 1.0), size=(size, len(diff)))
        hits += int((np.abs(signs @ diff) / len(diff) >= observed - 1e-12).sum())
    return (hits + 1) / (rounds + 1)
