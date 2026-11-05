import pytest

from irbench.report.stats import ROUNDS, paired_randomization_p


def test_identical_runs_are_not_different():
    a = {f"t{i}": i / 10 for i in range(10)}
    assert paired_randomization_p(a, dict(a)) == 1.0


def test_a_consistent_gain_is_significant():
    a = {f"t{i}": 0.5 + (i % 3) / 100 for i in range(30)}
    b = {t: v - 0.1 for t, v in a.items()}
    # Every flip but the observed one and its mirror lowers |mean|: p is at its floor.
    assert paired_randomization_p(a, b) == pytest.approx(1 / (ROUNDS + 1), abs=5e-4)


def test_noise_is_not_significant():
    a = {f"t{i}": (0.6 if i % 2 else 0.4) for i in range(20)}
    b = {f"t{i}": (0.4 if i % 2 else 0.6) for i in range(20)}
    assert paired_randomization_p(a, b) == 1.0  # mean difference 0


def test_only_shared_tasks_count_and_the_seed_is_fixed():
    a = {"t1": 1.0, "t2": 0.0, "t3": 1.0, "only_a": 9.0}
    b = {"t1": 0.0, "t2": 0.0, "t3": 0.5, "only_b": -9.0}
    p = paired_randomization_p(a, b)
    assert p == paired_randomization_p(a, b)
    assert 0.0 < p <= 1.0
    assert paired_randomization_p({"x": 1.0}, {"y": 1.0}) is None
