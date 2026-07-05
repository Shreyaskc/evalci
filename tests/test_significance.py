import itertools

import numpy as np
import pytest
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

from evalci._significance import (
    mcnemar_test,
    paired_bootstrap_p,
    paired_permutation_p,
    unpaired_bootstrap_p,
    unpaired_permutation_p,
)


def test_paired_permutation_exact_matches_brute_force():
    diffs = np.array([1.0, -1.0, 2.0, 0.5, -0.5, 3.0])
    p = paired_permutation_p(diffs, exact_threshold=12)

    obs = diffs.mean()
    n = len(diffs)
    combos = list(itertools.product([-1, 1], repeat=n))
    count = sum(1 for signs in combos if abs(np.mean(np.array(signs) * diffs)) >= abs(obs) - 1e-9)
    expected = count / len(combos)
    assert p == pytest.approx(expected, abs=1e-12)


def test_paired_permutation_zero_diff_gives_p_one():
    diffs = np.zeros(10)
    p = paired_permutation_p(diffs, exact_threshold=12)
    assert p == pytest.approx(1.0)


def test_paired_permutation_monte_carlo_converges_to_exact():
    rng = np.random.default_rng(7)
    diffs = rng.normal(loc=0.3, scale=1.0, size=10)
    exact_p = paired_permutation_p(diffs, exact_threshold=12)
    mc_p = paired_permutation_p(diffs, n_resamples=20000, rng=np.random.default_rng(1), exact_threshold=0)
    assert mc_p == pytest.approx(exact_p, abs=0.02)


def test_unpaired_permutation_p_in_valid_range():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 40)
    b = rng.normal(0.5, 1, 40)
    p = unpaired_permutation_p(a, b, n_resamples=5000, rng=rng)
    assert 0.0 < p <= 1.0


def test_permutation_p_small_for_large_effect():
    rng = np.random.default_rng(5)
    a = rng.normal(5, 0.1, 60)
    b = rng.normal(0, 0.1, 60)
    p = unpaired_permutation_p(a, b, n_resamples=2000, rng=rng)
    assert p < 0.01


def test_paired_bootstrap_p_small_for_large_effect():
    rng = np.random.default_rng(9)
    diffs = rng.normal(2.0, 0.2, 60)
    p = paired_bootstrap_p(diffs, n_resamples=2000, rng=rng)
    assert p < 0.01


def test_unpaired_bootstrap_p_large_for_no_effect():
    rng = np.random.default_rng(11)
    a = rng.normal(0, 1, 100)
    b = rng.normal(0, 1, 100)
    p = unpaired_bootstrap_p(a, b, n_resamples=2000, rng=rng)
    assert p > 0.05


@pytest.mark.parametrize("n01,n10", [(10, 3), (30, 10), (2, 1), (0, 0)])
def test_mcnemar_exact_matches_statsmodels(n01, n10):
    a = np.array([1] * n10 + [0] * n01)
    b = np.array([0] * n10 + [1] * n01)
    p, got01, got10, stat = mcnemar_test(a, b, exact=True)
    assert got01 == n01 and got10 == n10
    if n01 + n10 == 0:
        assert p == 1.0
        return
    ref = sm_mcnemar([[0, n01], [n10, 0]], exact=True)
    assert p == pytest.approx(ref.pvalue, abs=1e-9)


@pytest.mark.parametrize("n01,n10", [(30, 10), (50, 20), (15, 40)])
def test_mcnemar_asymptotic_matches_statsmodels(n01, n10):
    a = np.array([1] * n10 + [0] * n01)
    b = np.array([0] * n10 + [1] * n01)
    p, got01, got10, stat = mcnemar_test(a, b, exact=False, correction=True)
    ref = sm_mcnemar([[0, n01], [n10, 0]], exact=False, correction=True)
    assert stat == pytest.approx(ref.statistic, abs=1e-9)
    assert p == pytest.approx(ref.pvalue, abs=1e-9)
