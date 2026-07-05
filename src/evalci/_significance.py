"""Significance-test internals: paired/unpaired permutation, bootstrap, McNemar."""
import itertools

import numpy as np
from scipy import stats as _scipy_stats

EXACT_ENUMERATION_THRESHOLD = 12


def paired_bootstrap_ci(diffs, confidence=0.95, n_resamples=9999, rng=None):
    rng = rng or np.random.default_rng()
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = diffs[idx].mean(axis=1)
    alpha = 1 - confidence
    lo, hi = np.percentile(resampled_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def unpaired_bootstrap_ci(a, b, confidence=0.95, n_resamples=9999, rng=None):
    rng = rng or np.random.default_rng()
    na, nb = len(a), len(b)
    idx_a = rng.integers(0, na, size=(n_resamples, na))
    idx_b = rng.integers(0, nb, size=(n_resamples, nb))
    stats = a[idx_a].mean(axis=1) - b[idx_b].mean(axis=1)
    alpha = 1 - confidence
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_permutation_p(diffs, n_resamples=9999, rng=None, exact_threshold=EXACT_ENUMERATION_THRESHOLD):
    """Sign-flip permutation test on paired differences. Exact enumeration for small n."""
    rng = rng or np.random.default_rng()
    n = len(diffs)
    obs = diffs.mean()
    if n <= exact_threshold:
        signs = np.array(list(itertools.product([-1, 1], repeat=n)), dtype=float)
        resampled = (signs * diffs).mean(axis=1)
        count = np.sum(np.abs(resampled) >= abs(obs) - 1e-9)
        return float(count / len(signs))
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, n))
    resampled = (signs * diffs).mean(axis=1)
    count = np.sum(np.abs(resampled) >= abs(obs) - 1e-9)
    return float((count + 1) / (n_resamples + 1))


def unpaired_permutation_p(a, b, n_resamples=9999, rng=None):
    """Label-shuffle permutation test for two independent samples."""
    rng = rng or np.random.default_rng()
    na = len(a)
    combined = np.concatenate([a, b])
    obs = a.mean() - b.mean()
    tiled = np.broadcast_to(combined, (n_resamples, len(combined))).copy()
    perms = rng.permuted(tiled, axis=1)
    stat = perms[:, :na].mean(axis=1) - perms[:, na:].mean(axis=1)
    count = np.sum(np.abs(stat) >= abs(obs) - 1e-9)
    return float((count + 1) / (n_resamples + 1))


def paired_bootstrap_p(diffs, n_resamples=9999, rng=None):
    """Null-shifted bootstrap hypothesis test on paired differences."""
    rng = rng or np.random.default_rng()
    n = len(diffs)
    obs = diffs.mean()
    shifted = diffs - obs
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled = shifted[idx].mean(axis=1)
    count = np.sum(np.abs(resampled) >= abs(obs) - 1e-9)
    return float((count + 1) / (n_resamples + 1))


def unpaired_bootstrap_p(a, b, n_resamples=9999, rng=None):
    """Null-shifted bootstrap hypothesis test for two independent samples."""
    rng = rng or np.random.default_rng()
    obs = a.mean() - b.mean()
    combined_mean = np.concatenate([a, b]).mean()
    a_shift = a - a.mean() + combined_mean
    b_shift = b - b.mean() + combined_mean
    na, nb = len(a), len(b)
    idx_a = rng.integers(0, na, size=(n_resamples, na))
    idx_b = rng.integers(0, nb, size=(n_resamples, nb))
    stats = a_shift[idx_a].mean(axis=1) - b_shift[idx_b].mean(axis=1)
    count = np.sum(np.abs(stats) >= abs(obs) - 1e-9)
    return float((count + 1) / (n_resamples + 1))


def mcnemar_test(a, b, correction=True, exact=None):
    """McNemar's test on paired binary (0/1 correctness) outcomes.

    Returns (p_value, n01, n10, statistic) where n01 = b-correct-only,
    n10 = a-correct-only, statistic is None when the exact binomial test is used.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n = n01 + n10
    if exact is None:
        exact = n < 25
    if n == 0:
        return 1.0, n01, n10, 0.0
    if exact:
        k = min(n01, n10)
        p = _scipy_stats.binomtest(k, n, p=0.5, alternative="two-sided").pvalue
        return float(p), n01, n10, None
    if correction:
        statistic = (abs(n01 - n10) - 1) ** 2 / n
    else:
        statistic = (n01 - n10) ** 2 / n
    p = float(1 - _scipy_stats.chi2.cdf(statistic, df=1))
    return p, n01, n10, float(statistic)
