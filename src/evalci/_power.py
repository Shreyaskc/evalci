"""Power / sample-size analysis: analytic (normal approximation) and simulation."""
import numpy as np
from scipy import stats as _scipy_stats


def analytic_n(delta, target_power=0.8, baseline=0.5, alpha=0.05):
    """Two-proportion normal-approximation sample size for a given effect size."""
    p1, p2 = baseline, baseline + delta
    z_alpha = _scipy_stats.norm.ppf(1 - alpha / 2)
    z_beta = _scipy_stats.norm.ppf(target_power)
    var = p1 * (1 - p1) + p2 * (1 - p2)
    n = ((z_alpha + z_beta) ** 2) * var / (delta**2)
    return int(np.ceil(n))


def analytic_power(delta, n, baseline=0.5, alpha=0.05):
    """Achieved power of a two-proportion test for given n and effect size."""
    p1, p2 = baseline, baseline + delta
    z_alpha = _scipy_stats.norm.ppf(1 - alpha / 2)
    var = p1 * (1 - p1) + p2 * (1 - p2)
    z_beta = np.sqrt(n * (delta**2) / var) - z_alpha
    return float(_scipy_stats.norm.cdf(z_beta))


def simulate_power(delta, n, baseline=0.5, alpha=0.05, rho=0.0, sims=5000, random_state=None):
    """Monte Carlo power for a paired design with correlated item-level outcomes.

    `rho` is the fraction of items whose correctness is shared between the two
    models (a common-difficulty link) rather than drawn independently; it is a
    simplified correlation knob, not an exact bivariate-Bernoulli correlation.
    """
    rng = np.random.default_rng(random_state)
    p1, p2 = baseline, baseline + delta
    shared = rng.random((sims, n)) < rho
    shared_correct = rng.random((sims, n)) < (p1 + p2) / 2
    a_ind = rng.random((sims, n)) < p1
    b_ind = rng.random((sims, n)) < p2
    a = np.where(shared, shared_correct, a_ind).astype(float)
    b = np.where(shared, shared_correct, b_ind).astype(float)
    diffs = a - b
    means = diffs.mean(axis=1)
    sds = diffs.std(axis=1, ddof=1)
    sds[sds == 0] = 1e-12
    z = means / (sds / np.sqrt(n))
    pvals = 2 * (1 - _scipy_stats.norm.cdf(np.abs(z)))
    return float(np.mean(pvals < alpha))


def simulate_n(delta, target_power=0.8, baseline=0.5, alpha=0.05, rho=0.0, sims=3000, random_state=None):
    """Bisection search for the n achieving target_power under simulate_power."""
    lo, hi = 4, 64
    max_n = 1_000_000
    while simulate_power(delta, hi, baseline, alpha, rho, sims, random_state) < target_power:
        lo = hi
        hi *= 2
        if hi > max_n:
            return max_n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        p = simulate_power(delta, mid, baseline, alpha, rho, sims, random_state)
        if p < target_power:
            lo = mid
        else:
            hi = mid
    return hi
