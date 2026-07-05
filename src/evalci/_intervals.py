"""Interval estimators: Wilson, Clopper-Pearson, and bootstrap (percentile/BCa)."""
import numpy as np
from scipy import stats as _scipy_stats


def wilson_interval(k, n, confidence=0.95):
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    z = _scipy_stats.norm.ppf(0.5 + confidence / 2)
    phat = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    half = (z * np.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def clopper_pearson_interval(k, n, confidence=0.95):
    """Exact (Clopper-Pearson) interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    alpha = 1 - confidence
    lo = 0.0 if k == 0 else _scipy_stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _scipy_stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def bootstrap_interval(
    data,
    statistic=np.mean,
    method="BCa",
    confidence=0.95,
    n_resamples=9999,
    random_state=None,
    vectorized=True,
):
    """Bootstrap CI for an arbitrary statistic via scipy.stats.bootstrap."""
    data = np.asarray(data, dtype=float)
    scipy_method = {"bca": "BCa", "percentile": "percentile", "basic": "basic"}.get(
        method.lower(), method
    )
    if np.all(data == data[0]):
        # degenerate (zero-variance) sample: BCa's acceleration is undefined
        point = float(statistic(data))
        return point, point
    res = _scipy_stats.bootstrap(
        (data,),
        statistic,
        confidence_level=confidence,
        method=scipy_method,
        n_resamples=n_resamples,
        random_state=random_state,
        vectorized=vectorized,
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)
