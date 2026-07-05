import numpy as np
import pytest
from statsmodels.stats.proportion import proportion_confint

from evalci._intervals import bootstrap_interval, clopper_pearson_interval, wilson_interval


@pytest.mark.parametrize("k,n", [(80, 100), (1, 20), (0, 15), (15, 15), (500, 1000), (3, 3)])
def test_wilson_matches_statsmodels(k, n):
    lo, hi = wilson_interval(k, n, confidence=0.95)
    ref_lo, ref_hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    assert lo == pytest.approx(ref_lo, abs=1e-9)
    assert hi == pytest.approx(ref_hi, abs=1e-9)


@pytest.mark.parametrize("k,n", [(80, 100), (1, 20), (0, 15), (15, 15), (500, 1000), (3, 3)])
def test_clopper_pearson_matches_statsmodels(k, n):
    lo, hi = clopper_pearson_interval(k, n, confidence=0.95)
    ref_lo, ref_hi = proportion_confint(k, n, alpha=0.05, method="beta")
    assert lo == pytest.approx(ref_lo, abs=1e-9)
    assert hi == pytest.approx(ref_hi, abs=1e-9)


def test_clopper_pearson_is_wider_than_wilson():
    # Clopper-Pearson is famously conservative; Wilson is tighter on average.
    lo_w, hi_w = wilson_interval(50, 100)
    lo_cp, hi_cp = clopper_pearson_interval(50, 100)
    assert (hi_cp - lo_cp) >= (hi_w - lo_w)


def test_wilson_rejects_zero_n():
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_bootstrap_interval_percentile_coverage():
    # Coverage simulation: over many synthetic samples, the 95% CI should
    # contain the true mean roughly 95% of the time.
    rng = np.random.default_rng(42)
    true_p = 0.7
    hits = 0
    trials = 200
    for _ in range(trials):
        sample = (rng.random(150) < true_p).astype(float)
        lo, hi = bootstrap_interval(sample, np.mean, method="percentile", n_resamples=500, random_state=rng)
        hits += lo <= true_p <= hi
    coverage = hits / trials
    assert 0.85 <= coverage <= 1.0


def test_bootstrap_interval_degenerate_data():
    data = np.ones(50)
    lo, hi = bootstrap_interval(data, np.mean, n_resamples=200)
    assert lo == hi == pytest.approx(1.0)


def test_bootstrap_interval_bca_contains_percentile_estimate():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=2.0, size=300)
    lo, hi = bootstrap_interval(data, np.mean, method="BCa", n_resamples=2000, random_state=rng)
    assert lo < np.mean(data) < hi
