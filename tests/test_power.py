import pytest

from evalci._power import analytic_n, analytic_power, simulate_power
from evalci.stats import power as power_fn


def test_analytic_n_and_power_are_consistent():
    n = analytic_n(delta=0.05, target_power=0.8, baseline=0.5, alpha=0.05)
    achieved = analytic_power(delta=0.05, n=n, baseline=0.5, alpha=0.05)
    assert achieved == pytest.approx(0.8, abs=0.01)


def test_analytic_power_increases_with_n():
    p1 = analytic_power(delta=0.03, n=200, baseline=0.5)
    p2 = analytic_power(delta=0.03, n=2000, baseline=0.5)
    assert p2 > p1


def test_analytic_power_increases_with_delta():
    p1 = analytic_power(delta=0.02, n=500, baseline=0.5)
    p2 = analytic_power(delta=0.1, n=500, baseline=0.5)
    assert p2 > p1


def test_analytic_n_decreases_with_larger_delta():
    n_small_effect = analytic_n(delta=0.02, target_power=0.8)
    n_large_effect = analytic_n(delta=0.1, target_power=0.8)
    assert n_large_effect < n_small_effect


def test_simulation_matches_analytic_at_rho_zero():
    # With rho=0 (fully independent items) the simulation should roughly
    # agree with the closed-form two-proportion approximation.
    n = 800
    delta = 0.05
    analytic = analytic_power(delta, n, baseline=0.5, alpha=0.05)
    simulated = simulate_power(delta, n, baseline=0.5, alpha=0.05, rho=0.0, sims=8000, random_state=0)
    assert simulated == pytest.approx(analytic, abs=0.05)


def test_power_public_api_solves_for_n():
    result = power_fn(delta=0.05, power=0.8, method="analytic")
    assert result.method == "analytic"
    assert result.n > 0
    assert result.power == 0.8


def test_power_public_api_solves_for_achieved_power():
    result = power_fn(delta=0.05, n=500, method="analytic")
    assert result.n == 500
    assert 0 <= result.power <= 1


def test_power_public_api_simulation_n_search_hits_target_roughly():
    result = power_fn(delta=0.08, power=0.8, method="simulation", rho=0.0, sims=1500, random_state=0)
    achieved = simulate_power(0.08, result.n, rho=0.0, sims=4000, random_state=1)
    assert achieved == pytest.approx(0.8, abs=0.1)


def test_power_rejects_unknown_method():
    with pytest.raises(ValueError):
        power_fn(delta=0.05, method="not-a-method")
