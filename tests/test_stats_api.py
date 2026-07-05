import numpy as np
import pandas as pd
import pytest

from evalci.schema import from_records
from evalci.stats import CIResult, CompareResult, cluster_ci, ci, compare, multi_compare


def test_ci_wilson_binary():
    scores = np.array([1.0] * 80 + [0.0] * 20)
    result = ci(scores, method="wilson")
    assert isinstance(result, CIResult)
    assert result.estimate == pytest.approx(0.8)
    assert result.lower < 0.8 < result.upper


def test_ci_rejects_non_binary_for_wilson():
    with pytest.raises(ValueError):
        ci(np.array([0.1, 0.9, 1.0]), method="wilson")


def test_ci_bootstrap_accepts_continuous_scores():
    rng = np.random.default_rng(0)
    scores = rng.normal(3.0, 1.0, 200)
    result = ci(scores, method="bootstrap", n_resamples=500, random_state=rng)
    assert result.lower < scores.mean() < result.upper


def test_ci_empty_raises():
    with pytest.raises(ValueError):
        ci(np.array([]))


def test_ci_unpacks_as_tuple():
    scores = np.array([1.0, 1.0, 0.0, 1.0])
    lo, hi = ci(scores, method="wilson")
    assert lo <= hi


def test_compare_paired_permutation_shape():
    rng = np.random.default_rng(1)
    a = (rng.random(200) < 0.8).astype(float)
    b = (rng.random(200) < 0.8).astype(float)
    result = compare(a, b, method="permutation", n_resamples=1000, random_state=0)
    assert isinstance(result, CompareResult)
    assert result.n == 200
    assert 0.0 <= result.p_value <= 1.0


def test_compare_length_mismatch_raises_for_paired():
    with pytest.raises(ValueError):
        compare([1, 0, 1], [1, 0], paired=True)


def test_compare_mcnemar_requires_paired():
    with pytest.raises(ValueError):
        compare([1, 0], [0, 1], paired=False, method="mcnemar")


def test_compare_unknown_method_raises():
    with pytest.raises(ValueError):
        compare([1, 0], [0, 1], method="not-a-method")


def test_compare_unpaired_independent_samples():
    rng = np.random.default_rng(2)
    a = rng.normal(0.8, 0.1, 100)
    b = rng.normal(0.5, 0.1, 120)
    result = compare(a, b, paired=False, method="bootstrap", n_resamples=1000, random_state=0)
    assert result.delta == pytest.approx(a.mean() - b.mean())
    assert result.p_value < 0.01


def test_compare_detects_true_difference():
    rng = np.random.default_rng(3)
    a = (rng.random(500) < 0.9).astype(float)
    b = (rng.random(500) < 0.5).astype(float)
    result = compare(a, b, method="permutation", n_resamples=2000, random_state=0)
    assert result.p_value < 0.001
    assert result.delta > 0.3


def test_compare_no_difference_not_significant():
    rng = np.random.default_rng(4)
    a = (rng.random(500) < 0.7).astype(float)
    b = (rng.random(500) < 0.7).astype(float)
    result = compare(a, b, method="permutation", n_resamples=2000, random_state=0)
    assert result.p_value > 0.05


def _make_df(model_ps, n_items=200, subset="task", seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for model, p in model_ps.items():
        scores = (rng.random(n_items) < p).astype(float)
        frames.append(from_records(range(n_items), model, scores, subsets=[subset] * n_items))
    return pd.concat(frames, ignore_index=True)


def test_multi_compare_holm_correction_applied():
    df = _make_df({"a": 0.9, "b": 0.6, "c": 0.55}, seed=5)
    result = multi_compare(df, correction="holm", n_resamples=1000, random_state=0)
    assert set(result.columns) >= {"model_a", "model_b", "delta", "p_value", "p_adj", "significant"}
    assert (result["p_adj"] >= result["p_value"] - 1e-12).all()
    row = result[(result.model_a == "a") & (result.model_b == "b")].iloc[0]
    assert row["significant"]


def test_multi_compare_requires_two_models():
    df = from_records(range(10), "solo", np.ones(10))
    with pytest.raises(ValueError):
        multi_compare(df)


def test_cluster_ci_basic():
    rng = np.random.default_rng(6)
    clusters = np.repeat(np.arange(30), 4)
    scores = (rng.random(120) < 0.65).astype(float)
    result = cluster_ci(scores, clusters, n_resamples=1000, random_state=0)
    assert result.estimate == pytest.approx(scores.mean())
    assert result.lower < scores.mean() < result.upper


def test_cluster_ci_length_mismatch_raises():
    with pytest.raises(ValueError):
        cluster_ci(np.ones(10), np.arange(5))


def test_cluster_ci_wider_than_naive_bootstrap_for_correlated_clusters():
    # Items within a cluster are perfectly correlated (identical), so the
    # effective sample size is the number of clusters, not the number of items.
    rng = np.random.default_rng(7)
    n_clusters = 20
    cluster_values = (rng.random(n_clusters) < 0.6).astype(float)
    scores = np.repeat(cluster_values, 10)
    clusters = np.repeat(np.arange(n_clusters), 10)
    from evalci._intervals import bootstrap_interval

    naive_lo, naive_hi = bootstrap_interval(scores, np.mean, n_resamples=2000, random_state=0)
    result = cluster_ci(scores, clusters, n_resamples=2000, random_state=0)
    assert (result.upper - result.lower) > (naive_hi - naive_lo)
