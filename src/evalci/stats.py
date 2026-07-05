"""Public statistics API: ci, compare, power, multi_compare, cluster_ci."""
import itertools
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from . import _power
from ._correction import correct_pvalues
from ._intervals import bootstrap_interval, clopper_pearson_interval, wilson_interval
from ._significance import (
    mcnemar_test,
    paired_bootstrap_ci,
    paired_bootstrap_p,
    paired_permutation_p,
    unpaired_bootstrap_ci,
    unpaired_bootstrap_p,
    unpaired_permutation_p,
)
from .schema import to_paired_arrays, validate_schema

_BINARY_CI_METHODS = {"wilson", "clopper-pearson", "exact"}


@dataclass
class CIResult:
    estimate: float
    lower: float
    upper: float
    method: str
    n: int
    confidence: float = 0.95

    def __iter__(self):
        return iter((self.lower, self.upper))

    def __repr__(self):
        return (
            f"CIResult(estimate={self.estimate:.4f}, "
            f"ci=[{self.lower:.4f}, {self.upper:.4f}], "
            f"method={self.method!r}, n={self.n})"
        )


@dataclass
class CompareResult:
    delta: float
    ci: Tuple[float, float]
    p_value: float
    method: str
    paired: bool
    n: int
    confidence: float = 0.95
    extra: dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"CompareResult(delta={self.delta:.4f}, "
            f"ci=[{self.ci[0]:.4f}, {self.ci[1]:.4f}], "
            f"p_value={self.p_value:.4g}, method={self.method!r}, "
            f"paired={self.paired}, n={self.n})"
        )


@dataclass
class PowerResult:
    delta: float
    n: int
    power: float
    alpha: float
    method: str


def ci(scores, method="wilson", confidence=0.95, n_resamples=9999, random_state=None):
    """Confidence interval on a single model's score.

    method: "wilson" or "clopper-pearson"/"exact" for binary (0/1) scores,
    or "bootstrap" (percentile/BCa on the mean) for continuous scores.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        raise ValueError("scores must be non-empty")
    method_l = method.lower()
    if method_l in _BINARY_CI_METHODS:
        if not np.all(np.isin(scores, [0.0, 1.0])):
            raise ValueError(
                f"method={method!r} requires binary 0/1 scores; use method='bootstrap' "
                "for continuous scores"
            )
        k = int(scores.sum())
        if method_l == "wilson":
            lo, hi = wilson_interval(k, n, confidence)
        else:
            lo, hi = clopper_pearson_interval(k, n, confidence)
        return CIResult(estimate=k / n, lower=lo, upper=hi, method=method_l, n=n, confidence=confidence)
    if method_l == "bootstrap":
        est = float(scores.mean())
        lo, hi = bootstrap_interval(
            scores, np.mean, confidence=confidence, n_resamples=n_resamples, random_state=random_state
        )
        return CIResult(estimate=est, lower=lo, upper=hi, method="bootstrap", n=n, confidence=confidence)
    raise ValueError(f"unknown method: {method!r}")


def compare(
    a,
    b,
    paired=True,
    method="permutation",
    confidence=0.95,
    n_resamples=9999,
    correction=True,
    random_state=None,
):
    """Model-vs-model comparison. method: "permutation", "bootstrap", or "mcnemar" (paired only)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(random_state)
    method_l = method.lower()

    if paired:
        if len(a) != len(b):
            raise ValueError("paired comparison requires equal-length arrays")
        diffs = a - b
        n = len(diffs)
        delta = float(diffs.mean())
        lo, hi = paired_bootstrap_ci(diffs, confidence=confidence, n_resamples=n_resamples, rng=rng)
        extra = {}
        if method_l == "permutation":
            p = paired_permutation_p(diffs, n_resamples=n_resamples, rng=rng)
        elif method_l == "bootstrap":
            p = paired_bootstrap_p(diffs, n_resamples=n_resamples, rng=rng)
        elif method_l == "mcnemar":
            p, n01, n10, statistic = mcnemar_test(a, b, correction=correction)
            extra = {"n01": n01, "n10": n10, "statistic": statistic}
        else:
            raise ValueError(f"unknown method: {method!r}")
    else:
        if method_l == "mcnemar":
            raise ValueError("mcnemar requires paired=True")
        n = len(a) + len(b)
        delta = float(a.mean() - b.mean())
        lo, hi = unpaired_bootstrap_ci(a, b, confidence=confidence, n_resamples=n_resamples, rng=rng)
        extra = {}
        if method_l == "permutation":
            p = unpaired_permutation_p(a, b, n_resamples=n_resamples, rng=rng)
        elif method_l == "bootstrap":
            p = unpaired_bootstrap_p(a, b, n_resamples=n_resamples, rng=rng)
        else:
            raise ValueError(f"unknown method: {method!r}")

    return CompareResult(
        delta=delta,
        ci=(lo, hi),
        p_value=p,
        method=method_l,
        paired=paired,
        n=n,
        confidence=confidence,
        extra=extra,
    )


def power(
    delta,
    n=None,
    power=0.8,
    baseline=0.5,
    alpha=0.05,
    method="analytic",
    rho=0.0,
    sims=5000,
    random_state=None,
):
    """Sample-size (n is None) or achieved-power (n given) calculator.

    method="analytic" uses a two-proportion normal approximation (fast, ignores
    pairing correlation). method="simulation" simulates a paired design with a
    `rho` correlation knob between the two models' item-level correctness.
    """
    method_l = method.lower()
    if method_l == "analytic":
        if n is None:
            solved_n = _power.analytic_n(delta, target_power=power, baseline=baseline, alpha=alpha)
            return PowerResult(delta=delta, n=solved_n, power=power, alpha=alpha, method="analytic")
        achieved = _power.analytic_power(delta, n, baseline=baseline, alpha=alpha)
        return PowerResult(delta=delta, n=n, power=achieved, alpha=alpha, method="analytic")
    if method_l == "simulation":
        if n is None:
            solved_n = _power.simulate_n(
                delta, target_power=power, baseline=baseline, alpha=alpha, rho=rho,
                sims=sims, random_state=random_state,
            )
            return PowerResult(delta=delta, n=solved_n, power=power, alpha=alpha, method="simulation")
        achieved = _power.simulate_power(
            delta, n, baseline=baseline, alpha=alpha, rho=rho, sims=sims, random_state=random_state
        )
        return PowerResult(delta=delta, n=n, power=achieved, alpha=alpha, method="simulation")
    raise ValueError(f"unknown method: {method!r}")


def multi_compare(
    df,
    correction="holm",
    method="permutation",
    paired=True,
    alpha=0.05,
    n_resamples=9999,
    random_state=None,
    subset_col="subset",
):
    """Pairwise model comparisons across (optionally subset-stratified) benchmarks,
    with multiple-comparison correction across all resulting p-values."""
    validate_schema(df)
    models = sorted(df["model"].unique())
    if len(models) < 2:
        raise ValueError("need at least 2 models to compare")
    has_subset = subset_col in df.columns and df[subset_col].notna().any()
    subsets = sorted(df[subset_col].dropna().unique()) if has_subset else [None]

    rows = []
    for subset in subsets:
        for model_a, model_b in itertools.combinations(models, 2):
            a, b = to_paired_arrays(df, model_a, model_b, subset=subset, subset_col=subset_col)
            if len(a) == 0:
                continue
            res = compare(
                a, b, paired=paired, method=method, confidence=1 - alpha,
                n_resamples=n_resamples, random_state=random_state,
            )
            rows.append(
                {
                    "subset": subset,
                    "model_a": model_a,
                    "model_b": model_b,
                    "delta": res.delta,
                    "ci_low": res.ci[0],
                    "ci_high": res.ci[1],
                    "p_value": res.p_value,
                    "n": res.n,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["p_adj"] = correct_pvalues(result["p_value"].to_numpy(), method=correction)
    result["significant"] = result["p_adj"] < alpha
    if not has_subset:
        result = result.drop(columns=["subset"])
    return result


def cluster_ci(scores, clusters, statistic=np.mean, confidence=0.95, n_resamples=9999, random_state=None):
    """Clustered bootstrap CI (resamples whole clusters, e.g. repeated decodes or
    grouped questions, rather than individual items)."""
    scores = np.asarray(scores, dtype=float)
    clusters = np.asarray(clusters)
    if len(scores) != len(clusters):
        raise ValueError("scores and clusters must be the same length")
    rng = np.random.default_rng(random_state)
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    grouped = [scores[clusters == c] for c in unique_clusters]
    overall = float(statistic(scores))

    boot_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        resampled = np.concatenate([grouped[c] for c in chosen])
        boot_stats[i] = statistic(resampled)
    alpha = 1 - confidence
    lo, hi = np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CIResult(
        estimate=overall, lower=float(lo), upper=float(hi),
        method="cluster_bootstrap", n=len(scores), confidence=confidence,
    )
