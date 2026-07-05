import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from evalci._correction import benjamini_hochberg, correct_pvalues, holm


@pytest.mark.parametrize(
    "pvalues",
    [
        [0.01, 0.04, 0.03, 0.5],
        [0.001, 0.2, 0.03, 0.049, 0.6, 0.0001],
        [0.5],
        [0.05, 0.05, 0.05],
    ],
)
def test_holm_matches_statsmodels(pvalues):
    got = holm(pvalues)
    _, expected, _, _ = multipletests(pvalues, method="holm")
    np.testing.assert_allclose(got, expected, atol=1e-9)


@pytest.mark.parametrize(
    "pvalues",
    [
        [0.01, 0.04, 0.03, 0.5],
        [0.001, 0.2, 0.03, 0.049, 0.6, 0.0001],
        [0.5],
        [0.05, 0.05, 0.05],
    ],
)
def test_bh_matches_statsmodels(pvalues):
    got = benjamini_hochberg(pvalues)
    _, expected, _, _ = multipletests(pvalues, method="fdr_bh")
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_correct_pvalues_dispatch():
    p = [0.01, 0.2, 0.03]
    np.testing.assert_allclose(correct_pvalues(p, "holm"), holm(p))
    np.testing.assert_allclose(correct_pvalues(p, "bh"), benjamini_hochberg(p))


def test_correct_pvalues_unknown_method_raises():
    with pytest.raises(ValueError):
        correct_pvalues([0.1, 0.2], method="not-a-method")


def test_holm_never_less_conservative_than_bonferroni_for_min_pvalue():
    p = [0.001, 0.2, 0.03, 0.049]
    holm_adj = holm(p)
    bonferroni = np.clip(np.asarray(p) * len(p), 0, 1)
    assert holm_adj[np.argmin(p)] == pytest.approx(bonferroni[np.argmin(p)])
