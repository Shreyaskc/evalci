import numpy as np
import pytest

from evalci.report import report
from evalci.stats import CIResult, compare, ci, multi_compare
from evalci.schema import from_records
import pandas as pd


def test_report_ci_result_string():
    result = ci(np.array([1.0] * 80 + [0.0] * 20), method="wilson")
    text = report(result)
    assert "0.800" in text
    assert "95% CI" in text
    assert "n=100" in text


def test_report_compare_result_has_stars_for_significant():
    rng = np.random.default_rng(0)
    a = (rng.random(300) < 0.9).astype(float)
    b = (rng.random(300) < 0.5).astype(float)
    result = compare(a, b, method="permutation", n_resamples=1000, random_state=0)
    text = report(result)
    assert "Δ=" in text
    assert "p=" in text
    assert "*" in text


def test_report_compare_no_stars_when_disabled():
    rng = np.random.default_rng(0)
    a = (rng.random(300) < 0.9).astype(float)
    b = (rng.random(300) < 0.5).astype(float)
    result = compare(a, b, method="permutation", n_resamples=1000, random_state=0)
    text = report(result, stars=False)
    assert text.count("*") == 0


def test_report_table_markdown():
    rng = np.random.default_rng(1)
    frames = [
        from_records(range(100), m, (rng.random(100) < p).astype(float), subsets=["t"] * 100)
        for m, p in {"a": 0.9, "b": 0.5}.items()
    ]
    df = pd.concat(frames, ignore_index=True)
    mc = multi_compare(df, n_resamples=500, random_state=0)
    text = report(mc, format="markdown")
    assert text.startswith("|")
    assert "model_a" in text
    assert "sig" in text


def test_report_table_latex():
    rng = np.random.default_rng(1)
    frames = [
        from_records(range(100), m, (rng.random(100) < p).astype(float), subsets=["t"] * 100)
        for m, p in {"a": 0.9, "b": 0.5}.items()
    ]
    df = pd.concat(frames, ignore_index=True)
    mc = multi_compare(df, n_resamples=500, random_state=0)
    text = report(mc, format="latex")
    assert r"\begin{tabular}" in text
    assert r"\end{tabular}" in text


def test_report_unknown_type_raises():
    with pytest.raises(TypeError):
        report(object())
