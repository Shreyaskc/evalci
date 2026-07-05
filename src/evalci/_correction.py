"""Multiple-comparison correction: Holm and Benjamini-Hochberg."""
import numpy as np


def holm(pvalues):
    """Holm (1979) step-down Bonferroni correction. Returns adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running_max = 0.0
    for i, idx in enumerate(order):
        running_max = max(running_max, (n - i) * p[idx])
        adj[idx] = min(running_max, 1.0)
    return adj


def benjamini_hochberg(pvalues):
    """Benjamini-Hochberg (1995) FDR correction. Returns adjusted (q) values."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(n)
    running_min = 1.0
    for i, idx in enumerate(order):
        rank = n - i
        running_min = min(running_min, p[idx] * n / rank)
        adj[idx] = running_min
    return np.clip(adj, 0, 1)


def correct_pvalues(pvalues, method="holm"):
    method = method.lower()
    if method in ("holm", "bonferroni-holm"):
        return holm(pvalues)
    if method in ("bh", "benjamini-hochberg", "fdr_bh"):
        return benjamini_hochberg(pvalues)
    if method == "bonferroni":
        p = np.asarray(pvalues, dtype=float)
        return np.clip(p * len(p), 0, 1)
    raise ValueError(f"unknown correction method: {method!r}")
