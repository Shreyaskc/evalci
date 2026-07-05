"""The per-item results schema shared by multi_compare(), cluster_ci(), and adapters."""
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("item_id", "model", "score")
OPTIONAL_COLUMNS = ("subset", "sample_idx")
PER_ITEM_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


def validate_schema(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("expected a pandas DataFrame with columns "
                         f"{REQUIRED_COLUMNS} (+ optional {OPTIONAL_COLUMNS})")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"per-item DataFrame is missing required column(s): {missing}")


def to_paired_arrays(df, model_a, model_b, subset=None, subset_col="subset"):
    """Pivot a per-item DataFrame to two aligned score arrays for model_a/model_b.

    Multiple samples per item (`sample_idx`) are averaged into a single per-item
    score before pairing.
    """
    validate_schema(df)
    sub = df if subset is None or subset_col not in df.columns else df[df[subset_col] == subset]
    pivot = sub.pivot_table(index="item_id", columns="model", values="score", aggfunc="mean")
    missing = [m for m in (model_a, model_b) if m not in pivot.columns]
    if missing:
        raise ValueError(f"model(s) not found in DataFrame: {missing}")
    paired = pivot[[model_a, model_b]].dropna()
    return paired[model_a].to_numpy(dtype=float), paired[model_b].to_numpy(dtype=float)


def from_records(item_ids, models, scores, subsets=None, sample_idxs=None):
    """Build a per-item DataFrame from parallel arrays (used by adapters)."""
    n = len(item_ids)
    data = {
        "item_id": list(item_ids),
        "model": list(models) if not isinstance(models, str) else [models] * n,
        "score": np.asarray(scores, dtype=float),
    }
    data["subset"] = list(subsets) if subsets is not None else [None] * n
    data["sample_idx"] = list(sample_idxs) if sample_idxs is not None else [0] * n
    return pd.DataFrame(data)
