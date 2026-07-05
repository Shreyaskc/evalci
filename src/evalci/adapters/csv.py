"""Load a plain per-item CSV (columns: item_id, score[, subset, sample_idx]) into the evalci schema."""
from pathlib import Path

import pandas as pd

from ..schema import from_records


def load(path, model=None):
    path = Path(path)
    model = model or path.stem
    df = pd.read_csv(path)
    if "item_id" not in df.columns or "score" not in df.columns:
        raise ValueError(f"{path}: CSV must have 'item_id' and 'score' columns, got {list(df.columns)}")
    return from_records(
        df["item_id"],
        model,
        df["score"],
        subsets=df["subset"] if "subset" in df.columns else None,
        sample_idxs=df["sample_idx"] if "sample_idx" in df.columns else None,
    )
