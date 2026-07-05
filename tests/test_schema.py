import numpy as np
import pandas as pd
import pytest

from evalci.schema import from_records, to_paired_arrays, validate_schema


def test_validate_schema_requires_dataframe():
    with pytest.raises(TypeError):
        validate_schema([1, 2, 3])


def test_validate_schema_requires_columns():
    df = pd.DataFrame({"item_id": [1, 2], "model": ["a", "a"]})
    with pytest.raises(ValueError):
        validate_schema(df)


def test_from_records_broadcasts_scalar_model():
    df = from_records([1, 2, 3], "gpt-4", [1.0, 0.0, 1.0])
    assert (df["model"] == "gpt-4").all()
    assert list(df["score"]) == [1.0, 0.0, 1.0]
    assert list(df["sample_idx"]) == [0, 0, 0]


def test_to_paired_arrays_aligns_on_item_id():
    df = pd.concat(
        [
            from_records([1, 2, 3], "a", [1.0, 0.0, 1.0]),
            from_records([2, 3, 4], "b", [1.0, 1.0, 0.0]),
        ],
        ignore_index=True,
    )
    a, b = to_paired_arrays(df, "a", "b")
    # only items 2 and 3 are present for both models
    assert len(a) == 2
    assert len(b) == 2


def test_to_paired_arrays_averages_repeated_samples():
    df = from_records([1, 1, 2, 2], "a", [1.0, 0.0, 1.0, 1.0], sample_idxs=[0, 1, 0, 1])
    df_b = from_records([1, 2], "b", [1.0, 0.0])
    merged = pd.concat([df, df_b], ignore_index=True)
    a, b = to_paired_arrays(merged, "a", "b")
    assert sorted(a) == [0.5, 1.0]


def test_to_paired_arrays_missing_model_raises():
    df = from_records([1, 2], "a", [1.0, 0.0])
    with pytest.raises(ValueError):
        to_paired_arrays(df, "a", "nonexistent")


def test_to_paired_arrays_filters_by_subset():
    df = pd.concat(
        [
            from_records([1, 2], "a", [1.0, 0.0], subsets=["s1", "s2"]),
            from_records([1, 2], "b", [0.0, 1.0], subsets=["s1", "s2"]),
        ],
        ignore_index=True,
    )
    a, b = to_paired_arrays(df, "a", "b", subset="s1")
    assert len(a) == 1
    assert a[0] == 1.0 and b[0] == 0.0
