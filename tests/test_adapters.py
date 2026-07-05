import json

import pytest

from evalci.adapters import csv as csv_adapter
from evalci.adapters import helm, lm_eval_harness


def test_lm_eval_harness_single_json(tmp_path):
    path = tmp_path / "results.json"
    samples = {"mmlu": [{"doc_id": i, "acc": 1.0 if i % 2 == 0 else 0.0} for i in range(10)]}
    path.write_text(json.dumps({"samples": samples, "config": {"model": "hf"}}))

    df = lm_eval_harness.load(path, model="my-model")
    assert len(df) == 10
    assert set(df["model"]) == {"my-model"}
    assert set(df["subset"]) == {"mmlu"}
    assert list(df["score"]) == [1.0 if i % 2 == 0 else 0.0 for i in range(10)]


def test_lm_eval_harness_jsonl(tmp_path):
    path = tmp_path / "samples_gsm8k_2024.jsonl"
    with path.open("w") as f:
        for i in range(5):
            f.write(json.dumps({"doc_id": i, "exact_match": float(i % 2)}) + "\n")

    df = lm_eval_harness.load(path, model="m")
    assert len(df) == 5
    assert set(df["subset"]) == {"gsm8k"}


def test_lm_eval_harness_default_model_is_stem(tmp_path):
    path = tmp_path / "my_run.json"
    samples = {"t": [{"doc_id": 0, "acc": 1.0}]}
    path.write_text(json.dumps({"samples": samples}))
    df = lm_eval_harness.load(path)
    assert df["model"].iloc[0] == "my_run"


def test_lm_eval_harness_missing_samples_key_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"nope": []}))
    with pytest.raises(ValueError):
        lm_eval_harness.load(path)


def test_lm_eval_harness_custom_metric_key(tmp_path):
    path = tmp_path / "r.json"
    samples = {"t": [{"doc_id": i, "custom_metric": float(i)} for i in range(3)]}
    path.write_text(json.dumps({"samples": samples}))
    df = lm_eval_harness.load(path, metric_key="custom_metric")
    assert list(df["score"]) == [0.0, 1.0, 2.0]


def test_lm_eval_harness_unknown_metric_raises(tmp_path):
    path = tmp_path / "r.json"
    samples = {"t": [{"doc_id": 0, "weird_key": 1.0}]}
    path.write_text(json.dumps({"samples": samples}))
    with pytest.raises(ValueError):
        lm_eval_harness.load(path)


def test_helm_per_instance_stats(tmp_path):
    path = tmp_path / "per_instance_stats.json"
    records = [
        {"instance_id": f"id{i}", "stats": [{"name": {"name": "exact_match"}, "mean": float(i % 2), "count": 1}]}
        for i in range(8)
    ]
    path.write_text(json.dumps(records))

    df = helm.load(path, model="my-model", subset="scenario1")
    assert len(df) == 8
    assert set(df["model"]) == {"my-model"}
    assert set(df["subset"]) == {"scenario1"}


def test_helm_default_model_uses_stem(tmp_path):
    path = tmp_path / "helm_run_a.json"
    records = [{"instance_id": "id0", "stats": [{"name": {"name": "exact_match"}, "mean": 1.0, "count": 1}]}]
    path.write_text(json.dumps(records))
    df = helm.load(path)
    assert df["model"].iloc[0] == "helm_run_a"


def test_helm_two_files_in_same_dir_get_distinct_default_labels(tmp_path):
    for name, val in [("run_a.json", 1.0), ("run_b.json", 0.0)]:
        path = tmp_path / name
        records = [{"instance_id": "id0", "stats": [{"name": {"name": "exact_match"}, "mean": val, "count": 1}]}]
        path.write_text(json.dumps(records))
    df_a = helm.load(tmp_path / "run_a.json")
    df_b = helm.load(tmp_path / "run_b.json")
    assert df_a["model"].iloc[0] != df_b["model"].iloc[0]


def test_helm_missing_metric_raises(tmp_path):
    path = tmp_path / "r.json"
    records = [{"instance_id": "id0", "stats": [{"name": {"name": "other_metric"}, "mean": 1.0}]}]
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError):
        helm.load(path)


def test_csv_adapter(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text("item_id,score,subset\n1,1.0,s\n2,0.0,s\n")
    df = csv_adapter.load(path, model="m")
    assert len(df) == 2
    assert set(df["model"]) == {"m"}


def test_csv_adapter_requires_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError):
        csv_adapter.load(path)
