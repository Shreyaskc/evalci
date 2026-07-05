import json

import pytest

from evalci.cli import main


def _write_lm_eval(path, p, n=200, seed=0):
    import random

    rnd = random.Random(seed)
    samples = {"task": [{"doc_id": i, "acc": 1.0 if rnd.random() < p else 0.0} for i in range(n)]}
    path.write_text(json.dumps({"samples": samples}))


def test_cli_compare_basic(tmp_path, capsys):
    a = tmp_path / "model_a.json"
    b = tmp_path / "model_b.json"
    _write_lm_eval(a, 0.85, seed=1)
    _write_lm_eval(b, 0.55, seed=2)

    rc = main(["compare", str(a), str(b), "--n-resamples", "1000"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Δ=" in out
    assert "p=" in out


def test_cli_compare_mcnemar(tmp_path, capsys):
    a = tmp_path / "model_a.json"
    b = tmp_path / "model_b.json"
    _write_lm_eval(a, 0.8, seed=3)
    _write_lm_eval(b, 0.8, seed=3)  # identical seed -> identical scores -> delta ~ 0

    rc = main(["compare", str(a), str(b), "--method", "mcnemar"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "mcnemar" in out


def test_cli_compare_no_overlap_errors(tmp_path, capsys):
    a = tmp_path / "model_a.json"
    b = tmp_path / "model_b.json"
    a.write_text(json.dumps({"samples": {"task": [{"doc_id": "x1", "acc": 1.0}]}}))
    b.write_text(json.dumps({"samples": {"task": [{"doc_id": "x2", "acc": 1.0}]}}))

    rc = main(["compare", str(a), str(b)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "no overlapping item_ids" in err


def test_cli_compare_label_collision_errors(tmp_path, capsys):
    a = tmp_path / "same.json"
    a.write_text(json.dumps({"samples": {"task": [{"doc_id": 0, "acc": 1.0}]}}))
    # Two different files but same resolved model label (both default to path stem "same")
    import shutil

    subdir = tmp_path / "sub"
    subdir.mkdir()
    b = subdir / "same.json"
    shutil.copy(a, b)

    rc = main(["compare", str(a), str(b)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "same model label" in err


def test_cli_compare_explicit_model_labels(tmp_path, capsys):
    a = tmp_path / "same.json"
    b = tmp_path / "same.json"  # same path is nonsensical but labels still forced distinct
    _write_lm_eval(a, 0.8, seed=4)

    rc = main(["compare", str(a), str(a), "--model-a", "A", "--model-b", "B", "--n-resamples", "500"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Δ=0.000" in out  # identical file compared to itself


def test_cli_compare_csv_format_explicit(tmp_path, capsys):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("item_id,score\n" + "\n".join(f"{i},{1.0 if i % 2 == 0 else 0.0}" for i in range(20)))
    b.write_text("item_id,score\n" + "\n".join(f"{i},{1.0 if i % 3 == 0 else 0.0}" for i in range(20)))

    rc = main(["compare", str(a), str(b), "--format", "csv", "--n-resamples", "500"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Δ=" in out


def test_cli_unknown_format_raises(tmp_path):
    path = tmp_path / "mystery.data"
    path.write_text("not json")
    with pytest.raises(ValueError):
        main(["compare", str(path), str(path)])
