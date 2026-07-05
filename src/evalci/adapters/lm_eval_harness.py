"""Load per-item results from lm-evaluation-harness output into the evalci schema.

Supports both the `--log_samples` JSONL format (`samples_<task>_*.jsonl`, one
JSON record per line) and the older single-JSON format with a top-level
"samples" field. If your harness version uses different metric key names, pass
`metric_key` explicitly.
"""
import json
from pathlib import Path

from ..schema import from_records

DEFAULT_METRIC_KEYS = ("acc", "exact_match", "acc_norm", "score")


def _extract_score(record, metric_key):
    if metric_key is not None:
        if metric_key not in record:
            raise ValueError(f"metric_key={metric_key!r} not found in sample record: {list(record.keys())}")
        return float(record[metric_key])
    for key in DEFAULT_METRIC_KEYS:
        if key in record:
            return float(record[key])
    raise ValueError(
        f"could not find a metric in sample record (tried {DEFAULT_METRIC_KEYS}); "
        f"pass metric_key explicitly. keys present: {list(record.keys())}"
    )


def load(path, model=None, task=None, metric_key=None):
    """Parse an lm-evaluation-harness results file into a per-item DataFrame."""
    path = Path(path)
    model = model or path.stem
    records = []

    if path.suffix == ".jsonl":
        task_name = task or path.stem.replace("samples_", "").split("_2")[0]
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                records.append((task_name, json.loads(line)))
    else:
        data = json.loads(path.read_text())
        samples = data.get("samples")
        if samples is None:
            raise ValueError(f"{path}: no 'samples' field found; is this an lm-eval-harness output file?")
        if isinstance(samples, dict):
            for task_name, items in samples.items():
                if task is not None and task_name != task:
                    continue
                records.extend((task_name, r) for r in items)
        elif isinstance(samples, list):
            records.extend((task or "task", r) for r in samples)
        else:
            raise ValueError(f"{path}: unrecognized 'samples' structure: {type(samples)}")

    if not records:
        raise ValueError(f"{path}: no sample records found (task filter={task!r})")

    item_ids, subsets, sample_idxs, scores = [], [], [], []
    for task_name, r in records:
        item_ids.append(r.get("doc_id", r.get("idx", r.get("doc_hash"))))
        subsets.append(task_name)
        sample_idxs.append(r.get("repeat_idx", 0))
        scores.append(_extract_score(r, metric_key))

    return from_records(item_ids, model, scores, subsets=subsets, sample_idxs=sample_idxs)
