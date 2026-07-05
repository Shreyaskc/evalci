"""Load per-item results from a HELM `per_instance_stats.json` file into the evalci schema.

Expects the standard HELM per-instance-stats structure: a JSON list of records
each with an "instance_id" and a "stats" list of {"name": {"name": <metric>},
"mean": <value>, ...} entries. If your HELM run uses a different metric name,
pass `metric_key` explicitly.
"""
import json
from pathlib import Path

from ..schema import from_records

DEFAULT_METRIC_KEYS = ("exact_match", "quasi_exact_match", "f1_score")


def _stat_value(stat):
    if "mean" in stat and stat["mean"] is not None:
        return float(stat["mean"])
    if stat.get("count") == 1 and "sum" in stat:
        return float(stat["sum"])
    raise ValueError(f"could not extract a scalar value from stat record: {stat}")


def _extract_score(record, metric_key):
    stats = record.get("stats", [])
    names = [s.get("name", {}).get("name") for s in stats]
    if metric_key is not None:
        for s, name in zip(stats, names):
            if name == metric_key:
                return _stat_value(s)
        raise ValueError(f"metric_key={metric_key!r} not found; metrics present: {names}")
    for key in DEFAULT_METRIC_KEYS:
        for s, name in zip(stats, names):
            if name == key:
                return _stat_value(s)
    raise ValueError(
        f"could not find a metric in stats (tried {DEFAULT_METRIC_KEYS}); "
        f"pass metric_key explicitly. metrics present: {names}"
    )


def load(path, model=None, subset=None, metric_key=None):
    """Parse a HELM per_instance_stats.json file into a per-item DataFrame."""
    path = Path(path)
    model = model or path.stem
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a JSON list of per-instance stat records")
    if not records:
        raise ValueError(f"{path}: no records found")

    item_ids, scores = [], []
    for r in records:
        item_id = r.get("instance_id", r.get("instance", {}).get("id"))
        item_ids.append(item_id)
        scores.append(_extract_score(r, metric_key))

    subsets = [subset] * len(item_ids)
    return from_records(item_ids, model, scores, subsets=subsets)
