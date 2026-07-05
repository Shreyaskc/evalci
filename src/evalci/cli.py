"""`evalci compare results_a.json results_b.json` — CLI entry point."""
import argparse
import json
import sys
from pathlib import Path

from . import adapters
from .schema import to_paired_arrays
from .stats import compare
from .report import report


def _sniff_format(path):
    path = Path(path)
    if path.suffix == ".csv":
        return "csv"
    if path.suffix == ".jsonl":
        return "lm-eval-harness"
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return "csv"
        if isinstance(data, dict) and "samples" in data:
            return "lm-eval-harness"
        if isinstance(data, list):
            return "helm"
    raise ValueError(f"could not auto-detect format for {path}; pass --format explicitly")


def _load(path, fmt, model, metric_key):
    fmt = fmt or _sniff_format(path)
    if fmt == "lm-eval-harness":
        return adapters.load_lm_eval_harness(path, model=model, metric_key=metric_key)
    if fmt == "helm":
        return adapters.load_helm(path, model=model, metric_key=metric_key)
    if fmt == "csv":
        return adapters.load_csv(path, model=model)
    raise ValueError(f"unknown format: {fmt!r}")


def build_parser():
    parser = argparse.ArgumentParser(prog="evalci")
    sub = parser.add_subparsers(dest="command", required=True)

    cmp_p = sub.add_parser("compare", help="Compare two models' per-item results")
    cmp_p.add_argument("results_a")
    cmp_p.add_argument("results_b")
    cmp_p.add_argument("--format", choices=["lm-eval-harness", "helm", "csv"], default=None)
    cmp_p.add_argument("--method", choices=["permutation", "bootstrap", "mcnemar"], default="permutation")
    cmp_p.add_argument("--unpaired", action="store_true", help="treat samples as independent, not item-paired")
    cmp_p.add_argument("--metric-key", default=None)
    cmp_p.add_argument("--confidence", type=float, default=0.95)
    cmp_p.add_argument("--n-resamples", type=int, default=9999)
    cmp_p.add_argument("--output-format", choices=["markdown", "latex"], default="markdown")
    cmp_p.add_argument("--model-a", default=None, help="label for results_a (default: filename)")
    cmp_p.add_argument("--model-b", default=None, help="label for results_b (default: filename)")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "compare":
        df_a = _load(args.results_a, args.format, args.model_a, args.metric_key)
        df_b = _load(args.results_b, args.format, args.model_b, args.metric_key)
        model_a, model_b = df_a["model"].iloc[0], df_b["model"].iloc[0]
        if model_a == model_b:
            print(
                f"error: both inputs resolved to the same model label {model_a!r}; "
                "pass --model-a/--model-b to disambiguate",
                file=sys.stderr,
            )
            return 1
        import pandas as pd

        merged = pd.concat([df_a, df_b], ignore_index=True)
        a, b = to_paired_arrays(merged, model_a, model_b)
        if len(a) == 0:
            print(f"error: no overlapping item_ids between {args.results_a} and {args.results_b}", file=sys.stderr)
            return 1
        result = compare(
            a, b, paired=not args.unpaired, method=args.method,
            confidence=args.confidence, n_resamples=args.n_resamples,
        )
        print(report(result, format=args.output_format))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
