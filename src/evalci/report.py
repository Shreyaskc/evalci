"""Render ci()/compare()/multi_compare() results as Markdown or LaTeX."""
import pandas as pd

from .stats import CIResult, CompareResult


def _stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _fmt(x, precision):
    return f"{x:.{precision}f}"


def _report_ci(result, precision):
    return (
        f"{_fmt(result.estimate, precision)} "
        f"[{int(result.confidence * 100)}% CI {_fmt(result.lower, precision)}, "
        f"{_fmt(result.upper, precision)}], n={result.n}"
    )


def _report_compare(result, stars, precision):
    star = f"{_stars(result.p_value)}" if stars else ""
    kind = "paired" if result.paired else "unpaired"
    return (
        f"Δ={_fmt(result.delta, precision)}, "
        f"{int(result.confidence * 100)}% CI [{_fmt(result.ci[0], precision)}, {_fmt(result.ci[1], precision)}], "
        f"{kind} {result.method} p={result.p_value:.4g}{star}, n={result.n}"
    )


def _markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _latex_table(headers, rows):
    lines = [r"\begin{tabular}{" + "l" * len(headers) + "}", r"\toprule",
              " & ".join(headers) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(str(c) for c in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _report_table(df, format, stars, precision):
    has_subset = "subset" in df.columns
    headers = (["subset"] if has_subset else []) + ["model_a", "model_b", "delta", "ci", "p_adj"]
    if stars:
        headers.append("sig")
    rows = []
    for _, r in df.iterrows():
        row = []
        if has_subset:
            row.append(r["subset"])
        row.append(r["model_a"])
        row.append(r["model_b"])
        row.append(_fmt(r["delta"], precision))
        row.append(f"[{_fmt(r['ci_low'], precision)}, {_fmt(r['ci_high'], precision)}]")
        row.append(f"{r['p_adj']:.4g}")
        if stars:
            row.append(_stars(r["p_adj"]))
        rows.append(row)
    if format == "latex":
        return _latex_table(headers, rows)
    return _markdown_table(headers, rows)


def report(result, format="markdown", stars=True, precision=3):
    """Render a CIResult, CompareResult, or multi_compare() DataFrame."""
    if isinstance(result, pd.DataFrame):
        return _report_table(result, format, stars, precision)
    if isinstance(result, CompareResult):
        return _report_compare(result, stars, precision)
    if isinstance(result, CIResult):
        return _report_ci(result, precision)
    raise TypeError(f"don't know how to report a {type(result)!r}")
