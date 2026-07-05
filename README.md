# evalci

Statistically sound comparisons between LLMs on benchmarks: confidence
intervals on accuracy, paired significance tests, power analysis, clustered
standard errors for multi-sample decoding, and multiple-comparison correction
across many models/benchmarks — all validated against `statsmodels`/exact
enumeration fixtures.

```python
>>> import evalci
>>> result = evalci.compare(model_a_scores, model_b_scores, method="permutation")
>>> evalci.report(result)
'Δ=0.034, 95% CI [0.005, 0.063], paired permutation p=0.025*, n=1319'  # exact numbers depend on your data
```

## Status

Core library (statistics, eval-shaped workflows, adapters, CLI) is implemented
and tested. Not yet released to PyPI; no arXiv paper or DOI yet.

## Install

Not yet on PyPI. Install from source:

```bash
git clone https://github.com/Shreyaskc/evalci.git
cd evalci
pip install -e ".[test]"   # add [test] to also get pytest/statsmodels for running the test suite
```

Requires Python ≥3.9. Runtime dependencies are numpy, scipy, and pandas only.

## Usage

### Confidence interval on a single model

```python
import evalci

# binary (0/1) per-item correctness
evalci.ci(scores, method="wilson")           # Wilson score interval
evalci.ci(scores, method="clopper-pearson")  # exact interval

# continuous scores (e.g. a similarity metric)
evalci.ci(scores, method="bootstrap")        # percentile/BCa bootstrap on the mean
```

### Comparing two models on the same items

```python
result = evalci.compare(model_a_scores, model_b_scores, paired=True, method="permutation")
# result.delta, result.ci, result.p_value, result.n

evalci.compare(a, b, method="bootstrap")   # null-shifted bootstrap hypothesis test
evalci.compare(a, b, method="mcnemar")     # McNemar's test for paired binary outcomes
evalci.compare(a, b, paired=False, method="permutation")  # independent samples
```

### Sample-size / power calculator

```python
evalci.power(delta=0.03, power=0.8)          # required n to detect a 3-point gap at 80% power
evalci.power(delta=0.03, n=1500)             # achieved power at n=1500
evalci.power(delta=0.03, power=0.8, method="simulation", rho=0.3)  # correlated-items simulation
```

### Many models × many benchmarks, with correction

```python
import pandas as pd

# per-item schema: item_id, model, score, [subset], [sample_idx]
df = pd.DataFrame(...)
table = evalci.multi_compare(df, correction="holm")
print(evalci.report(table, format="markdown"))
```

### Clustered standard errors (repeated decoding, grouped questions)

```python
# clusters groups multiple samples of the same underlying item
evalci.cluster_ci(scores, clusters)
```

### Loading results from eval harnesses

```python
from evalci.adapters import load_lm_eval_harness, load_helm, load_csv

df_a = load_lm_eval_harness("results_a.json", model="model-a")
df_b = load_helm("per_instance_stats.json", model="model-b")
```

### CLI

```bash
evalci compare results_a.json results_b.json --method permutation
evalci compare results_a.json results_b.json --format helm --method mcnemar
```

Auto-detects lm-evaluation-harness / HELM / CSV format from the file
extension/content; pass `--format` to override, and `--model-a`/`--model-b` to
label the two runs explicitly.

## What's validated, and how

Statistical correctness is the point of this library, so the test suite
cross-checks every routine against an independent reference rather than just
re-testing its own math:

- Wilson and Clopper-Pearson intervals against `statsmodels.stats.proportion.proportion_confint`
- McNemar's test (exact and asymptotic) against `statsmodels.stats.contingency_tables.mcnemar`
- Holm and Benjamini-Hochberg correction against `statsmodels.stats.multitest.multipletests`
- The paired permutation test against brute-force exact enumeration of all sign flips (small n)
- Bootstrap CIs via a coverage simulation (nominal 95% CIs should contain the true parameter ~95% of the time)

`statsmodels` is a test-only dependency (`pip install -e ".[test]"`), not a
runtime dependency.

```bash
pytest tests/
```

## API surface

`evalci.ci`, `evalci.compare`, `evalci.power`, `evalci.multi_compare`,
`evalci.cluster_ci`, `evalci.report`, `evalci.adapters.{load_lm_eval_harness,
load_helm, load_csv}`.

## License

MIT — see [LICENSE](LICENSE).
