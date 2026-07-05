"""Case study for the evalci arXiv paper: how many adjacent-rank gaps on a
public leaderboard survive a significance test?

Data source: Table 2 ("Performance of finetuned Llama 3 models on key
benchmark evaluations") of Grattafiori et al. 2024, "The Llama 3 Herd of
Models", arXiv:2407.21783 -- MMLU 5-shot accuracy for 9 instruction-tuned
models, all evaluated in the same table. MMLU test-set size (n=14,042) is
from Hendrycks et al. 2021, "Measuring Massive Multitask Language
Understanding", arXiv:2009.03300.

Per-item correctness is not public for these models, so we reconstruct, for
each model, an i.i.d. binary vector of length n consistent with its reported
aggregate accuracy (k = round(accuracy * n) ones, the rest zeros). This is a
conservative approximation: a true paired-by-item test, if per-item data were
available, would only be *more* powerful than the unpaired test used here
(under the realistic assumption that items have positively-correlated
difficulty across models), so the count of non-significant adjacent gaps
below is an upper bound, not an underestimate.
"""
import numpy as np
import pandas as pd

import evalci
from evalci.schema import from_records

MMLU_N = 14042  # Hendrycks et al. 2021

MODELS = {
    "Llama 3 8B": 0.694,
    "GPT-3.5 Turbo": 0.707,
    "Mixtral 8x22B": 0.769,
    "Nemotron-4 340B": 0.826,
    "Llama 3 70B": 0.836,
    "GPT-4 (0125)": 0.851,
    "Llama 3 405B": 0.873,
    "GPT-4o": 0.891,
    "Claude 3.5 Sonnet": 0.899,
}


def build_per_item_frame():
    frames = []
    for model, acc in MODELS.items():
        k = round(acc * MMLU_N)
        scores = np.array([1.0] * k + [0.0] * (MMLU_N - k))
        frames.append(from_records(range(MMLU_N), model, scores))
    return pd.concat(frames, ignore_index=True)


def main():
    df = build_per_item_frame()
    ranked = sorted(MODELS.items(), key=lambda kv: kv[1])

    table = evalci.multi_compare(
        df, correction="holm", method="permutation", paired=False,
        n_resamples=9999, random_state=0,
    )

    print("All C(9,2)=36 pairwise comparisons, Holm-corrected (family-wise across all 36):")
    print(evalci.report(table, format="markdown"))
    print()

    print("Adjacent-rank pairs only (the headline figure):")
    adjacent_rows = []
    for (model_a, acc_a), (model_b, acc_b) in zip(ranked, ranked[1:]):
        row = table[
            ((table.model_a == model_a) & (table.model_b == model_b))
            | ((table.model_a == model_b) & (table.model_b == model_a))
        ].iloc[0]
        adjacent_rows.append(row)
    adjacent = pd.DataFrame(adjacent_rows)
    print(evalci.report(adjacent, format="markdown"))

    n_not_significant = int((~adjacent["significant"]).sum())
    print(f"\n{n_not_significant}/{len(adjacent)} adjacent-rank gaps are NOT significant "
          f"at alpha=0.05 after Holm correction across all 36 pairwise comparisons.")

    adjacent.to_csv("paper/figures/adjacent_rank_gaps.csv", index=False)
    table.to_csv("paper/figures/all_pairwise_comparisons.csv", index=False)


if __name__ == "__main__":
    main()
