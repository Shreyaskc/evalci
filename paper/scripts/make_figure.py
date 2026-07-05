"""Generates the case-study figure: MMLU accuracy with 95% Wilson CIs for 9
public models, with the 3 statistically-indistinguishable adjacent pairs
(after Holm correction) highlighted. Run after leaderboard_case_study.py."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evalci

INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
TIE_BAND = "#fab219"

MMLU_N = 14042
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


def main():
    ranked = sorted(MODELS.items(), key=lambda kv: kv[1])
    names = [m for m, _ in ranked]
    cis = []
    for _, acc in ranked:
        k = round(acc * MMLU_N)
        scores = np.array([1.0] * k + [0.0] * (MMLU_N - k))
        cis.append(evalci.ci(scores, method="wilson"))

    adjacent = pd.read_csv("paper/figures/adjacent_rank_gaps.csv")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    y = np.arange(len(names))

    for yi, c in zip(y, cis):
        ax.plot([c.lower, c.upper], [yi, yi], color=BLUE, linewidth=1.5, solid_capstyle="round")
        ax.plot(c.estimate, yi, "o", color=BLUE, markersize=6, zorder=3)

    # shade the 3 statistically-indistinguishable adjacent pairs
    for i in range(len(names) - 1):
        row = adjacent[
            ((adjacent.model_a == names[i]) & (adjacent.model_b == names[i + 1]))
            | ((adjacent.model_a == names[i + 1]) & (adjacent.model_b == names[i]))
        ].iloc[0]
        if not row["significant"]:
            ax.axhspan(i - 0.4, i + 1.4, color=TIE_BAND, alpha=0.15, zorder=0)
            ax.text(
                0.905, i + 0.5, f"not significant\n(p={row['p_adj']:.2g})",
                fontsize=7.5, color=INK_MUTED, style="italic", va="center",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9, color=INK_PRIMARY)
    ax.set_xlabel("MMLU (5-shot) accuracy, with 95% Wilson CI", fontsize=9, color=INK_MUTED)
    ax.set_xlim(0.65, 0.95)
    ax.tick_params(axis="x", labelsize=8, colors=INK_MUTED)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.7, zorder=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_title(
        "3 of 8 adjacent leaderboard ranks are not significantly different",
        fontsize=10.5, color=INK_PRIMARY, loc="left", pad=10,
    )
    fig.tight_layout()
    fig.savefig("paper/figures/leaderboard_cis.pdf")
    print("wrote paper/figures/leaderboard_cis.pdf")


if __name__ == "__main__":
    main()
