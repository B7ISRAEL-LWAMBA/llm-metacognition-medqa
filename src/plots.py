"""The three figures. Each takes the DataFrame from metrics.load_results()."""

import matplotlib.pyplot as plt
import numpy as np

from metrics import reliability_bins, risk_coverage

# Colorblind-checked categorical palette; marker and line style give each
# condition a second cue besides color.
CONDITIONS = {
    "A": {"label": "A: verbalized", "color": "#2a78d6", "marker": "o", "ls": "-"},
    "B": {"label": "B: self-consistency", "color": "#eb6834", "marker": "s", "ls": "--"},
    "C": {"label": "C: critic", "color": "#1baf7a", "marker": "^", "ls": ":"},
}
CORRECT_COLOR, WRONG_COLOR = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})


def _clean(df):
    return df[~df["parse_error"].astype(bool)]


def reliability_diagram(df, path=None):
    df = _clean(df)
    models = sorted(df["model_key"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.6), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="-", label="perfect calibration", zorder=1)
        for cond, style in CONDITIONS.items():
            g = df[(df["model_key"] == model) & (df["condition"] == cond)]
            if g.empty:
                continue
            bins = reliability_bins(g["is_correct"].astype(int), g["confidence"])
            # Marker area scales with the number of answers in the bin.
            ax.plot(bins["confidence"], bins["accuracy"], color=style["color"], ls=style["ls"], lw=2,
                    label=style["label"], zorder=2)
            ax.scatter(bins["confidence"], bins["accuracy"], s=12 + 2 * bins["n"], color=style["color"],
                       marker=style["marker"], edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
        ax.set(title=model, xlabel="Confidence (bin mean)", xlim=(0, 1.02), ylim=(0, 1.02))
    axes[0][0].set_ylabel("Accuracy in bin")
    axes[0][0].legend(loc="upper left")
    fig.suptitle("Reliability diagram (marker size = answers in bin)", x=0.01, ha="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def confidence_by_correctness(df, path=None):
    df = _clean(df)
    models = sorted(df["model_key"].unique())
    conds = [c for c in CONDITIONS if c in set(df["condition"])]
    fig, axes = plt.subplots(len(models), len(conds), figsize=(3.6 * len(conds), 2.8 * len(models)),
                             sharex=True, squeeze=False)
    edges = np.linspace(0, 1, 21)
    for i, model in enumerate(models):
        for j, cond in enumerate(conds):
            ax = axes[i][j]
            g = df[(df["model_key"] == model) & (df["condition"] == cond)]
            right = g.loc[g["is_correct"].astype(bool), "confidence"]
            wrong = g.loc[~g["is_correct"].astype(bool), "confidence"]
            # Share of each group, so a small "wrong" group is still visible.
            for vals, color, label in [(right, CORRECT_COLOR, f"correct (n={len(right)})"),
                                       (wrong, WRONG_COLOR, f"wrong (n={len(wrong)})")]:
                if len(vals):
                    ax.hist(vals, bins=edges, weights=np.ones(len(vals)) / len(vals), color=color,
                            alpha=0.55, edgecolor="#fcfcfb", linewidth=1, label=label)
            ax.set_title(f"{model} · {CONDITIONS[cond]['label']}", fontsize=9)
            ax.legend(fontsize=8, loc="upper left")
            if j == 0:
                ax.set_ylabel("Share of group")
            if i == len(models) - 1:
                ax.set_xlabel("Confidence")
    fig.suptitle("Confidence on right vs wrong answers (separation = metacognitive sensitivity)",
                 x=0.01, ha="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def risk_coverage_plot(df, path=None):
    df = _clean(df)
    models = sorted(df["model_key"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.4), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        for cond, style in CONDITIONS.items():
            g = df[(df["model_key"] == model) & (df["condition"] == cond)]
            if g.empty:
                continue
            rc = risk_coverage(g["is_correct"].astype(int), g["confidence"])
            ax.plot(rc["coverage"], rc["accuracy"], color=style["color"], ls=style["ls"], lw=2,
                    marker=style["marker"], markersize=5, markeredgecolor="#fcfcfb", label=style["label"])
        ax.set(title=model, xlabel="Coverage (share answered by the model)", xlim=(0, 1.02))
        ax.invert_xaxis()  # left to right: the model defers more and more to a human
    axes[0][0].set_ylabel("Accuracy on answers kept")
    axes[0][0].legend(loc="lower left")
    fig.suptitle("Risk-coverage: deferring low-confidence answers to a clinician", x=0.01, ha="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
