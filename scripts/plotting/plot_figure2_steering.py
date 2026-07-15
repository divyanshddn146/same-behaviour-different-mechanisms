"""
plot_figure2_steering.py
=========================
Generate Figure 2: main cross-model steering profile.

Figure shows signed steering effects for the subject-number direction
at three injection positions: subject, distractor, final.

Uses the main is/are SP steering results:
  target condition: SP
  alpha: +1
  layers: 1+
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

OUT = Path("figures/paper_main")
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Style: clean research-grade PDF
# ---------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------
# Data: main is/are steering results
# Values are signed effects, layers 1+ averaged.
# CIs are bootstrap 95% intervals.
# ---------------------------------------------------------------------

MODELS = ["Phi-2", "Llama-3.2-3B", "Qwen2.5-3B"]
POSITIONS = ["Subject", "Distractor", "Final"]

# These values are copied from:
# results/steering/is_are/summary/steering_bootstrap_phi2.csv
# results/steering/is_are/summary/steering_bootstrap_llama32_3b.csv
# results/steering/is_are/summary/steering_bootstrap_qwen25_3b.csv
#
# Rows used:
# direction == "subject"
# position in {"subject", "distractor", "final"}
# alpha == 1.0
# layers == "1+"

DATA = {
    "Phi-2": {
        "Subject":   (2.846, 2.790, 2.905),
        "Distractor": (0.363, 0.348, 0.379),
        "Final":     (3.223, 3.171, 3.275),
        "ratio": "0.88×",
    },
    "Llama-3.2-3B": {
        "Subject":   (3.110, 3.058, 3.160),
        "Distractor": (0.282, 0.273, 0.292),
        "Final":     (3.147, 3.115, 3.180),
        "ratio": "0.99×",
    },
    "Qwen2.5-3B": {
        "Subject":   (4.827, 4.724, 4.930),
        "Distractor": (0.106, 0.093, 0.119),
        "Final":     (1.29, 1.251, 1.322),
        "ratio": "3.76×",
    },
}

# Position colors, intentionally different from behavioural model colors.
POSITION_COLORS = {
    "Subject": "#009E73",     # green / teal
    "Distractor": "#E69F00",  # amber
    "Final": "#4D4D4D",       # charcoal
}


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    1, 3,
    figsize=(11.2, 3.45),
    sharey=True,
)

x = np.arange(len(POSITIONS))
bar_w = 0.62

global_max = max(
    DATA[m][p][2]
    for m in MODELS
    for p in POSITIONS
)
ymax = global_max + 0.55

for ax, model in zip(axes, MODELS):
    vals = np.array([DATA[model][p][0] for p in POSITIONS])
    ci_lo = np.array([DATA[model][p][1] for p in POSITIONS])
    ci_hi = np.array([DATA[model][p][2] for p in POSITIONS])

    yerr = np.vstack([vals - ci_lo, ci_hi - vals])
    colors = [POSITION_COLORS[p] for p in POSITIONS]

    ax.yaxis.grid(True, alpha=0.13, linestyle="-", linewidth=0.55, zorder=0)
    ax.xaxis.grid(False)

    ax.bar(
        x,
        vals,
        width=bar_w,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        yerr=yerr,
        capsize=3.2,
        error_kw={
            "elinewidth": 0.9,
            "ecolor": "0.25",
            "capthick": 0.9,
        },
        zorder=3,
    )

    # Label subject and final values only: cleaner than labelling all bars.
    for xi, pos, val, hi in zip(x, POSITIONS, vals, ci_hi):
        if pos in {"Subject", "Final"}:
            ax.text(
                xi,
                hi + 0.10,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.7,
                fontweight="bold",
                color="0.20",
            )

    # S/F ratio annotation
    ratio = DATA[model]["ratio"]
    ratio_color = "0.20" if model != "Qwen2.5-3B" else POSITION_COLORS["Subject"]
    ax.text(
        0.5,
        0.93,
        f"S/F = {ratio}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=ratio_color,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="0.82",
            linewidth=0.6,
            alpha=0.95,
        ),
    )

    ax.set_title(model, fontsize=10.8, fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(POSITIONS, fontsize=9)
    ax.set_ylim(-0.18, ymax)
    ax.axhline(0, color="0.20", linewidth=0.9, zorder=5)
    ax.spines["bottom"].set_visible(False)

axes[0].set_ylabel("Signed steering effect", fontsize=10)

# Shared legend
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, fc=POSITION_COLORS["Subject"], label="Subject"),
    plt.Rectangle((0, 0), 1, 1, fc=POSITION_COLORS["Distractor"], label="Distractor"),
    plt.Rectangle((0, 0), 1, 1, fc=POSITION_COLORS["Final"], label="Final"),
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.04),
    fontsize=8.8,
)

fig.tight_layout(rect=[0, 0, 1, 0.95])

fig.savefig(OUT / "figure2_steering_main.pdf", bbox_inches="tight")
fig.savefig(OUT / "figure2_steering_main.png", bbox_inches="tight")
plt.close(fig)

print(f"Saved: {OUT / 'figure2_steering_main.pdf'}")
print(f"Saved: {OUT / 'figure2_steering_main.png'}")