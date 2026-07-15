"""
generate_behavioural_figures.py
================================
Rebuild Figure 1 (main paper) with two panels using real is/are numbers,
and build Appendix Figure A1 with attraction asymmetry across all four
auxiliary paradigms.

Data source: behavioral_analysis_outputs/behavioral_summary_all_verbs.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SUMMARY_DIR = Path("results/behavioural/summary")
OUT_MAIN = Path("figures/paper_main")
OUT_BEHAV = Path("figures/behavioural")

OUT_MAIN.mkdir(parents=True, exist_ok=True)
OUT_BEHAV.mkdir(parents=True, exist_ok=True)

CSV = SUMMARY_DIR / "behavioral_summary_all_verbs.csv"
df = pd.read_csv(CSV)

MODELS = ["Phi-2", "Llama-3.2-3B", "Qwen2.5-3B"]
MODEL_COLORS = {
    "Phi-2":        "#5e4fa2",
    "Llama-3.2-3B": "#3288bd",
    "Qwen2.5-3B":   "#d53e4f",
}
VERB_LABELS = {
    "be_present":   "is / are",
    "be_past":      "was / were",
    "have_present": "has / have",
    "do_present":   "does / do",
}
VERB_ORDER = ["be_present", "be_past", "have_present", "do_present"]

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

# =============================================================================
# Figure 1 (main paper): two panels for is/are only
#   (a) SS/SP/PS/PP margins per model
#   (b) plural-attractor vs singular-attractor cost per model
# =============================================================================

isare = df[df["verb_pair"] == "be_present"].copy()

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))

# --- Panel (a): four-condition margins ---
ax = axes[0]
ax.yaxis.grid(True, alpha=0.18, linestyle="-", linewidth=0.6)
ax.xaxis.grid(False)
conds = ["SS", "SP", "PS", "PP"]
x = np.arange(len(conds))
bar_w = 0.25

for i, m in enumerate(MODELS):
    row = isare[isare["model"] == m].iloc[0]
    vals = [row["SS_mean"], row["SP_mean"], row["PS_mean"], row["PP_mean"]]
    ci_lo = [row["SS_ci_lo"], row["SP_ci_lo"], row["PS_ci_lo"], row["PP_ci_lo"]]
    ci_hi = [row["SS_ci_hi"], row["SP_ci_hi"], row["PS_ci_hi"], row["PP_ci_hi"]]
    err = [[v - lo for v, lo in zip(vals, ci_lo)],
           [hi - v for v, hi in zip(vals, ci_hi)]]
    ax.bar(x + (i - 1) * bar_w, vals, bar_w,
           color=MODEL_COLORS[m], alpha=0.9,
           edgecolor="white", linewidth=0.5,
           yerr=err, capsize=3,
           error_kw={"elinewidth": 0.9, "ecolor": "0.3"},
           label=m,zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(conds, fontsize=10)
ax.set_ylabel("Verb-margin Δ (logit units)", fontsize=10)
ax.set_title("(a) Agreement margins",
             fontsize=11, fontweight="bold")
# Add small space below zero so bars do not visually merge with the x-axis.
ax.set_ylim(-0.25, 8)

# Draw zero baseline inside the plot.
ax.axhline(0, color="0.2", linewidth=0.9, zorder=5)

# Hide bottom spine because the zero line now acts as the baseline.
ax.spines["bottom"].set_visible(False)

# --- Panel (b): attractor cost decomposition ---
ax = axes[1]
ax.yaxis.grid(True, alpha=0.18, linestyle="-", linewidth=0.6)
ax.xaxis.grid(False)
cost_labels = ["Plural-attractor cost\n(SS − SP)",
               "Singular-attractor cost\n(PP − PS)"]
x = np.arange(len(cost_labels))
bar_w = 0.25

for i, m in enumerate(MODELS):
    row = isare[isare["model"] == m].iloc[0]
    vals = [row["plural_attr_mean"], row["singular_attr_mean"]]
    ci_lo = [row["plural_attr_ci_lo"], row["singular_attr_ci_lo"]]
    ci_hi = [row["plural_attr_ci_hi"], row["singular_attr_ci_hi"]]
    err = [[v - lo for v, lo in zip(vals, ci_lo)],
           [hi - v for v, hi in zip(vals, ci_hi)]]
    ax.bar(x + (i - 1) * bar_w, vals, bar_w,
           color=MODEL_COLORS[m], alpha=0.9,
           edgecolor="white", linewidth=0.5,
           yerr=err, capsize=3,
           error_kw={"elinewidth": 0.9, "ecolor": "0.3"},
           label=m,zorder=3)

    # Annotate asymmetry value above the plural bar
    asym = row["asymmetry_mean"]
    ax.text(
    0 + (i - 1) * bar_w,
    vals[0] + err[1][0] + 0.12,
    f"A={asym:.2f}",
    ha="center",
    va="bottom",
    fontsize=7.5,
    color=MODEL_COLORS[m],
    fontweight="bold",
)

ax.set_xticks(x)
ax.set_xticklabels(cost_labels, fontsize=9)
ax.set_ylabel("Cost (logit units)", fontsize=10)
ax.set_title("(b) Attractor costs",
             fontsize=11, fontweight="bold")
# Add small space below zero so bars do not visually merge with the x-axis.
ax.set_ylim(-0.18, 5.5)

# Draw zero baseline inside the plot.
ax.axhline(0, color="0.2", linewidth=0.9, zorder=5)

# Hide bottom spine because the zero line now acts as the baseline.
ax.spines["bottom"].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.04),
    fontsize=8.5,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_MAIN / "figure1_behavioural.pdf", bbox_inches="tight")
fig.savefig(OUT_MAIN / "figure1_behavioural.png", bbox_inches="tight")
plt.close(fig)
print("Saved figure1_behavioural.pdf")


# =============================================================================
# Appendix Figure A1: attraction asymmetry across all 4 verb pairs
# =============================================================================

fig, ax = plt.subplots(figsize=(9, 4))
ax.yaxis.grid(True, alpha=0.18, linestyle="-", linewidth=0.6)
ax.xaxis.grid(False)

x = np.arange(len(VERB_ORDER))
bar_w = 0.25

for i, m in enumerate(MODELS):
    vals = []
    ci_lo = []
    ci_hi = []
    for vp in VERB_ORDER:
        row = df[(df["model"] == m) & (df["verb_pair"] == vp)].iloc[0]
        vals.append(row["asymmetry_mean"])
        ci_lo.append(row["asymmetry_ci_lo"])
        ci_hi.append(row["asymmetry_ci_hi"])
    err = [[v - lo for v, lo in zip(vals, ci_lo)],
           [hi - v for v, hi in zip(vals, ci_hi)]]
    ax.bar(x + (i - 1) * bar_w, vals, bar_w,
           color=MODEL_COLORS[m], alpha=0.9,
           edgecolor="white", linewidth=0.9,
           yerr=err, capsize=3.5,
           error_kw={"elinewidth": 0.9, "ecolor": "0.3"},
           label=m,zorder=3)

    # Value labels on top
    for xi, v in zip(x, vals):
        if v >= 0:
            y = v + 0.15
            va = "bottom"
        else:
            y = v - 0.18
            va = "top"

        ax.text(
            xi + (i - 1) * bar_w,
            y,
            f"{v:.2f}",
            ha="center",
            va=va,
            fontsize=7.5,
            color=MODEL_COLORS[m],
            fontweight="bold",
        )

ax.axhline(0, color="0.2", linewidth=0.9, zorder=5)
ax.set_xticks(x)
ax.set_xticklabels([VERB_LABELS[vp] for vp in VERB_ORDER],
                   fontsize=10, style="italic")
ax.set_ylabel("Asymmetry: (SS − SP) − (PP − PS)", fontsize=10)
ax.set_title("Attraction asymmetry across auxiliary paradigms",
             fontsize=11.5, fontweight="bold")
ax.legend(fontsize=9, loc="upper right", frameon=False)
ax.set_ylim(-0.8, 3.2)

plt.tight_layout()
fig.savefig(OUT_BEHAV / "figureA1_asymmetry.pdf", bbox_inches="tight")
fig.savefig(OUT_BEHAV / "figureA1_asymmetry.png", bbox_inches="tight")
plt.close(fig)
print("Saved figureA1_asymmetry.pdf")


# =============================================================================
# Print Table A1 LaTeX rows (for pasting into the appendix)
# =============================================================================
print()
print("---- Appendix Table A1 LaTeX rows ----")
for m in MODELS:
    print(f"\\multicolumn{{9}}{{l}}{{\\textit{{{m}}}}} \\\\")
    for vp in VERB_ORDER:
        r = df[(df["model"] == m) & (df["verb_pair"] == vp)].iloc[0]
        print(f"  {VERB_LABELS[vp]} & "
              f"{r['SS_mean']:.2f} & {r['SP_mean']:.2f} & "
              f"{r['PS_mean']:.2f} & {r['PP_mean']:.2f} & "
              f"{r['plural_attr_mean']:.2f} & {r['singular_attr_mean']:.2f} & "
              f"{r['asymmetry_mean']:.2f} & {int(r['n_items'])} \\\\")
    print("  \\midrule")
