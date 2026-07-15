"""
behavioral_analysis.py
======================
Step 1: Filter clean_items_all_models_intersection.csv → clean_be_present_all_models_intersection.csv
Step 2: Compute behavioral metrics (SS/SP/PS/PP margins, attraction effects, asymmetry)
        per model, for be_present and all verb pairs.
Step 3: Save tables + generate paper-ready plots.

Run after make_dataset.py has completed.

Outputs
-------
clean_be_present_all_models_intersection.csv   ← main causal dataset
behavioral_margins_be_present.csv              ← per-item margins, be_present only
behavioral_summary_all_verbs.csv               ← summary table all verb pairs × models
behavioral_summary_be_present.csv              ← summary table be_present only
plot_margins_by_condition.png                  ← Figure 1: bar chart SS/SP/PS/PP
plot_attraction_asymmetry.png                  ← Figure 2: plural vs singular attraction
plot_margin_distributions.png                  ← Figure 3: violin/box distributions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = "agreement_dataset_outputs"
OUTPUT_DIR  = "behavioral_analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map model tags to clean display names
MODEL_DISPLAY = {
    "phi2":       "Phi-2",
    "llama32_3b": "Llama-3.2-3B",
    "qwen25_3b":  "Qwen2.5-3B",
}
MODEL_ORDER = ["phi2", "llama32_3b", "qwen25_3b"]

CONDITIONS   = ["SS", "SP", "PS", "PP"]
VERB_PAIRS   = ["be_present", "be_past", "have_present", "do_present"]
VERB_DISPLAY = {
    "be_present":   "is / are",
    "be_past":      "was / were",
    "have_present": "has / have",
    "do_present":   "does / do",
}

# Bootstrap settings
N_BOOTSTRAP   = 2000
BOOTSTRAP_SEED = 42

# ============================================================
# COLORS  (clean academic palette)
# ============================================================

COND_COLORS = {
    "SS": "#2166ac",   # blue
    "SP": "#d73027",   # red
    "PS": "#1a9850",   # green
    "PP": "#f46d43",   # orange
}

MODEL_COLORS = {
    "phi2":       "#5e4fa2",
    "llama32_3b": "#3288bd",
    "qwen25_3b":  "#d53e4f",
}

# ============================================================
# STEP 1: FILTER INTERSECTION → be_present
# ============================================================

def step1_filter_be_present():
    inter_path = os.path.join(DATASET_DIR, "clean_items_all_models_intersection.csv")
    if not os.path.exists(inter_path):
        raise FileNotFoundError(
            f"Could not find {inter_path}. "
            "Run make_dataset.py first."
        )

    inter_df = pd.read_csv(inter_path)
    print(f"Intersection CSV: {len(inter_df)} rows | "
          f"{inter_df['item_id'].nunique()} base items | "
          f"verb pairs: {inter_df['verb_pair'].unique().tolist()}")

    be_df = inter_df[inter_df["verb_pair"] == "be_present"].copy()
    n_items = be_df["item_id"].nunique()
    print(f"\nbe_present subset: {len(be_df)} rows | {n_items} base items")

    out_path = os.path.join(OUTPUT_DIR, "clean_be_present_all_models_intersection.csv")
    be_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    return inter_df, be_df


# ============================================================
# STEP 2: LOAD PER-MODEL AUDIT FILES
# ============================================================

def load_audit_files():
    """
    Load agreement_audit_{tag}.csv for each model.
    These contain per-row margins scored by each model.
    """
    audits = {}
    for tag in MODEL_ORDER:
        path = os.path.join(DATASET_DIR, f"agreement_audit_{tag}.csv")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping {tag}")
            continue
        df = pd.read_csv(path)
        audits[tag] = df
        print(f"  Loaded audit {tag}: {len(df)} rows | "
              f"{df['item_id'].nunique()} items | "
              f"verb pairs: {df['verb_pair'].unique().tolist()}")
    return audits


# ============================================================
# STEP 3: BEHAVIORAL METRICS + BOOTSTRAP
# ============================================================

def bootstrap_mean_ci(values, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED, ci=95):
    """Bootstrap CI for mean of values (array-like)."""
    rng = np.random.default_rng(seed)
    vals = np.array(values)
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return vals.mean(), lo, hi


def compute_behavioral_summary(audits: dict, verb_pair: str = None) -> pd.DataFrame:
    """
    For each model × verb_pair (or just be_present if specified):
    compute mean margins, attraction effects, asymmetry, with bootstrap CIs.

    Filtering: only use items where ALL 4 conditions pass (margin > 0)
    for that (item_id, verb_pair) in that model.
    """
    rows = []

    for tag in MODEL_ORDER:
        if tag not in audits:
            continue
        df = audits[tag].copy()

        vps = [verb_pair] if verb_pair else VERB_PAIRS

        for vp in vps:
            sub = df[df["verb_pair"] == vp].copy()
            if sub.empty:
                continue

            # Item-level filter: keep only items where all 4 conditions pass
            item_pass = sub.groupby("item_id")["pass_margin_gt_0"].all()
            clean_items = item_pass[item_pass].index
            sub = sub[sub["item_id"].isin(clean_items)]

            if sub.empty:
                continue

            n_items = sub["item_id"].nunique()

            def cond_margins(cond):
                return sub[sub["condition"] == cond]["margin"].values

            ss = cond_margins("SS")
            sp = cond_margins("SP")
            ps = cond_margins("PS")
            pp = cond_margins("PP")

            # Per-item paired differences
            # Build per-item pivot for paired bootstrap
            pivot = sub.pivot_table(
                index="item_id", columns="condition", values="margin"
            )
            pivot = pivot.dropna(subset=["SS", "SP", "PS", "PP"])

            plural_attr_per_item    = (pivot["SS"] - pivot["SP"]).values
            singular_attr_per_item  = (pivot["PP"] - pivot["PS"]).values
            asymmetry_per_item      = plural_attr_per_item - singular_attr_per_item

            # Means + CIs
            ss_mean, ss_lo, ss_hi = bootstrap_mean_ci(pivot["SS"].values)
            sp_mean, sp_lo, sp_hi = bootstrap_mean_ci(pivot["SP"].values)
            ps_mean, ps_lo, ps_hi = bootstrap_mean_ci(pivot["PS"].values)
            pp_mean, pp_lo, pp_hi = bootstrap_mean_ci(pivot["PP"].values)

            pl_mean, pl_lo, pl_hi = bootstrap_mean_ci(plural_attr_per_item)
            sg_mean, sg_lo, sg_hi = bootstrap_mean_ci(singular_attr_per_item)
            as_mean, as_lo, as_hi = bootstrap_mean_ci(asymmetry_per_item)

            # Accuracy
            ss_acc = sub[sub["condition"] == "SS"]["pass_margin_gt_0"].mean()
            sp_acc = sub[sub["condition"] == "SP"]["pass_margin_gt_0"].mean()
            ps_acc = sub[sub["condition"] == "PS"]["pass_margin_gt_0"].mean()
            pp_acc = sub[sub["condition"] == "PP"]["pass_margin_gt_0"].mean()

            rows.append({
                "model_tag":   tag,
                "model":       MODEL_DISPLAY.get(tag, tag),
                "verb_pair":   vp,
                "verb_display":VERB_DISPLAY.get(vp, vp),
                "n_items":     n_items,

                # Mean margins
                "SS_mean": ss_mean, "SS_ci_lo": ss_lo, "SS_ci_hi": ss_hi,
                "SP_mean": sp_mean, "SP_ci_lo": sp_lo, "SP_ci_hi": sp_hi,
                "PS_mean": ps_mean, "PS_ci_lo": ps_lo, "PS_ci_hi": ps_hi,
                "PP_mean": pp_mean, "PP_ci_lo": pp_lo, "PP_ci_hi": pp_hi,

                # Attraction effects (paired, per-item)
                "plural_attr_mean":   pl_mean, "plural_attr_ci_lo":   pl_lo, "plural_attr_ci_hi":   pl_hi,
                "singular_attr_mean": sg_mean, "singular_attr_ci_lo": sg_lo, "singular_attr_ci_hi": sg_hi,
                "asymmetry_mean":     as_mean, "asymmetry_ci_lo":     as_lo, "asymmetry_ci_hi":     as_hi,

                # Accuracy
                "SS_acc": ss_acc, "SP_acc": sp_acc,
                "PS_acc": ps_acc, "PP_acc": pp_acc,
            })

    return pd.DataFrame(rows)


# ============================================================
# STEP 4: PLOTS
# ============================================================

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "figure.dpi":         150,
})


def plot_margins_by_condition(summary_be: pd.DataFrame):
    """
    Figure 1: grouped bar chart showing SS/SP/PS/PP margins per model.
    One group per model, four bars per group (one per condition).
    Error bars = 95% bootstrap CI.
    """
    models   = [t for t in MODEL_ORDER if t in summary_be["model_tag"].values]
    n_models = len(models)
    n_conds  = 4
    bar_w    = 0.18
    x        = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(9, 5))

    offsets = np.linspace(-(n_conds-1)/2 * bar_w, (n_conds-1)/2 * bar_w, n_conds)

    for ci, cond in enumerate(CONDITIONS):
        means, lo_errs, hi_errs = [], [], []
        for tag in models:
            row = summary_be[summary_be["model_tag"] == tag].iloc[0]
            m   = row[f"{cond}_mean"]
            lo  = row[f"{cond}_ci_lo"]
            hi  = row[f"{cond}_ci_hi"]
            means.append(m)
            lo_errs.append(m - lo)
            hi_errs.append(hi - m)

        ax.bar(
            x + offsets[ci], means, bar_w,
            label=f"{cond}",
            color=COND_COLORS[cond],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
            yerr=[lo_errs, hi_errs],
            capsize=3,
            error_kw={"elinewidth": 1.2, "ecolor": "0.3"},
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(t, t) for t in models], fontsize=12)
    ax.set_ylabel("Mean logit margin (correct − wrong)", fontsize=11)
    ax.set_title("Agreement margins by condition (is / are)", fontsize=13, fontweight="bold")
    ax.legend(title="Condition", ncol=4, fontsize=10,
              title_fontsize=10, loc="upper right")

    # Annotate SS-SP gap
    for mi, tag in enumerate(models):
        row    = summary_be[summary_be["model_tag"] == tag].iloc[0]
        gap    = row["plural_attr_mean"]
        y_pos  = max(row["SS_mean"], row["SP_mean"]) + 0.25
        ax.annotate(
            f"Δ={gap:.2f}",
            xy=(x[mi], y_pos),
            ha="center", va="bottom",
            fontsize=8.5, color="#222",
            fontweight="bold",
        )

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plot_margins_by_condition.png")
    fig.savefig(path, bbox_inches="tight")

    paper_png = os.path.join(OUTPUT_DIR, "figure1_behavioural.png")
    paper_pdf = os.path.join(OUTPUT_DIR, "figure1_behavioural.pdf")
    fig.savefig(paper_png, bbox_inches="tight")
    fig.savefig(paper_pdf, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved: {path}")
    print(f"Saved: {paper_png}")
    print(f"Saved: {paper_pdf}")


def plot_attraction_asymmetry(summary_be: pd.DataFrame):
    """
    Figure 2: side-by-side bars showing plural attraction vs singular attraction
    per model, with CIs. Also shows asymmetry = plural − singular.
    """
    models  = [t for t in MODEL_ORDER if t in summary_be["model_tag"].values]
    x       = np.arange(len(models))
    bar_w   = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: plural vs singular attraction
    ax = axes[0]
    pl_means, pl_lo, pl_hi = [], [], []
    sg_means, sg_lo, sg_hi = [], [], []

    for tag in models:
        row = summary_be[summary_be["model_tag"] == tag].iloc[0]
        pl_means.append(row["plural_attr_mean"])
        pl_lo.append(row["plural_attr_mean"] - row["plural_attr_ci_lo"])
        pl_hi.append(row["plural_attr_ci_hi"] - row["plural_attr_mean"])
        sg_means.append(row["singular_attr_mean"])
        sg_lo.append(row["singular_attr_mean"] - row["singular_attr_ci_lo"])
        sg_hi.append(row["singular_attr_ci_hi"] - row["singular_attr_mean"])

    ax.bar(x - bar_w/2, pl_means, bar_w,
           label="Plural distractor effect\n(SS − SP)",
           color="#d73027", alpha=0.85, edgecolor="white",
           yerr=[pl_lo, pl_hi], capsize=4,
           error_kw={"elinewidth": 1.3, "ecolor": "0.3"})
    ax.bar(x + bar_w/2, sg_means, bar_w,
           label="Singular distractor effect\n(PP − PS)",
           color="#2166ac", alpha=0.85, edgecolor="white",
           yerr=[sg_lo, sg_hi], capsize=4,
           error_kw={"elinewidth": 1.3, "ecolor": "0.3"})

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(t, t) for t in models], fontsize=11)
    ax.set_ylabel("Mean attraction effect (logit units)", fontsize=11)
    ax.set_title("Distractor attraction effects", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # Right: asymmetry only
    ax2 = axes[1]
    as_means, as_lo, as_hi = [], [], []
    for tag in models:
        row = summary_be[summary_be["model_tag"] == tag].iloc[0]
        as_means.append(row["asymmetry_mean"])
        as_lo.append(row["asymmetry_mean"] - row["asymmetry_ci_lo"])
        as_hi.append(row["asymmetry_ci_hi"] - row["asymmetry_mean"])

    colors = [MODEL_COLORS.get(t, "#666") for t in models]
    bars   = ax2.bar(x, as_means, 0.45,
                     color=colors, alpha=0.85, edgecolor="white",
                     yerr=[as_lo, as_hi], capsize=4,
                     error_kw={"elinewidth": 1.3, "ecolor": "0.3"})

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_DISPLAY.get(t, t) for t in models], fontsize=11)
    ax2.set_ylabel("Asymmetry (plural − singular attraction)", fontsize=11)
    ax2.set_title("Plural-over-singular asymmetry", fontsize=12, fontweight="bold")

    # Value labels on bars
    for bar, v in zip(bars, as_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle("Behavioral agreement-attraction asymmetry  (is / are)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plot_attraction_asymmetry.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_margin_distributions(audits: dict, verb_pair: str = "be_present"):
    """
    Figure 3: box plots of per-item margins across conditions, one panel per model.
    Shows variance across items, not just means.
    """
    models = [t for t in MODEL_ORDER if t in audits]
    n      = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, tag in zip(axes, models):
        df = audits[tag]
        df = df[df["verb_pair"] == verb_pair].copy()

        # Item-level filter
        item_pass = df.groupby("item_id")["pass_margin_gt_0"].all()
        clean_items = item_pass[item_pass].index
        df = df[df["item_id"].isin(clean_items)]

        data   = [df[df["condition"] == c]["margin"].values for c in CONDITIONS]
        colors = [COND_COLORS[c] for c in CONDITIONS]

        bp = ax.boxplot(
            data,
            patch_artist=True,
            notch=False,
            widths=0.5,
            medianprops={"color": "white", "linewidth": 2},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
            flierprops={"marker": "o", "markersize": 2.5,
                        "markerfacecolor": "0.5", "alpha": 0.4},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.80)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(CONDITIONS, fontsize=11)
        ax.set_title(MODEL_DISPLAY.get(tag, tag), fontsize=12, fontweight="bold")
        ax.set_xlabel("Condition", fontsize=10)

    axes[0].set_ylabel("Logit margin (correct − wrong)", fontsize=11)
    fig.suptitle("Distribution of agreement margins by condition  (is / are)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plot_margin_distributions.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_all_verb_pairs(summary_all: pd.DataFrame):
    """
    Figure 4: asymmetry across all verb pairs and models (heatmap-style bar grid).
    Shows the behavioral effect is not is/are-specific.
    """
    models = [t for t in MODEL_ORDER if t in summary_all["model_tag"].values]
    vps    = [vp for vp in VERB_PAIRS if vp in summary_all["verb_pair"].values]
    if not vps:
        return

    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4.5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, tag in zip(axes, models):
        sub = summary_all[summary_all["model_tag"] == tag]
        sub = sub[sub["verb_pair"].isin(vps)].copy()
        sub = sub.set_index("verb_pair").reindex(vps)

        means    = sub["asymmetry_mean"].values
        lo_errs  = (sub["asymmetry_mean"] - sub["asymmetry_ci_lo"]).values
        hi_errs  = (sub["asymmetry_ci_hi"] - sub["asymmetry_mean"]).values
        labels   = [VERB_DISPLAY.get(v, v) for v in vps]
        y        = np.arange(len(vps))

        bar_colors = ["#d73027" if m > 0 else "#2166ac" for m in means]

        ax.barh(
            y, means,
            color=bar_colors, alpha=0.82, edgecolor="white",
            xerr=[lo_errs, hi_errs], capsize=3,
            error_kw={"elinewidth": 1.2, "ecolor": "0.3"},
        )
        ax.axvline(0, color="black", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_title(MODEL_DISPLAY.get(tag, tag), fontsize=12, fontweight="bold")
        ax.set_xlabel("Asymmetry (logit units)", fontsize=10)

    fig.suptitle("Plural-over-singular asymmetry across verb pairs",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plot_asymmetry_all_verb_pairs.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# STEP 5: PRINT PAPER-READY SUMMARY TO STDOUT
# ============================================================

def print_paper_summary(summary_be: pd.DataFrame):
    print("\n" + "="*80)
    print("PAPER-READY NUMBERS  (be_present / is / are)")
    print("="*80)

    header = (f"{'Model':<18} {'SS':>8} {'SP':>8} {'PS':>8} {'PP':>8} "
              f"{'SS−SP':>8} {'PP−PS':>8} {'Asym':>8} {'N':>5}")
    print(header)
    print("─" * len(header))

    for tag in MODEL_ORDER:
        sub = summary_be[summary_be["model_tag"] == tag]
        if sub.empty:
            continue
        r = sub.iloc[0]
        print(
            f"{MODEL_DISPLAY.get(tag, tag):<18} "
            f"{r['SS_mean']:>8.3f} {r['SP_mean']:>8.3f} "
            f"{r['PS_mean']:>8.3f} {r['PP_mean']:>8.3f} "
            f"{r['plural_attr_mean']:>8.3f} {r['singular_attr_mean']:>8.3f} "
            f"{r['asymmetry_mean']:>8.3f} {int(r['n_items']):>5d}"
        )

    print()
    print("Bootstrap 95% CIs (plural attraction | singular attraction | asymmetry):")
    for tag in MODEL_ORDER:
        sub = summary_be[summary_be["model_tag"] == tag]
        if sub.empty:
            continue
        r = sub.iloc[0]
        print(
            f"  {MODEL_DISPLAY.get(tag, tag):<18}  "
            f"plural [{r['plural_attr_ci_lo']:.3f}, {r['plural_attr_ci_hi']:.3f}]  "
            f"singular [{r['singular_attr_ci_lo']:.3f}, {r['singular_attr_ci_hi']:.3f}]  "
            f"asymmetry [{r['asymmetry_ci_lo']:.3f}, {r['asymmetry_ci_hi']:.3f}]"
        )

    print()
    print("Accuracy (% items correct) by condition:")
    for tag in MODEL_ORDER:
        sub = summary_be[summary_be["model_tag"] == tag]
        if sub.empty:
            continue
        r = sub.iloc[0]
        print(
            f"  {MODEL_DISPLAY.get(tag, tag):<18}  "
            f"SS={r['SS_acc']*100:.1f}%  SP={r['SP_acc']*100:.1f}%  "
            f"PS={r['PS_acc']*100:.1f}%  PP={r['PP_acc']*100:.1f}%"
        )

    print()
    print("One-liner (fill in after running):")
    parts = []
    for tag in MODEL_ORDER:
        sub = summary_be[summary_be["model_tag"] == tag]
        if sub.empty:
            continue
        r    = sub.iloc[0]
        name = MODEL_DISPLAY.get(tag, tag)
        parts.append(f"{r['plural_attr_mean']:.2f} for {name}")
    print(
        "  Plural distractors substantially reduce singular-subject agreement margins: "
        "SS − SP is " + ", ".join(parts) + "."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*80)
    print("STEP 1: Filter intersection → be_present")
    print("="*80)
    inter_df, be_df = step1_filter_be_present()

    print("\n" + "="*80)
    print("STEP 2: Load per-model audit files")
    print("="*80)
    audits = load_audit_files()

    if not audits:
        print("\nNo audit files found. Run make_dataset.py first.")
        return

    print("\n" + "="*80)
    print("STEP 3: Compute behavioral metrics (be_present)")
    print("="*80)
    summary_be = compute_behavioral_summary(audits, verb_pair="be_present")
    summary_be.to_csv(
        os.path.join(OUTPUT_DIR, "behavioral_summary_be_present.csv"), index=False
    )
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'behavioral_summary_be_present.csv')}")

    print("\n" + "="*80)
    print("STEP 3b: Compute behavioral metrics (all verb pairs)")
    print("="*80)
    summary_all = compute_behavioral_summary(audits, verb_pair=None)
    summary_all.to_csv(
        os.path.join(OUTPUT_DIR, "behavioral_summary_all_verbs.csv"), index=False
    )
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'behavioral_summary_all_verbs.csv')}")

    # Per-item margins for be_present (for appendix / supplementary)
    margin_rows = []
    for tag in MODEL_ORDER:
        if tag not in audits:
            continue
        df = audits[tag]
        df = df[(df["verb_pair"] == "be_present")].copy()
        df["model_tag"] = tag
        df["model"]     = MODEL_DISPLAY.get(tag, tag)
        margin_rows.append(df[["model_tag", "model", "item_id", "condition",
                                "prompt", "margin", "pass_margin_gt_0",
                                "pass_margin_gt_1"]])
    if margin_rows:
        pd.concat(margin_rows).to_csv(
            os.path.join(OUTPUT_DIR, "behavioral_margins_be_present.csv"), index=False
        )

    print("\n" + "="*80)
    print("STEP 4: Generate plots")
    print("="*80)
    if not summary_be.empty:
        plot_margins_by_condition(summary_be)
        plot_attraction_asymmetry(summary_be)
        plot_margin_distributions(audits)

    if not summary_all.empty:
        plot_all_verb_pairs(summary_all)

    print("\n" + "="*80)
    print("STEP 5: Paper-ready summary")
    print("="*80)
    if not summary_be.empty:
        print_paper_summary(summary_be)

    print("\n" + "="*80)
    print("DONE — outputs in:", OUTPUT_DIR)
    print("="*80)
    print("Files produced:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(path)
        print(f"  {f:<55} {size:>8,} bytes")


if __name__ == "__main__":
    main()