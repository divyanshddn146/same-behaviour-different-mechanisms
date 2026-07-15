#!/usr/bin/env python3
"""
plot_figure3_github_final.py
============================
Create the paper Figure 3 from experiment output CSV files only.

Panels
------
(a) All-position causal profile from main all-position steering outputs.
(b) Distance-stress source/final ratios from distance all-position attenuation outputs.
(c) Robustness source/final ratios from original, held-out, was/were, and anchor outputs.

This script is intended for the public repository. It does not hard-code the
reported robustness values; it reads them from the CSVs produced by the
experiment scripts and writes a small derived CSV with the plotted ratios.

Expected default inputs, relative to repository root
----------------------------------------------------
Panel (a):
  results/all_position_steering/is_are/summary/
    test3_allpos_position_profile_phi2.csv
    test3_allpos_position_profile_llama32_3b.csv
    test3_allpos_position_profile_qwen25_3b.csv

Panel (b):
  results/distance_stress/attenuation/is_are/allpos_key_ratios.csv

Panel (c):
  results/steering/is_are/layerwise/
    steering_summary_by_layer_phi2.csv
    steering_summary_by_layer_llama32_3b.csv
    steering_summary_by_layer_qwen25_3b.csv

  results/heldout_steering/is_are/
    robust_split_phi2_summary.csv
    robust_split_llama32_3b_summary.csv
    robust_split_qwen25_3b_summary.csv

  results/steering/was_were/layerwise/
    steering_summary_by_layer_phi2.csv
    steering_summary_by_layer_llama32_3b.csv
    steering_summary_by_layer_qwen25_3b.csv

  results/anchor_robustness/is_are/ratios/
    anchor_steering_ratios_phi2.csv
    anchor_steering_ratios_llama32_3b.csv
    anchor_steering_ratios_qwen25_3b.csv

Outputs
-------
  figures/paper_main/figure3_localization_robustness.pdf
  figures/paper_main/figure3_localization_robustness.png
  figures/paper_main/figure3_panel_a_allpos_from_csv.csv
  figures/paper_main/figure3_panel_b_distance_from_csv.csv
  figures/paper_main/figure3_panel_c_ratios_from_csv.csv
  figures/paper_main/figure3_input_manifest.txt

Usage
-----
python scripts/plotting/plot_figure3_localization_robustness.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model and plotting configuration
# ---------------------------------------------------------------------------

MODEL_ORDER = ["phi2", "llama32_3b", "qwen25_3b"]
MODEL_LABELS = {
    "phi2": "Phi-2",
    "llama32_3b": "Llama-3.2-3B",
    "qwen25_3b": "Qwen2.5-3B",
}

# Model colours: use the same family as the behavioural/grouped-bar figures.
MODEL_COLORS = {
    "phi2": "#6C63B5",       # purple
    "llama32_3b": "#4C78A8", # blue
    "qwen25_3b": "#E4576B",  # red/pink
}

# Token-position colours for panel (a), matching the steering-position scheme.
POSITION_COLORS = {
    "subject": "#009E73",       # green/teal
    "of": "#8BCFB0",            # light green, contextual carry-forward
    "distractor": "#E69F00",    # amber
    "final": "#4D4D4D",         # charcoal
    "other": "#C7C7C7",         # neutral grey
}

CANONICAL_ROLES = ["The", "subject", "of", "the", "distractor", "in", "final"]

# Fallback mapping when only summary-by-layer files exist.
MODEL_POSITION_MAP = {
    "phi2": {
        "The": 0,
        "subject": 1,
        "of": 2,
        "the": 3,
        "distractor": 4,
        "in": 5,
        "final": 6,
    },
    # Llama has a BOS offset in these outputs.
    "llama32_3b": {
        "The": 1,
        "subject": 2,
        "of": 3,
        "the": 4,
        "distractor": 5,
        "in": 6,
        "final": 7,
    },
    "qwen25_3b": {
        "The": 0,
        "subject": 1,
        "of": 2,
        "the": 3,
        "distractor": 4,
        "in": 5,
        "final": 6,
    },
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def require_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected file for {description}, got: {path}")
    return path


def require_dir(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Expected directory for {description}, got: {path}")
    return path


def first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists() and p.is_file():
            return p
    return None


def safe_div(num: float, den: float) -> float:
    if abs(den) < 1e-12:
        return float("nan")
    return float(num / den)


def normalize_model_tag(x: str) -> str:
    s = str(x).strip()
    aliases = {
        "phi": "phi2",
        "phi-2": "phi2",
        "Phi-2": "phi2",
        "llama": "llama32_3b",
        "Llama": "llama32_3b",
        "Llama-3.2-3B": "llama32_3b",
        "qwen": "qwen25_3b",
        "Qwen": "qwen25_3b",
        "Qwen2.5-3B": "qwen25_3b",
    }
    return aliases.get(s, s)


# ---------------------------------------------------------------------------
# Panel (a): all-position profile
# ---------------------------------------------------------------------------

def _role_from_profile_row(row: pd.Series, model_tag: str) -> str | None:
    """Map a position-profile row to one of the seven canonical roles."""
    role = str(row.get("role", "")).strip()
    token_string = str(row.get("token_string", "")).strip()
    token_pos = int(row["token_pos"])

    # Prefer explicit role labels where available.
    if role in {"subject", "distractor", "final"}:
        return role

    # Function words are explicit in token_string in recovered profiles.
    low_tok = token_string.lower()
    if low_tok == "the":
        # There are two "the" positions; disambiguate by model-specific position.
        if token_pos == MODEL_POSITION_MAP[model_tag]["The"]:
            return "The"
        if token_pos == MODEL_POSITION_MAP[model_tag]["the"]:
            return "the"
    if low_tok == "of":
        return "of"
    if low_tok == "in":
        return "in"

    # Fall back to canonical token_pos mapping.
    reverse = {v: k for k, v in MODEL_POSITION_MAP[model_tag].items()}
    return reverse.get(token_pos)


def read_position_profile(main_allpos_dir: Path, model_tag: str) -> pd.DataFrame:
    """
    Read one model's all-position profile.

    Preferred input:
      test3_allpos_position_profile_{model}.csv

    Fallback input:
      test3_allpos_summary_by_layer_{model}.csv
      In the fallback case, layer-averaged means over layer > 0 are computed.
    """
    profile_path = main_allpos_dir / f"test3_allpos_position_profile_{model_tag}.csv"
    if profile_path.exists():
        df = pd.read_csv(profile_path)
        required = {"token_pos", "signed_effect_mean"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{profile_path} missing columns: {sorted(missing)}")

        rows = []
        for _, row in df.iterrows():
            canonical = _role_from_profile_row(row, model_tag)
            if canonical not in CANONICAL_ROLES:
                continue
            rows.append({
                "model_tag": model_tag,
                "canonical_role": canonical,
                "value": float(row["signed_effect_mean"]),
                "ci_lo": float(row.get("signed_effect_ci_lo", np.nan)),
                "ci_hi": float(row.get("signed_effect_ci_hi", np.nan)),
                "n_items": int(row.get("n_items", 1)),
                "token_pos": int(row.get("token_pos", -1)),
                "source_file": str(profile_path),
            })
        out = pd.DataFrame(rows)
    else:
        by_layer_path = main_allpos_dir / f"test3_allpos_summary_by_layer_{model_tag}.csv"
        require_file(by_layer_path, f"all-position profile or by-layer summary for {model_tag}")
        df = pd.read_csv(by_layer_path)
        required = {"token_pos", "layer", "signed_effect_mean"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{by_layer_path} missing columns: {sorted(missing)}")

        df = df[df["layer"] > 0].copy()
        rows = []
        for role in CANONICAL_ROLES:
            pos = MODEL_POSITION_MAP[model_tag][role]
            sub = df[df["token_pos"] == pos].copy()
            if sub.empty:
                raise ValueError(f"No rows for {model_tag} role={role} token_pos={pos} in {by_layer_path}")
            # Some positions can have multiple rows per layer due to tokenisation.
            # Pick the row with most items per layer, then average across layers.
            if "n_items" in sub.columns:
                sub = (
                    sub.sort_values("n_items", ascending=False)
                       .drop_duplicates(subset="layer", keep="first")
                       .sort_values("layer")
                )
            rows.append({
                "model_tag": model_tag,
                "canonical_role": role,
                "value": float(sub["signed_effect_mean"].mean()),
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "source_file": str(by_layer_path),
            })
        out = pd.DataFrame(rows)

    # Ensure exactly one row per canonical role, in order.
    if out.empty:
        raise ValueError(f"Could not extract canonical all-position rows for {model_tag}")

    # Deduplicate tokenizer variants by choosing the row with the most items.
    # This is important for models such as Phi-2, where rare multi-subtoken
    # subjects/distractors can create extra rows with the same canonical role.
    # We do NOT choose the largest effect, because that would overweight rare
    # tokenization variants.
    if "n_items" not in out.columns:
        out["n_items"] = 1
    out = (
        out.sort_values(["model_tag", "canonical_role", "n_items"], ascending=[True, True, False])
           .drop_duplicates(subset=["model_tag", "canonical_role"], keep="first")
    )

    missing_roles = [r for r in CANONICAL_ROLES if r not in set(out["canonical_role"])]
    if missing_roles:
        raise ValueError(f"Missing all-position roles for {model_tag}: {missing_roles}")

    out["canonical_role"] = pd.Categorical(out["canonical_role"], CANONICAL_ROLES, ordered=True)
    return out.sort_values("canonical_role").reset_index(drop=True)


def read_all_position_data(main_allpos_dir: Path) -> pd.DataFrame:
    require_dir(main_allpos_dir, "main all-position output directory")
    return pd.concat(
        [read_position_profile(main_allpos_dir, tag) for tag in MODEL_ORDER],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Panel (b): distance ratios
# ---------------------------------------------------------------------------

def read_distance_ratios(path: Path) -> pd.DataFrame:
    require_file(path, "distance ratio CSV")
    df = pd.read_csv(path)

    if "model_name" in df.columns:
        model_col = "model_name"
    elif "model_tag" in df.columns:
        model_col = "model_tag"
    else:
        raise ValueError(f"{path} must contain model_name or model_tag column")

    if "distance" not in df.columns:
        raise ValueError(f"{path} must contain distance column")

    if "subject_div_final" in df.columns:
        ratio_col = "subject_div_final"
    elif "subject_final_ratio" in df.columns:
        ratio_col = "subject_final_ratio"
    else:
        raise ValueError(f"{path} must contain subject_div_final or subject_final_ratio column")

    out = df[[model_col, "distance", ratio_col]].copy()
    out.columns = ["model_tag", "distance", "ratio"]
    out["model_tag"] = out["model_tag"].map(normalize_model_tag)
    out = out[out["model_tag"].isin(MODEL_ORDER)].copy()

    # The short condition is omitted because distractor/final coincide in that setup.
    out = out[out["distance"].isin(["medium", "long", "extra_long", "extra long"])].copy()
    out["distance"] = out["distance"].replace({"extra long": "extra_long"})

    distance_order = ["medium", "long", "extra_long"]
    out["distance"] = pd.Categorical(out["distance"], distance_order, ordered=True)
    out = out.sort_values(["distance", "model_tag"]).reset_index(drop=True)

    # Check all 3x3 entries are present.
    expected = {(m, d) for m in MODEL_ORDER for d in distance_order}
    got = {(r.model_tag, str(r.distance)) for r in out.itertuples()}
    missing = sorted(expected - got)
    if missing:
        raise ValueError(f"Missing distance ratio entries in {path}: {missing}")

    return out


# ---------------------------------------------------------------------------
# Panel (c): robustness ratios from real CSVs
# ---------------------------------------------------------------------------

def ratio_from_steering_summary_by_layer(path: Path, model_tag: str) -> Tuple[float, str]:
    """Compute subject/final ratio from main steering_summary_by_layer CSV."""
    require_file(path, f"steering summary by layer for {model_tag}")
    df = pd.read_csv(path)
    required = {"direction", "position", "layer", "alpha", "real_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    sub = df[
        (df["direction"] == "subject")
        & (df["position"].isin(["subject", "final"]))
        & (df["alpha"].astype(float) == 1.0)
        & (df["layer"].astype(int) > 0)
    ].copy()
    if sub.empty:
        raise ValueError(f"No subject-direction alpha=+1 layer>0 rows in {path}")

    means = sub.groupby("position")["real_mean"].mean()
    if "subject" not in means.index or "final" not in means.index:
        raise ValueError(f"Missing subject or final rows after filtering {path}")
    ratio = safe_div(float(means["subject"]), float(means["final"]))
    return ratio, str(path)


def _find_robust_summary_file(heldout_dir: Path, model_tag: str) -> Path:
    """
    Find held-out summary CSV for a model.

    Historical outputs sometimes have subdirectory names robust_split_llama/qwen,
    but summary filenames are not always model-specific. We therefore inspect
    candidate files and choose the one whose model_tag column matches.
    """
    require_dir(heldout_dir, "held-out robustness directory")

    candidates: List[Path] = []
    # Direct common names.
    candidates.extend(heldout_dir.glob(f"**/*{model_tag}*summary*.csv"))
    # Historical robust split folders/files.
    candidates.extend(heldout_dir.glob("**/robust_split*summary*.csv"))
    candidates.extend(heldout_dir.glob("**/*summary.csv"))

    seen = set()
    unique = []
    for p in candidates:
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        unique.append(p)

    for p in unique:
        try:
            df = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        if "model_tag" in df.columns and model_tag in set(df["model_tag"].astype(str)):
            return p

    raise FileNotFoundError(
        f"Could not find held-out summary for model_tag={model_tag} under {heldout_dir}. "
        f"Looked at {len(unique)} candidate summary files."
    )


def ratio_from_position_summary(path: Path, model_tag: str, value_col: str = "real_mean") -> Tuple[float, str]:
    """Compute subject/final ratio from a summary CSV with one row per position."""
    require_file(path, f"position summary for {model_tag}")
    df = pd.read_csv(path)
    required = {"position", value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    if "model_tag" in df.columns:
        df = df[df["model_tag"].astype(str) == model_tag].copy()
    if "alpha" in df.columns:
        df = df[df["alpha"].astype(float) == 1.0].copy()
    if "layers" in df.columns:
        df = df[df["layers"].astype(str) == "1+"].copy()
    if "summary_level" in df.columns:
        df = df[df["summary_level"].astype(str) == "overall"].copy()

    sub = df[df["position"].isin(["subject", "final"])].copy()
    if sub.empty:
        raise ValueError(f"No subject/final rows in {path}")

    means = sub.groupby("position")[value_col].mean()
    if "subject" not in means.index or "final" not in means.index:
        raise ValueError(f"Missing subject or final rows in {path}")

    ratio = safe_div(float(means["subject"]), float(means["final"]))
    return ratio, str(path)


def ratio_from_anchor_file(anchor_dir: Path, model_tag: str) -> Tuple[float, str]:
    path = anchor_dir / f"anchor_steering_ratios_{model_tag}.csv"
    require_file(path, f"anchor ratio CSV for {model_tag}")
    df = pd.read_csv(path)
    if "subject_final_ratio" in df.columns:
        row = df.iloc[0]
        return float(row["subject_final_ratio"]), str(path)
    # If a ratio file is not available but a summary file is present, fall back to summary.
    summary_path = anchor_dir / f"anchor_steering_summary_{model_tag}.csv"
    return ratio_from_position_summary(summary_path, model_tag, value_col="real_mean")


def read_robustness_ratios(
    main_steering_dir: Path,
    heldout_dir: Path,
    waswere_steering_dir: Path,
    anchor_dir: Path,
) -> pd.DataFrame:
    require_dir(main_steering_dir, "main steering output directory")
    require_dir(heldout_dir, "held-out steering output directory")
    require_dir(waswere_steering_dir, "was/were steering output directory")
    require_dir(anchor_dir, "anchor robustness output directory")

    rows = []
    for model_tag in MODEL_ORDER:
        # Original: main is/are targeted steering.
        p = main_steering_dir / f"steering_summary_by_layer_{model_tag}.csv"
        ratio, src = ratio_from_steering_summary_by_layer(p, model_tag)
        rows.append({"condition": "Original", "model_tag": model_tag, "ratio": ratio, "source_file": src})

        # Held-out: train/test split steering outputs.
        heldout_path = _find_robust_summary_file(heldout_dir, model_tag)
        ratio, src = ratio_from_position_summary(heldout_path, model_tag, value_col="real_mean")
        rows.append({"condition": "Held-out", "model_tag": model_tag, "ratio": ratio, "source_file": src})

        # was/were: targeted steering replication.
        p = waswere_steering_dir / f"steering_summary_by_layer_{model_tag}.csv"
        ratio, src = ratio_from_steering_summary_by_layer(p, model_tag)
        rows.append({"condition": "was/were", "model_tag": model_tag, "ratio": ratio, "source_file": src})

        # Anchors: all-four-condition anchor robustness ratio.
        ratio, src = ratio_from_anchor_file(anchor_dir, model_tag)
        rows.append({"condition": "Anchors", "model_tag": model_tag, "ratio": ratio, "source_file": src})

    out = pd.DataFrame(rows)
    condition_order = ["Original", "Held-out", "was/were", "Anchors"]
    out["condition"] = pd.Categorical(out["condition"], condition_order, ordered=True)
    out["model_tag"] = pd.Categorical(out["model_tag"], MODEL_ORDER, ordered=True)
    return out.sort_values(["condition", "model_tag"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_a_colors(role: str) -> str:
    if role in POSITION_COLORS:
        return POSITION_COLORS[role]
    return POSITION_COLORS["other"]


def draw_panel_a(axs: List[plt.Axes], allpos: pd.DataFrame) -> None:
    ymax = max(5.4, float(allpos["value"].max()) + 0.45)
    ymin = min(-0.18, float(allpos["value"].min()) - 0.08)

    for ax, model_tag in zip(axs, MODEL_ORDER):
        sub = allpos[allpos["model_tag"] == model_tag].copy()
        sub["canonical_role"] = pd.Categorical(sub["canonical_role"], CANONICAL_ROLES, ordered=True)
        sub = sub.sort_values("canonical_role")

        x = np.arange(len(CANONICAL_ROLES))
        vals = sub["value"].to_numpy(dtype=float)
        ci_lo = sub["ci_lo"].to_numpy(dtype=float)
        ci_hi = sub["ci_hi"].to_numpy(dtype=float)

        yerr = None
        if np.isfinite(ci_lo).all() and np.isfinite(ci_hi).all():
            yerr = np.vstack([vals - ci_lo, ci_hi - vals])

        colors = [panel_a_colors(str(role)) for role in sub["canonical_role"]]

        ax.yaxis.grid(True, alpha=0.13, linestyle="-", linewidth=0.55, zorder=0)
        ax.xaxis.grid(False)
        ax.bar(
            x,
            vals,
            width=0.68,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            yerr=yerr,
            capsize=2.5 if yerr is not None else 0,
            error_kw={"elinewidth": 0.8, "ecolor": "0.25", "capthick": 0.8},
            zorder=3,
        )

        # Label the two main diagnostic bars.
        for xi, role, val in zip(x, sub["canonical_role"].astype(str), vals):
            if role in {"subject", "final"}:
                ax.text(
                    xi,
                    val + 0.12,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.6,
                    fontweight="bold",
                    color="0.20",
                )

        ax.set_title(MODEL_LABELS[model_tag], fontsize=9.3, fontweight="bold", pad=6)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["The", "subject", "of", "the", "distractor", "in", "final"],
            rotation=28,
            ha="right",
            fontsize=7.2,
        )
        ax.axhline(0, color="0.20", linewidth=0.9, zorder=5)
        ax.spines["bottom"].set_visible(False)

    axs[0].set_ylabel("Signed steering effect", fontsize=8.5)


def draw_grouped_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    group_order: List[str],
    title: str,
    ylabel: str = "Subject / final ratio",
    annotate_qwen: bool = True,
) -> None:
    x = np.arange(len(group_order))
    width = 0.24
    offsets = {"phi2": -width, "llama32_3b": 0.0, "qwen25_3b": width}

    ax.yaxis.grid(True, alpha=0.14, linestyle="-", linewidth=0.55, zorder=0)
    ax.xaxis.grid(False)

    for model_tag in MODEL_ORDER:
        vals = []
        for g in group_order:
            sub = df[(df[group_col].astype(str) == g) & (df["model_tag"].astype(str) == model_tag)]
            if sub.empty:
                vals.append(np.nan)
            else:
                vals.append(float(sub[value_col].iloc[0]))
        xpos = x + offsets[model_tag]
        ax.bar(
            xpos,
            vals,
            width=width,
            label=MODEL_LABELS[model_tag],
            color=MODEL_COLORS[model_tag],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

        if annotate_qwen and model_tag == "qwen25_3b":
            for xi, val in zip(xpos, vals):
                if np.isfinite(val):
                    ax.text(
                        xi,
                        val + 0.07,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7.2,
                        fontweight="bold",
                        color=MODEL_COLORS[model_tag],
                    )

    ax.axhline(1.0, color="0.45", linestyle="--", linewidth=0.8, zorder=2)
    ax.text(
    0.85,
    1.03,
    "parity",
    transform=ax.get_yaxis_transform(),
    ha="right",
    va="bottom",
    fontsize=7.5,
    color="0.35",
)
    ax.set_xticks(x)
    ax.set_xticklabels(group_order, fontsize=7.8)
    ax.set_ylabel(ylabel, fontsize=8.5)
    if title:
        ax.set_title(title, fontsize=9.0, fontweight="bold", pad=5)
    ax.set_ylim(-0.08, max(4.1, float(df[value_col].max()) + 0.35))
    ax.axhline(0, color="0.20", linewidth=0.9, zorder=5)
    ax.spines["bottom"].set_visible(False)


def make_figure(
    allpos: pd.DataFrame,
    distance: pd.DataFrame,
    robustness: pd.DataFrame,
    out_dir: Path,
    output_name: str,
) -> None:
    setup_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.4, 6.25))
    gs = fig.add_gridspec(
    2,
    6,
    height_ratios=[1.05, 1.0],
    width_ratios=[1, 1, 1, 1, 1, 1],
    hspace=0.36,
    wspace=0.55,
)

    ax_a1 = fig.add_subplot(gs[0, 0:2])
    ax_a2 = fig.add_subplot(gs[0, 2:4], sharey=ax_a1)
    ax_a3 = fig.add_subplot(gs[0, 4:6], sharey=ax_a1)
    ax_b = fig.add_subplot(gs[1, 0:3])
    ax_c = fig.add_subplot(gs[1, 3:6], sharey=ax_b)

    draw_panel_a([ax_a1, ax_a2, ax_a3], allpos)
    plt.setp(ax_a2.get_yticklabels(), visible=False)
    plt.setp(ax_a3.get_yticklabels(), visible=False)

    # Panel labels.
    fig.text(0.1, 0.965, "(a) All-position causal profile", fontsize=9.0, fontweight="bold")
    fig.text(0.1, 0.455, "(b) Distance stress", fontsize=9.0, fontweight="bold")
    fig.text(0.510, 0.455, "(c) Robustness ratios", fontsize=9.0, fontweight="bold")

    # fig.text(
    #     0.5,
    #     0.94,
    #     "Localization and robustness of the cross-model dissociation",
    #     ha="center",
    #     va="top",
    #     fontsize=10.0,
    #     fontweight="bold",
    # )

    dist = distance.copy()
    dist["distance_label"] = dist["distance"].astype(str).replace({"extra_long": "extra long"})
    draw_grouped_bars(
        ax_b,
        dist,
        group_col="distance_label",
        value_col="ratio",
        group_order=["medium", "long", "extra long"],
        title="",
    )

    rob = robustness.copy()
    rob["condition_label"] = rob["condition"].astype(str)
    draw_grouped_bars(
        ax_c,
        rob,
        group_col="condition_label",
        value_col="ratio",
        group_order=["Original", "Held-out", "was/were", "Anchors"],
        title="",
    )

    # Shared legend for model-colour panels.
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=MODEL_COLORS[m], label=MODEL_LABELS[m])
        for m in MODEL_ORDER
    ]
    ax_c.legend(
    handles=handles,
    frameon=False,
    fontsize=7.0,
    loc="upper center",
    bbox_to_anchor=(-0.06, -.15),
    ncol=3,
)
    # Compact legend for panel A token colours.
    token_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=POSITION_COLORS["subject"], label="Subject"),
        plt.Rectangle((0, 0), 1, 1, fc=POSITION_COLORS["of"], label="of"),
        plt.Rectangle((0, 0), 1, 1, fc=POSITION_COLORS["distractor"], label="Distractor"),
        plt.Rectangle((0, 0), 1, 1, fc=POSITION_COLORS["final"], label="Final"),
        plt.Rectangle((0, 0), 1, 1, fc=POSITION_COLORS["other"], label="Other"),
    ]
    ax_a3.legend(handles=token_handles, frameon=False, fontsize=6.8, loc="upper right")

    pdf_path = out_dir / f"{output_name}.pdf"
    png_path = out_dir / f"{output_name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_manifest(
    out_dir: Path,
    allpos: pd.DataFrame,
    distance_path: Path,
    robustness: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    manifest_path = out_dir / "figure3_input_manifest.txt"
    lines = []
    lines.append("Figure 3 input manifest")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Command-line inputs:")
    lines.append(f"  main_allpos_dir       = {args.main_allpos_dir}")
    lines.append(f"  distance_ratios       = {args.distance_ratios}")
    lines.append(f"  main_steering_dir     = {args.main_steering_dir}")
    lines.append(f"  heldout_dir           = {args.heldout_dir}")
    lines.append(f"  waswere_steering_dir  = {args.waswere_steering_dir}")
    lines.append(f"  anchor_dir            = {args.anchor_dir}")
    lines.append("")
    lines.append("Panel A source files:")
    for p in sorted(set(allpos["source_file"].astype(str))):
        lines.append(f"  {p}")
    lines.append("")
    lines.append("Panel B source file:")
    lines.append(f"  {distance_path}")
    lines.append("")
    lines.append("Panel C source files:")
    for cond in ["Original", "Held-out", "was/were", "Anchors"]:
        lines.append(f"  {cond}:")
        sub = robustness[robustness["condition"].astype(str) == cond]
        for _, row in sub.iterrows():
            lines.append(f"    {MODEL_LABELS[str(row['model_tag'])]}: {row['source_file']}")
    lines.append("")
    manifest_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Figure 3 localization/robustness plot from experiment CSV files."
    )
    parser.add_argument(
        "--main_allpos_dir",
        type=Path,
        default=Path("results/all_position_steering/is_are/summary"),
        help="Directory containing main is/are all-position position-profile outputs.",
    )
    parser.add_argument(
        "--distance_ratios",
        type=Path,
        default=Path("results/distance_stress/attenuation/is_are/allpos_key_ratios.csv"),
        help="CSV containing distance subject/final ratios.",
    )
    parser.add_argument(
        "--main_steering_dir",
        type=Path,
        default=Path("results/steering/is_are/layerwise"),
        help="Directory containing main is/are steering layerwise outputs.",
    )
    parser.add_argument(
        "--heldout_dir",
        type=Path,
        default=Path("results/heldout_steering/is_are"),
        help="Directory containing held-out steering summaries.",
    )
    parser.add_argument(
        "--waswere_steering_dir",
        type=Path,
        default=Path("results/steering/was_were/layerwise"),
        help="Directory containing was/were steering layerwise outputs.",
    )
    parser.add_argument(
        "--anchor_dir",
        type=Path,
        default=Path("results/anchor_robustness/is_are/ratios"),
        help="Directory containing anchor robustness ratio outputs.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("figures/paper_main"),
        help="Output directory for Figure 3 and derived CSVs.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="figure3_localization_robustness",
        help="Base filename for PDF/PNG outputs.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    allpos = read_all_position_data(args.main_allpos_dir)
    distance = read_distance_ratios(args.distance_ratios)
    robustness = read_robustness_ratios(
        main_steering_dir=args.main_steering_dir,
        heldout_dir=args.heldout_dir,
        waswere_steering_dir=args.waswere_steering_dir,
        anchor_dir=args.anchor_dir,
    )

    # Save derived data used in the plot for reproducibility.
    allpos.to_csv(args.out_dir / "figure3_panel_a_allpos_from_csv.csv", index=False)
    distance.to_csv(args.out_dir / "figure3_panel_b_distance_from_csv.csv", index=False)
    robustness.to_csv(args.out_dir / "figure3_panel_c_ratios_from_csv.csv", index=False)
    print(f"Saved: {args.out_dir / 'figure3_panel_a_allpos_from_csv.csv'}")
    print(f"Saved: {args.out_dir / 'figure3_panel_b_distance_from_csv.csv'}")
    print(f"Saved: {args.out_dir / 'figure3_panel_c_ratios_from_csv.csv'}")

    make_figure(
        allpos=allpos,
        distance=distance,
        robustness=robustness,
        out_dir=args.out_dir,
        output_name=args.output_name,
    )
    write_manifest(args.out_dir, allpos, args.distance_ratios, robustness, args)

    # Print robustness numbers so mistakes are obvious in logs.
    print("\nPanel C ratios read/computed from CSVs:")
    printable = robustness.copy()
    printable["model"] = printable["model_tag"].astype(str).map(MODEL_LABELS)
    print(printable[["condition", "model", "ratio", "source_file"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
