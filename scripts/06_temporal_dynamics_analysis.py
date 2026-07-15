"""
temporal_dynamics_analysis.py
=============================
Quantify the temporal dynamics of causal controllability from all-position
steering data.

For each model, computes three metrics from the per-layer signed effects:

  1. Source AUC      = sum over layers >= 1 of (subject_effect + of_effect)
  2. Final AUC       = sum over layers >= 1 of final_effect
  3. Final onset     = first layer >= 1 where final_effect >= threshold (default 1.5)

The source/final AUC ratio summarizes the trajectory asymmetry, and the onset
layer captures *when* the final-token representation comes online.

Why this exists
---------------
The all-position heatmaps show qualitative differences between models, but a
referee may ask "how much later does Qwen's final onset happen?" or "how
much more source-loaded is Qwen?". This script answers those with single
scalar values rather than visual inspection.

Inputs
------
  results/all_position_steering/is_are/layerwise/test3_allpos_summary_by_layer_{model}.csv

For was/were, change only INPUT_DIR and OUTPUT_DIR in the CONFIG block to:
  results/all_position_steering/was_were/layerwise
  results/temporal/was_were

Outputs
-------
  results/temporal/is_are/temporal_dynamics_summary.csv
  results/temporal/is_are/temporal_dynamics_paper_numbers.txt

For was/were:
  results/temporal/was_were/temporal_dynamics_summary.csv
  results/temporal/was_were/temporal_dynamics_paper_numbers.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

# Default: is/are temporal dynamics.
# To reproduce was/were, change ONLY these two lines to the was/were paths below.
INPUT_DIR = Path("results/all_position_steering/is_are/layerwise")
OUTPUT_DIR = Path("results/temporal/is_are")

# For was/were, use:
# INPUT_DIR = Path("results/all_position_steering/was_were/layerwise")
# OUTPUT_DIR = Path("results/temporal/was_were")

MODELS = [
    {"tag": "phi2",       "display": "Phi-2"},
    {"tag": "llama32_3b", "display": "Llama-3.2-3B"},
    {"tag": "qwen25_3b",  "display": "Qwen2.5-3B"},
]

# Per-model token position mapping (handles Llama BOS offset)
MODEL_POSITION_MAP = {
    "phi2":       {"subject": 1, "of": 2, "final": 6},
    "llama32_3b": {"subject": 2, "of": 3, "final": 7},
    "qwen25_3b":  {"subject": 1, "of": 2, "final": 6},
}

# Threshold for "final onset": first layer where final effect crosses this value.
# 1.5 logit units is a reasonable choice — comfortably above noise and well below
# saturation. Sensitivity analysis with 1.0 and 2.0 also reported.
ONSET_THRESHOLD = 1.5
ALT_THRESHOLDS = [1.0, 2.0]


# ============================================================
# CORE COMPUTATION
# ============================================================

def get_layer_effects(summary_df, target_pos):
    """
    Return an array indexed by layer (1..max_layer) of signed effects at
    target_pos. If a position has multiple rows per layer (due to multi-subtoken
    items in the original script), pick the row with the most items.
    """
    rows = summary_df[summary_df["token_pos"] == target_pos]
    if len(rows) == 0:
        return None

    # Pick best row per layer by n_items
    rows = (
        rows.sort_values("n_items", ascending=False)
            .drop_duplicates(subset="layer", keep="first")
            .sort_values("layer")
    )

    # Drop layer 0 (embedding); keep layers 1..N
    rows = rows[rows["layer"] > 0]

    effects = rows["signed_effect_mean"].values
    layers = rows["layer"].values
    return layers, effects


def compute_auc(effects):
    """Simple sum (rectangle rule, layer-width = 1)."""
    return float(np.sum(effects))


def compute_onset(layers, effects, threshold):
    """First layer where effect crosses threshold. Returns -1 if never."""
    for L, e in zip(layers, effects):
        if e >= threshold:
            return int(L)
    return -1


def compute_metrics_for_model(summary_df, model_tag):
    pos_map = MODEL_POSITION_MAP[model_tag]

    # Get per-layer effects for each canonical position
    subj_layers, subj_effects = get_layer_effects(summary_df, pos_map["subject"])
    of_layers, of_effects = get_layer_effects(summary_df, pos_map["of"])
    final_layers, final_effects = get_layer_effects(summary_df, pos_map["final"])

    # Align layers across positions (use intersection — should normally be identical)
    common_layers = sorted(set(subj_layers) & set(of_layers) & set(final_layers))
    common_layers = np.array(common_layers)

    def align(layers, effects):
        idx = np.array([np.where(layers == L)[0][0] for L in common_layers])
        return effects[idx]

    subj = align(subj_layers, subj_effects)
    of_ = align(of_layers, of_effects)
    final = align(final_layers, final_effects)

    source_combined = subj + of_

    metrics = {
        "model_tag": model_tag,
        "n_layers": len(common_layers),
        "source_auc_subject_only": compute_auc(subj),
        "source_auc_of_only": compute_auc(of_),
        "source_auc_combined": compute_auc(source_combined),
        "final_auc": compute_auc(final),
    }

    src_combined = metrics["source_auc_combined"]
    final_auc = metrics["final_auc"]
    metrics["source_final_ratio_combined"] = (
        src_combined / final_auc if abs(final_auc) > 1e-8 else float("inf")
    )
    metrics["source_final_ratio_subject_only"] = (
        metrics["source_auc_subject_only"] / final_auc
        if abs(final_auc) > 1e-8 else float("inf")
    )

    # Final onset at default threshold and alternates
    metrics["final_onset_layer"] = compute_onset(common_layers, final, ONSET_THRESHOLD)
    metrics["final_onset_threshold"] = ONSET_THRESHOLD
    for t in ALT_THRESHOLDS:
        metrics[f"final_onset_at_{t}"] = compute_onset(common_layers, final, t)

    # Mean per-layer effects (useful sanity-check numbers)
    metrics["subject_mean_per_layer"] = float(np.mean(subj))
    metrics["of_mean_per_layer"] = float(np.mean(of_))
    metrics["final_mean_per_layer"] = float(np.mean(final))

    # Peak layers
    metrics["subject_peak_layer"] = int(common_layers[int(np.argmax(subj))])
    metrics["of_peak_layer"] = int(common_layers[int(np.argmax(of_))])
    metrics["final_peak_layer"] = int(common_layers[int(np.argmax(final))])
    metrics["subject_peak_value"] = float(np.max(subj))
    metrics["of_peak_value"] = float(np.max(of_))
    metrics["final_peak_value"] = float(np.max(final))

    return metrics


# ============================================================
# REPORTING
# ============================================================

def format_paper_numbers(all_metrics):
    lines = []
    lines.append("TEMPORAL DYNAMICS — PAPER NUMBERS")
    lines.append("=" * 78)
    lines.append("Quantifying source vs final-token controllability across layers.")
    lines.append(f"Final onset threshold = {ONSET_THRESHOLD} (signed effect in logit units).")
    lines.append("")

    # Main table
    lines.append(
        f"{'Model':<14} {'n_layers':>8} "
        f"{'Src AUC':>10} {'Final AUC':>10} "
        f"{'Src/Fin':>9} {'Onset@1.5':>10} "
        f"{'Onset@1.0':>10} {'Onset@2.0':>10}"
    )
    lines.append("-" * 78)

    for m in all_metrics:
        lines.append(
            f"{m['model_tag']:<14} {m['n_layers']:>8} "
            f"{m['source_auc_combined']:>10.2f} {m['final_auc']:>10.2f} "
            f"{m['source_final_ratio_combined']:>9.2f}x "
            f"{m['final_onset_layer']:>10} "
            f"{m['final_onset_at_1.0']:>10} "
            f"{m['final_onset_at_2.0']:>10}"
        )

    lines.append("")
    lines.append("Source AUC components (subject + of):")
    lines.append(f"{'Model':<14} {'Subject AUC':>13} {'Of AUC':>10} {'Total':>10}")
    lines.append("-" * 50)
    for m in all_metrics:
        lines.append(
            f"{m['model_tag']:<14} "
            f"{m['source_auc_subject_only']:>13.2f} "
            f"{m['source_auc_of_only']:>10.2f} "
            f"{m['source_auc_combined']:>10.2f}"
        )

    lines.append("")
    lines.append("Per-layer peaks (layer of maximum effect, peak value):")
    lines.append(
        f"{'Model':<14} {'Subj peak':>14} {'Of peak':>14} {'Final peak':>14}"
    )
    lines.append("-" * 60)
    for m in all_metrics:
        lines.append(
            f"{m['model_tag']:<14} "
            f"L{m['subject_peak_layer']}={m['subject_peak_value']:.2f}     "
            f"L{m['of_peak_layer']}={m['of_peak_value']:.2f}     "
            f"L{m['final_peak_layer']}={m['final_peak_value']:.2f}"
        )

    lines.append("")
    lines.append("=" * 78)
    lines.append("Paper-ready sentences:")
    lines.append("")

    # Build the headline interpretation
    qwen = next((m for m in all_metrics if m["model_tag"] == "qwen25_3b"), None)
    phi = next((m for m in all_metrics if m["model_tag"] == "phi2"), None)
    lla = next((m for m in all_metrics if m["model_tag"] == "llama32_3b"), None)

    if qwen and phi and lla:
        lines.append(
            f"Across the network, Qwen2.5-3B has source/final AUC ratio "
            f"{qwen['source_final_ratio_combined']:.2f}x, vs "
            f"{phi['source_final_ratio_combined']:.2f}x for Phi-2 and "
            f"{lla['source_final_ratio_combined']:.2f}x for Llama-3.2-3B."
        )
        lines.append("")
        lines.append(
            f"Final-token onset layer (first layer where the final-position "
            f"signed effect exceeds {ONSET_THRESHOLD}) is layer "
            f"{qwen['final_onset_layer']} in Qwen, vs layer "
            f"{phi['final_onset_layer']} in Phi-2 and "
            f"{lla['final_onset_layer']} in Llama-3.2-3B."
        )
        lines.append("")
        lines.append(
            "This quantifies the temporal asymmetry: Qwen retains source-side "
            "controllability across a substantially larger fraction of the "
            "network and the final-token representation becomes a strong "
            "steering target only in the late layers."
        )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for model in MODELS:
        tag = model["tag"]
        path = INPUT_DIR / f"test3_allpos_summary_by_layer_{tag}.csv"
        if not path.exists():
            print(f"[WARN] Missing {path}, skipping {tag}")
            continue

        df = pd.read_csv(path)
        metrics = compute_metrics_for_model(df, tag)
        all_metrics.append(metrics)

    if not all_metrics:
        print("[ERROR] No metrics computed.")
        return

    summary_df = pd.DataFrame(all_metrics)
    summary_path = OUTPUT_DIR / "temporal_dynamics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    report = format_paper_numbers(all_metrics)
    report_path = OUTPUT_DIR / "temporal_dynamics_paper_numbers.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved: {report_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
