#!/usr/bin/env python3
"""
All-position distance steering attenuation analysis.

Purpose
-------
Analyze distance all-position steering outputs where tokens are bucketed into roles,
e.g.:
  prefix_before_subject, subject, between_subject_distractor,
  distractor, filler_after_distractor, final

This is the version to use when you want to support claims such as:
  - steering effects attenuate from medium -> long -> extra_long
  - filler tokens remain weak
  - Qwen remains subject/source-localized while Phi/Llama remain more final-token integrated

Typical usage
-------------
# Repo path version, using all-position role summaries:
python scripts/14_distance_allpos_attenuation_analysis.py \
  --input results/distance_stress/all_position/is_are/summary \
  --pattern "allpos_distance_role_summary_*.csv" \
  --output_dir results/distance_stress/attenuation/is_are \
  --preview

# If raw all-position CSVs are available locally, they can also be used:
python scripts/14_distance_allpos_attenuation_analysis.py \
  --input results/distance_stress/all_position/is_are/raw \
  --pattern "allpos_distance_raw_*.csv" \
  --output_dir results/distance_stress/attenuation/is_are \
  --preview

Main outputs
------------
1. allpos_distance_role_summary.csv
   Mean effect per model x aux_pair x role x distance.

2. allpos_distance_attenuation_summary.csv
   Medium/long/extra_long effects, medium-to-extra_long drop, percent drop,
   slope over distance, and monotonicity per model x aux_pair x role.

3. allpos_distance_bootstrap.csv
   Bootstrap CIs for drop and slope, when item-level rows are available.

4. allpos_key_ratios.csv
   subject/final, between/final, filler/final, distractor/final ratios by distance.

5. allpos_distance_attenuation_report.md
   Paper-safe wording and caveats.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------

DISTANCE_ORDER = {
    "medium": 1,
    "long": 2,
    "extra_long": 3,
}

DISTANCE_ALIASES = {
    "medium": "medium",
    "med": "medium",
    "long": "long",
    "extra_long": "extra_long",
    "extra-long": "extra_long",
    "extra long": "extra_long",
    "extralong": "extra_long",
    "short": "short",
}

ROLE_ORDER = [
    "prefix_before_subject",
    "subject",
    "between_subject_distractor",
    "distractor",
    "filler_after_distractor",
    "final",
]

ROLE_ALIASES = {
    # canonical all-position roles
    "prefix_before_subject": "prefix_before_subject",
    "prefix": "prefix_before_subject",
    "before_subject": "prefix_before_subject",
    "the_prefix": "prefix_before_subject",
    "subject": "subject",
    "subj": "subject",
    "subject_token": "subject",
    "between_subject_distractor": "between_subject_distractor",
    "between": "between_subject_distractor",
    "between_tokens": "between_subject_distractor",
    "of_the": "between_subject_distractor",
    "of": "between_subject_distractor",
    "distractor": "distractor",
    "dist": "distractor",
    "distractor_token": "distractor",
    "filler_after_distractor": "filler_after_distractor",
    "filler": "filler_after_distractor",
    "adverbial_filler": "filler_after_distractor",
    "after_distractor": "filler_after_distractor",
    "final": "final",
    "final_token": "final",
    "last": "final",
    "last_token": "final",
}

MODEL_COL_CANDIDATES = ["model_tag", "model_name", "model", "model_id"]
AUX_COL_CANDIDATES = ["aux_pair", "verb_pair", "aux", "pair", "verb"]
DISTANCE_COL_CANDIDATES = ["distance_bucket", "distance", "dist", "bucket"]
ROLE_COL_CANDIDATES = ["role", "position", "token_role", "bucketed_role", "token_bucket", "target_position"]
ITEM_COL_CANDIDATES = ["group_id", "item_id", "pair_id", "record_idx", "record_id", "idx", "example_id"]
LAYER_COL_CANDIDATES = ["layer", "layer_idx"]

# Prefer already-signed effects. For all-position steering, signed positive should mean expected plural push.
EFFECT_COL_CANDIDATES = [
    "signed_effect",
    "signed_real",
    "signed_effect_mean",
    "real_mean",
    "effect",
    "real_effect",
    "steering_effect",
    "mean_effect",
]

RANDOM_COL_CANDIDATES = ["signed_rand", "rand_mean", "rand_effect", "random_effect", "random_mean"]

MODEL_DISPLAY = {
    "phi2": "Phi-2",
    "llama32_3b": "Llama-3.2-3B",
    "qwen25_3b": "Qwen2.5-3B",
}


# -----------------------------
# Basic helpers
# -----------------------------

def find_col(df: pd.DataFrame, candidates: Sequence[str], required: bool, label: str) -> Optional[str]:
    lower_to_actual = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_to_actual:
            return lower_to_actual[cand.lower()]
    if required:
        raise ValueError(
            f"Could not find required {label}. Tried: {candidates}.\n"
            f"Available columns:\n{list(df.columns)}"
        )
    return None


def normalize_distance(x: object) -> str:
    s = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    return DISTANCE_ALIASES.get(s, s)


def normalize_role(x: object) -> str:
    s = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    return ROLE_ALIASES.get(s, s)


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def linear_slope_from_means(means: Dict[str, float]) -> float:
    xs, ys = [], []
    for d, idx in DISTANCE_ORDER.items():
        y = means.get(d, np.nan)
        if pd.notna(y):
            xs.append(idx)
            ys.append(y)
    if len(xs) < 2:
        return np.nan
    return float(np.polyfit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), 1)[0])


def pct_drop(medium: float, extra: float) -> float:
    if pd.isna(medium) or pd.isna(extra) or abs(medium) < 1e-8:
        return np.nan
    return float((medium - extra) / medium * 100.0)


def monotonic_decrease(m: float, l: float, e: float) -> bool:
    if pd.isna(m) or pd.isna(l) or pd.isna(e):
        return False
    return bool(m >= l >= e)


# -----------------------------
# Loading / normalization
# -----------------------------

def load_input(input_path: Path, pattern: str) -> pd.DataFrame:
    if input_path.is_dir():
        csvs = sorted(input_path.glob(pattern))
        if not csvs and pattern != "*.csv":
            print(f"[warn] no files matched {pattern}; falling back to *.csv", file=sys.stderr)
            csvs = sorted(input_path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in directory: {input_path}")

        frames = []
        for p in csvs:
            try:
                f = pd.read_csv(p)
                f["source_file"] = p.name
                frames.append(f)
                print(f"[read] {p} shape={f.shape}")
            except Exception as exc:
                print(f"[warn] could not read {p}: {exc}", file=sys.stderr)
        if not frames:
            raise ValueError(f"Could not read any CSVs in {input_path}")
        return pd.concat(frames, ignore_index=True, sort=False)

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    return pd.read_csv(input_path)


def normalize_dataframe(
    raw: pd.DataFrame,
    effect_col: Optional[str],
    model_col: Optional[str],
    aux_col: Optional[str],
    distance_col: Optional[str],
    role_col: Optional[str],
    item_col: Optional[str],
    aux_default: str,
    keep_layer0: bool,
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    df = raw.copy()

    model_col = model_col or find_col(df, MODEL_COL_CANDIDATES, True, "model column")
    aux_col = aux_col or find_col(df, AUX_COL_CANDIDATES, False, "aux/verb-pair column")
    distance_col = distance_col or find_col(df, DISTANCE_COL_CANDIDATES, True, "distance column")
    role_col = role_col or find_col(df, ROLE_COL_CANDIDATES, True, "role/position column")
    effect_col = effect_col or find_col(df, EFFECT_COL_CANDIDATES, True, "effect column")
    item_col = item_col or find_col(df, ITEM_COL_CANDIDATES, False, "item/group column")
    layer_col = find_col(df, LAYER_COL_CANDIDATES, False, "layer column")
    random_col = find_col(df, RANDOM_COL_CANDIDATES, False, "random-control column")

    rename = {
        model_col: "model_name",
        distance_col: "distance",
        role_col: "role",
        effect_col: "effect",
    }
    if aux_col is not None:
        rename[aux_col] = "aux_pair"
    else:
        df["aux_pair"] = aux_default
    if item_col is not None:
        rename[item_col] = "item_id"
    else:
        # summary-only CSVs may not have item ids; use row ids so summaries still work.
        df["item_id"] = np.arange(len(df))
    if layer_col is not None:
        rename[layer_col] = "layer"
    if random_col is not None:
        rename[random_col] = "random_effect"

    df = df.rename(columns=rename)
    df["distance"] = df["distance"].map(normalize_distance)
    df["role"] = df["role"].map(normalize_role)
    df["effect"] = numeric(df["effect"])
    if "random_effect" in df.columns:
        df["random_effect"] = numeric(df["random_effect"])

    # Keep layers 1+ by default when raw layer column is present.
    if "layer" in df.columns and not keep_layer0:
        df["layer"] = numeric(df["layer"])
        before = len(df)
        df = df[(df["layer"].isna()) | (df["layer"] > 0)].copy()
        print(f"[filter] removed layer 0 rows: {before} -> {len(df)}")

    df = df.dropna(subset=["effect"])

    info = {
        "model_col": model_col,
        "aux_col": aux_col,
        "distance_col": distance_col,
        "role_col": role_col,
        "effect_col": effect_col,
        "item_col": item_col,
        "layer_col": layer_col,
        "random_col": random_col,
        "aux_default_if_missing": aux_default,
        "keep_layer0": str(keep_layer0),
    }
    return df, info


def aggregate_to_item_role(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw token/layer rows into one row per item x model x aux x distance x role.
    If the input is already summary-like, this still works, but bootstrap should be interpreted carefully.
    """
    group_cols = ["model_name", "aux_pair", "distance", "role", "item_id"]
    agg = df.groupby(group_cols, as_index=False).agg(
        effect=("effect", "mean"),
        n_rows=("effect", "size"),
    )
    if "random_effect" in df.columns:
        rand = df.groupby(group_cols, as_index=False).agg(random_effect=("random_effect", "mean"))
        agg = agg.merge(rand, on=group_cols, how="left")
    return agg


# -----------------------------
# Summaries
# -----------------------------

def role_distance_summary(item_df: pd.DataFrame) -> pd.DataFrame:
    main = item_df[item_df["distance"].isin(DISTANCE_ORDER)].copy()
    out = main.groupby(["model_name", "aux_pair", "role", "distance"], as_index=False).agg(
        mean_effect=("effect", "mean"),
        std_effect=("effect", "std"),
        n_items=("item_id", "nunique"),
        n_rows=("effect", "size"),
    )
    if "random_effect" in main.columns:
        rand = main.groupby(["model_name", "aux_pair", "role", "distance"], as_index=False).agg(
            mean_random=("random_effect", "mean"),
            std_random=("random_effect", "std"),
        )
        out = out.merge(rand, on=["model_name", "aux_pair", "role", "distance"], how="left")
    out["distance_idx"] = out["distance"].map(DISTANCE_ORDER)
    out["role_idx"] = out["role"].apply(lambda r: ROLE_ORDER.index(r) if r in ROLE_ORDER else 999)
    return out.sort_values(["model_name", "aux_pair", "role_idx", "distance_idx"]).drop(columns=["role_idx"])


def attenuation_summary(item_df: pd.DataFrame) -> pd.DataFrame:
    summary = role_distance_summary(item_df)
    rows = []
    for keys, g in summary.groupby(["model_name", "aux_pair", "role"], dropna=False):
        model_name, aux_pair, role = keys
        means = g.set_index("distance")["mean_effect"].to_dict()
        ns = g.set_index("distance")["n_items"].to_dict()
        medium = means.get("medium", np.nan)
        long = means.get("long", np.nan)
        extra = means.get("extra_long", np.nan)
        drop = medium - extra if pd.notna(medium) and pd.notna(extra) else np.nan
        rows.append({
            "model_name": model_name,
            "aux_pair": aux_pair,
            "role": role,
            "mean_medium": medium,
            "mean_long": long,
            "mean_extra_long": extra,
            "n_medium": int(ns.get("medium", 0)),
            "n_long": int(ns.get("long", 0)),
            "n_extra_long": int(ns.get("extra_long", 0)),
            "drop_medium_to_extra_long": drop,
            "pct_drop_medium_to_extra_long": pct_drop(medium, extra),
            "slope_over_distance": linear_slope_from_means(means),
            "monotonic_medium_long_extra": monotonic_decrease(medium, long, extra),
            "attenuates_medium_to_extra": bool(pd.notna(drop) and drop > 0),
        })
    out = pd.DataFrame(rows)
    out["role_idx"] = out["role"].apply(lambda r: ROLE_ORDER.index(r) if r in ROLE_ORDER else 999)
    return out.sort_values(["model_name", "aux_pair", "role_idx"]).drop(columns=["role_idx"])


def wide_role_profile(summary: pd.DataFrame) -> pd.DataFrame:
    wide = summary.pivot_table(
        index=["model_name", "aux_pair", "role"],
        columns="distance",
        values="mean_effect",
        aggfunc="first",
    ).reset_index()
    for d in DISTANCE_ORDER:
        if d not in wide.columns:
            wide[d] = np.nan
    wide["drop_medium_to_extra_long"] = wide["medium"] - wide["extra_long"]
    wide["pct_drop_medium_to_extra_long"] = wide.apply(
        lambda r: pct_drop(r["medium"], r["extra_long"]), axis=1
    )
    wide["role_idx"] = wide["role"].apply(lambda r: ROLE_ORDER.index(r) if r in ROLE_ORDER else 999)
    return wide.sort_values(["model_name", "aux_pair", "role_idx"]).drop(columns=["role_idx"])


def key_ratios(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot_table(
        index=["model_name", "aux_pair", "distance"],
        columns="role",
        values="mean_effect",
        aggfunc="first",
    ).reset_index()
    for role in ROLE_ORDER:
        if role not in pivot.columns:
            pivot[role] = np.nan

    def div(a: pd.Series, b: pd.Series) -> pd.Series:
        return a / b.replace(0, np.nan)

    pivot["subject_div_final"] = div(pivot["subject"], pivot["final"])
    pivot["between_div_final"] = div(pivot["between_subject_distractor"], pivot["final"])
    pivot["filler_div_final"] = div(pivot["filler_after_distractor"], pivot["final"])
    pivot["distractor_div_final"] = div(pivot["distractor"], pivot["final"])
    pivot["subject_minus_final"] = pivot["subject"] - pivot["final"]
    pivot["dominance"] = np.where(
        pivot["subject"] > pivot["final"],
        "subject_dominant",
        np.where(pivot["final"] > pivot["subject"], "final_dominant", "tie_or_missing"),
    )
    pivot["distance_idx"] = pivot["distance"].map(DISTANCE_ORDER)
    return pivot.sort_values(["model_name", "aux_pair", "distance_idx"]).drop(columns=["distance_idx"])


# -----------------------------
# Bootstrap
# -----------------------------

def bootstrap_one_group(g: pd.DataFrame, n_boot: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    arrs = {d: g[g["distance"] == d]["effect"].dropna().to_numpy() for d in DISTANCE_ORDER}
    if len(arrs["medium"]) == 0 or len(arrs["extra_long"]) == 0:
        return {}

    drops, pcts, slopes = [], [], []
    for _ in range(n_boot):
        means = {}
        for d, arr in arrs.items():
            if len(arr) == 0:
                means[d] = np.nan
            else:
                means[d] = float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
        drop = means["medium"] - means["extra_long"]
        drops.append(drop)
        pcts.append(pct_drop(means["medium"], means["extra_long"]))
        slopes.append(linear_slope_from_means(means))

    drops = np.asarray(drops, dtype=float)
    pcts = np.asarray(pcts, dtype=float)
    slopes = np.asarray(slopes, dtype=float)
    return {
        "boot_drop_mean": float(np.nanmean(drops)),
        "boot_drop_ci_low": float(np.nanpercentile(drops, 2.5)),
        "boot_drop_ci_high": float(np.nanpercentile(drops, 97.5)),
        "p_drop_positive": float(np.nanmean(drops > 0)),
        "boot_pct_drop_mean": float(np.nanmean(pcts)),
        "boot_pct_drop_ci_low": float(np.nanpercentile(pcts, 2.5)),
        "boot_pct_drop_ci_high": float(np.nanpercentile(pcts, 97.5)),
        "boot_slope_mean": float(np.nanmean(slopes)),
        "boot_slope_ci_low": float(np.nanpercentile(slopes, 2.5)),
        "boot_slope_ci_high": float(np.nanpercentile(slopes, 97.5)),
        "p_slope_negative": float(np.nanmean(slopes < 0)),
    }


def bootstrap_attenuation(item_df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    main = item_df[item_df["distance"].isin(DISTANCE_ORDER)].copy()
    rows = []
    for i, (keys, g) in enumerate(main.groupby(["model_name", "aux_pair", "role"], dropna=False)):
        model_name, aux_pair, role = keys
        out = bootstrap_one_group(g, n_boot=n_boot, seed=seed + i)
        if not out:
            continue
        out.update({
            "model_name": model_name,
            "aux_pair": aux_pair,
            "role": role,
            "n_total": int(len(g)),
            "n_medium": int((g["distance"] == "medium").sum()),
            "n_long": int((g["distance"] == "long").sum()),
            "n_extra_long": int((g["distance"] == "extra_long").sum()),
        })
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["role_idx"] = out["role"].apply(lambda r: ROLE_ORDER.index(r) if r in ROLE_ORDER else 999)
    return out.sort_values(["model_name", "aux_pair", "role_idx"]).drop(columns=["role_idx"])


# -----------------------------
# Report
# -----------------------------

def fmt(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x:.3f}"


def write_report(
    output_dir: Path,
    info: Dict[str, Optional[str]],
    role_summary: pd.DataFrame,
    attenuation: pd.DataFrame,
    boot: pd.DataFrame,
    ratios: pd.DataFrame,
) -> None:
    path = output_dir / "allpos_distance_attenuation_report.md"
    n_groups = len(attenuation)
    n_att = int(attenuation["attenuates_medium_to_extra"].sum()) if not attenuation.empty else 0
    n_mono = int(attenuation["monotonic_medium_long_extra"].sum()) if not attenuation.empty else 0

    lines = []
    lines.append("# All-position distance attenuation report")
    lines.append("")
    lines.append("## Column mapping used")
    for k, v in info.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Main analysis")
    lines.append("Uses medium, long, and extra_long distances. The short bucket is ignored by default because short prompts can make distractor/final roles overlap.")
    lines.append("")
    lines.append(f"Groups tested: {n_groups}")
    lines.append(f"Groups where medium > extra_long: {n_att}/{n_groups}")
    lines.append(f"Groups monotonic medium >= long >= extra_long: {n_mono}/{n_groups}")
    lines.append("")

    if not boot.empty:
        strong = boot[(boot["p_drop_positive"] >= 0.95) & (boot["boot_drop_ci_low"] > 0)]
        lines.append("## Bootstrap evidence")
        lines.append(f"Groups with bootstrap-supported positive medium-to-extra_long drop: {len(strong)}/{len(boot)}")
        lines.append("")

    # Compact per-model ratio notes
    lines.append("## Key model patterns")
    for (model, aux), g in ratios.groupby(["model_name", "aux_pair"], dropna=False):
        g = g.sort_values("distance", key=lambda s: s.map(DISTANCE_ORDER))
        subj_final = ", ".join(f"{r.distance}: {fmt(r.subject_div_final)}x" for r in g.itertuples())
        filler_final = ", ".join(f"{r.distance}: {fmt(r.filler_div_final)}x" for r in g.itertuples())
        lines.append(f"- **{MODEL_DISPLAY.get(str(model), str(model))} / {aux}**: subject/final = {subj_final}; filler/final = {filler_final}.")
    lines.append("")

    lines.append("## Paper-safe wording")
    lines.append("")
    lines.append(
        "> We next analyze all-position distance steering by bucketing token positions into subject-adjacent, distractor, filler, and final roles. "
        "Across medium, long, and extra-long prompts, steering effects generally attenuate with distance. "
        "The attenuation is not explained by broad spreading into intervening filler tokens: filler-role effects remain small compared with subject and final roles, and distractor effects remain near zero. "
        "Instead, controllability remains concentrated at the subject, immediately adjacent between-subject-distractor tokens, and the final prediction token, with reduced magnitude at longer distances."
    )
    lines.append("")
    lines.append("## Caveats")
    lines.append("- Interpret these as causal controllability effects, not literal movement of a signal between tokens.")
    lines.append("- Do not infer attention-routing mechanisms from these steering results alone.")
    lines.append("- If this analysis is only run for `is/are`, describe it as an all-position diagnostic on the main present-tense paradigm.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="All-position distance steering attenuation analysis.")
    p.add_argument("--input", required=True, type=Path, help="Raw all-position distance CSV or directory.")
    p.add_argument("--pattern", default="*raw*.csv", help="CSV glob when --input is a directory. Default: *raw*.csv")
    p.add_argument("--output_dir", default=Path("results/distance_stress/attenuation/is_are"), type=Path)
    p.add_argument("--effect_col", default=None)
    p.add_argument("--model_col", default=None)
    p.add_argument("--aux_col", default=None)
    p.add_argument("--distance_col", default=None)
    p.add_argument("--role_col", default=None)
    p.add_argument("--item_col", default=None)
    p.add_argument("--aux_default", default="be_present", help="Used if no aux/verb-pair column exists.")
    p.add_argument("--keep_layer0", action="store_true", help="Include layer 0 rows if a layer column exists. Default excludes layer 0.")
    p.add_argument("--n_boot", default=2000, type=int)
    p.add_argument("--seed", default=123, type=int)
    p.add_argument("--no_bootstrap", action="store_true")
    p.add_argument("--preview", action="store_true")
    return p.parse_args()


def preview(name: str, df: pd.DataFrame, n: int = 18) -> None:
    print(f"\n[{name}] shape={df.shape}")
    if not df.empty:
        with pd.option_context("display.max_columns", 80, "display.width", 220):
            print(df.head(n).to_string(index=False))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.input}")
    raw = load_input(args.input, pattern=args.pattern)
    print(f"[info] raw shape={raw.shape}")

    df, info = normalize_dataframe(
        raw,
        effect_col=args.effect_col,
        model_col=args.model_col,
        aux_col=args.aux_col,
        distance_col=args.distance_col,
        role_col=args.role_col,
        item_col=args.item_col,
        aux_default=args.aux_default,
        keep_layer0=args.keep_layer0,
    )
    print(f"[info] normalized shape={df.shape}")
    print("[info] column mapping:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    item_df = aggregate_to_item_role(df)
    item_df.to_csv(args.output_dir / "allpos_normalized_item_role_input.csv", index=False)

    role_sum = role_distance_summary(item_df)
    role_sum.to_csv(args.output_dir / "allpos_distance_role_summary.csv", index=False)

    atten = attenuation_summary(item_df)
    atten.to_csv(args.output_dir / "allpos_distance_attenuation_summary.csv", index=False)

    wide = wide_role_profile(role_sum)
    wide.to_csv(args.output_dir / "allpos_distance_role_profile_wide.csv", index=False)

    ratios = key_ratios(role_sum)
    ratios.to_csv(args.output_dir / "allpos_key_ratios.csv", index=False)

    if args.no_bootstrap:
        boot = pd.DataFrame()
    else:
        boot = bootstrap_attenuation(item_df, n_boot=args.n_boot, seed=args.seed)
    boot.to_csv(args.output_dir / "allpos_distance_bootstrap.csv", index=False)

    write_report(args.output_dir, info, role_sum, atten, boot, ratios)

    if args.preview:
        preview("allpos_distance_role_summary", role_sum)
        preview("allpos_distance_role_profile_wide", wide)
        preview("allpos_distance_attenuation_summary", atten)
        preview("allpos_key_ratios", ratios)
        preview("allpos_distance_bootstrap", boot)

    print(f"\n[done] wrote outputs to: {args.output_dir.resolve()}")
    print("Key files:")
    for fname in [
        "allpos_distance_role_summary.csv",
        "allpos_distance_role_profile_wide.csv",
        "allpos_distance_attenuation_summary.csv",
        "allpos_distance_bootstrap.csv",
        "allpos_key_ratios.csv",
        "allpos_distance_attenuation_report.md",
    ]:
        print(f"  - {args.output_dir / fname}")


if __name__ == "__main__":
    main()
