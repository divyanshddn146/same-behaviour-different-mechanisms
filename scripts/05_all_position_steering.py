"""
05_all_position_steering.py
===========================
All-position steering: applying the subject-number direction at every token
position in the prompt, and measuring the causal effect on auxiliary-verb
logits.

Core question
-------------
Do models differ in where agreement control is causally usable?
For each model, we compare steering effects across subject, distractor,
intermediate, and final-token positions.

Template roles:
  pos 0: The
  pos 1: {subject}       e.g. "manager"
  pos 2: of
  pos 3: the
  pos 4: {distractor}    e.g. "restaurants"
  pos 5: in
  pos 6: question        (final token, agreement predicted here)

Method
------
For each item, for each layer, steer at EVERY actual token position.
Direction: subject-number direction (loaded from saved Exp3 pkl files).
Alpha: +1.0 only (plural push).
Target condition: SP (singular subject + plural distractor).

Token positions are extracted from the actual tokenized prompt per item
to handle any tokenizer differences — no hard-coded 0-6 assumptions.

GPU array support
-----------------
  sbatch --array=0-2 job.sh      → each task runs one model
  python 05_all_position_steering.py --model_idx 0   → phi2
  python 05_all_position_steering.py --model_idx 1   → llama
  python 05_all_position_steering.py --model_idx 2   → qwen
  python 05_all_position_steering.py                 → all models sequentially

Outputs (per model)
-------------------
test3_allpos_raw_{tag}.csv
test3_allpos_position_profile_{tag}.csv   ← mean over layers 1+ per position
test3_allpos_summary_by_layer_{tag}.csv   ← mean per (position, layer)
test3_layer0_{tag}.csv                    ← layer 0 separate
test3_paper_numbers_{tag}.txt
plot_allpos_profile_{tag}.png             ← main figure
plot_allpos_heatmap_{tag}.png             ← position × layer heatmap

After all models run:
plot_allmodels_comparison.png             ← cross-model comparison
"""

import os
import gc
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

DATASET_DIR  = "agreement_dataset_outputs"
BEHAV_DIR    = "behavioral_analysis_outputs"
STEER_DIR    = "exp3_steering_outputs"   # load saved subject directions
OUTPUT_DIR   = "test3_allpos_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOCAL_FILES_ONLY  = True
TRUST_REMOTE_CODE = True

MODELS = [
    {"model_id": "microsoft/phi-2",         "tag": "phi2",       "force_fp32": True},
    {"model_id": "meta-llama/Llama-3.2-3B", "tag": "llama32_3b", "force_fp32": False},
    {"model_id": "Qwen/Qwen2.5-3B",         "tag": "qwen25_3b",  "force_fp32": False},
]

VERB_PAIR  = "be_present"
SINGULAR_V = "is"
PLURAL_V   = "are"

# ------------------------------------------------------------------
# To reproduce the was/were all-position replication, copy this script
# and change only the CONFIG values below:
#
# STEER_DIR    = "exp3_steering_outputs_was_were"
# OUTPUT_DIR   = "test3_allpos_outputs_was_were"
# VERB_PAIR    = "be_past"
# SINGULAR_V   = "was"
# PLURAL_V     = "were"
#
# load_clean_items() will automatically look for:
#   clean_be_past_all_models_intersection.csv
# because VERB_PAIR = "be_past".
#
# No all-position steering logic changes are required.
# ------------------------------------------------------------------

TARGET_COND = "SP"   # attraction condition: singular subject + plural distractor
ALPHA       = 1.0    # plural push; signed_effect = -effect so positive = correct

# Bootstrap
N_BOOTSTRAP    = 2000
BOOTSTRAP_SEED = 42

MODEL_DISPLAY = {
    "phi2":       "Phi-2",
    "llama32_3b": "Llama-3.2-3B",
    "qwen25_3b":  "Qwen2.5-3B",
}

MODEL_COLORS = {
    "phi2":       "#5e4fa2",
    "llama32_3b": "#3288bd",
    "qwen25_3b":  "#d53e4f",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_clean_items() -> pd.DataFrame:
    preferred = os.path.join(BEHAV_DIR, f"clean_{VERB_PAIR}_all_models_intersection.csv")
    fallback  = os.path.join(DATASET_DIR, "clean_items_all_models_intersection.csv")

    if os.path.exists(preferred):
        path = preferred
    else:
        path = fallback

    df = pd.read_csv(path)
    df = df[df["verb_pair"] == VERB_PAIR].copy()
    df = df.drop_duplicates(subset=["item_id", "condition"])
    print(f"Loaded {df['item_id'].nunique()} clean base items for {VERB_PAIR}.")
    return df


def build_item_condition_map(df: pd.DataFrame) -> dict:
    item_map = {}
    for item_id, grp in df.groupby("item_id"):
        conds = {}
        for _, row in grp.iterrows():
            conds[row["condition"]] = row.to_dict()
        if len(conds) == 4:
            item_map[item_id] = conds
    print(f"Items with all 4 conditions: {len(item_map)}")
    return item_map


# ============================================================
# MODEL LOADING
# ============================================================

def safe_cuda() -> bool:
    try:
        if torch.cuda.is_available():
            torch.zeros(1).cuda()
            return True
    except RuntimeError:
        pass
    return False


def load_model(model_id: str, force_fp32: bool = False):
    print(f"\n{'='*70}\nLOADING: {model_id}\n{'='*70}")
    use_cuda = safe_cuda()
    device   = torch.device("cuda" if use_cuda else "cpu")
    dtype    = torch.float32 if force_fp32 else (
               torch.float16 if use_cuda else torch.float32)
    print(f"  device={device}  dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kw = dict(
        local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=dtype,
    )
    if use_cuda and not force_fp32:
        load_kw["device_map"] = "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="eager", **load_kw)
    except (TypeError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)

    model.eval()
    if not use_cuda or force_fp32:
        model.to(device)

    n_layers = len(model.model.layers)
    print(f"  Layers={n_layers}")
    return model, tokenizer, device, n_layers


def unload_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    print("  Unloaded.")


# ============================================================
# VERB IDS
# ============================================================

def get_verb_ids(tokenizer):
    is_id  = tokenizer.encode(" " + SINGULAR_V, add_special_tokens=False)
    are_id = tokenizer.encode(" " + PLURAL_V,   add_special_tokens=False)
    assert len(is_id) == 1 and len(are_id) == 1, \
        f"Verb not single token: is={is_id} are={are_id}"
    return is_id[0], are_id[0]


# ============================================================
# TOKEN POSITION EXTRACTION
# ============================================================

def locate_word_end(tokenizer, prompt: str, word: str) -> int:
    """
    Return the index of the LAST subtoken of word in the tokenized prompt.
    Tries with and without a leading space.
    Returns -1 if not found.
    """
    ids = tokenizer(prompt, add_special_tokens=True,
                    return_tensors="pt")["input_ids"][0].tolist()
    for prefix in (" ", ""):
        wids = tokenizer.encode(prefix + word, add_special_tokens=False)
        n = len(wids)
        for i in range(len(ids) - n + 1):
            if ids[i:i+n] == wids:
                return i + n - 1
    return -1


def get_all_positions(tokenizer, prompt: str, subject: str, distractor: str):
    """
    Returns:
      all_positions: list of all non-special token position indices
      tok_strings:   list of role-based labels (subject/distractor/final/pos_N)
      role_map:      dict mapping position → role label
    """
    ids      = tokenizer(prompt, add_special_tokens=True,
                         return_tensors="pt")["input_ids"][0].tolist()

    # ── Fix 2: exclude BOS and all special tokens ─────────────────────────
    special_ids = set(tokenizer.all_special_ids)
    all_pos     = [i for i, tok_id in enumerate(ids)
                   if tok_id not in special_ids]

    # Final token is last non-special position
    final = all_pos[-1] if all_pos else len(ids) - 1

    subj_pos = locate_word_end(tokenizer, prompt, subject)
    dist_pos = locate_word_end(tokenizer, prompt, distractor)

    # ── Fix 3: warn if subject or distractor not found ────────────────────
    if subj_pos == -1:
        tqdm.write(f"  [WARN] subject not found in prompt: "
                   f"subject={repr(subject)}  prompt={repr(prompt)}")
    if dist_pos == -1:
        tqdm.write(f"  [WARN] distractor not found in prompt: "
                   f"distractor={repr(distractor)}  prompt={repr(prompt)}")

    # ── Fix 4: use clean role labels, not raw token strings ───────────────
    # Raw tokens like 'Ġmanager' or '▁manager' are messy in plots.
    # Use role labels for named positions; pos_N for others.
    tok_strs_raw = tokenizer.convert_ids_to_tokens(ids)

    role_map = {}
    for p in all_pos:
        if p == subj_pos:
            role_map[p] = "subject"
        elif p == dist_pos:
            role_map[p] = "distractor"
        elif p == final:
            role_map[p] = "final"
        else:
            role_map[p] = f"pos_{p}"

    # tok_strings: use role label for known positions, raw for others
    tok_strings = []
    for p in all_pos:
        role = role_map.get(p, f"pos_{p}")
        if role in ("subject", "distractor", "final"):
            tok_strings.append(role)
        else:
            # Clean up tokenizer artifacts (Ġ, ▁, ##) for display
            raw = tok_strs_raw[p] if p < len(tok_strs_raw) else "?"
            clean = raw.replace("Ġ", "").replace("▁", "").replace("##", "")
            tok_strings.append(clean if clean else raw)

    return all_pos, tok_strings, role_map


# ============================================================
# FORWARD PASS
# ============================================================

@torch.no_grad()
def get_D_score(model, tokenizer, device, prompt, is_id, are_id):
    enc    = tokenizer(prompt, return_tensors="pt",
                       add_special_tokens=True).to(device)
    out    = model(**enc)
    logits = out.logits.float()
    final  = enc["attention_mask"].sum() - 1
    return (logits[0, final, is_id] - logits[0, final, are_id]).item()


# ============================================================
# STEERING HOOK
# ============================================================

class SteeringHook:
    def __init__(self, model, layer_idx, token_pos, direction, alpha):
        self.model     = model
        self.layer_idx = int(layer_idx)
        self.token_pos = int(token_pos)
        self.direction = direction.detach().float().cpu()
        self.alpha     = float(alpha)
        self.handle    = None

    def __enter__(self):
        layer     = self.model.model.layers[self.layer_idx]
        tok_pos   = self.token_pos
        direction = self.direction
        alpha     = self.alpha

        def hook(module, inputs, output):
            h = output[0].clone() if isinstance(output, tuple) else output.clone()
            h[0, tok_pos, :] += alpha * direction.to(
                device=h.device, dtype=h.dtype)
            return (h,) + output[1:] if isinstance(output, tuple) else h

        self.handle = layer.register_forward_hook(hook)
        return self

    def __exit__(self, *args):
        if self.handle:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def steered_D(model, tokenizer, device,
              prompt, layer_idx, tok_pos, direction, alpha,
              is_id, are_id):
    enc      = tokenizer(prompt, return_tensors="pt",
                         add_special_tokens=True).to(device)
    final_p  = enc["attention_mask"].sum().item() - 1
    with SteeringHook(model, layer_idx, tok_pos, direction, alpha):
        out    = model(**enc)
        logits = out.logits.float()
    return (logits[0, final_p, is_id] - logits[0, final_p, are_id]).item()


# ============================================================
# LOAD SAVED DIRECTIONS
# ============================================================

def load_directions(model_tag: str):
    """Load subject-number directions saved by exp3_steering.py."""
    path = os.path.join(STEER_DIR, f"steering_directions_{model_tag}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Direction file not found: {path}\n"
            f"Run exp3_steering.py first to generate directions.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    dirs = data["directions"]["subject"]
    print(f"  Loaded directions from {path}  ({len(dirs)} layers)")
    return dirs


# ============================================================
# MAIN ALL-POSITION STEERING LOOP
# ============================================================

def run_allpos_steering(model, tokenizer, device, n_layers,
                         item_map, directions, is_id, are_id, model_tag):
    """
    For each item:
      - tokenize the SP prompt
      - extract ALL actual token positions
      - steer at each position × each layer with alpha=+1
      - record D_clean, D_steered, effect, signed_effect
    """
    rows   = []
    items  = list(item_map.items())
    layers = list(range(n_layers))

    print(f"\n{'='*70}")
    print(f"ALL-POSITION STEERING  [{model_tag}]")
    print(f"  {len(items)} items | {n_layers} layers | alpha={ALPHA} | target={TARGET_COND}")
    print(f"{'='*70}")

    for item_id, conds in tqdm(items, desc=f"[{model_tag}]", unit="item"):

        target_row    = conds[TARGET_COND]
        target_prompt = target_row["prompt"]
        subject       = (target_row["subject_singular"]
                         if TARGET_COND[0] == "S" else target_row["subject_plural"])
        distractor    = (target_row["distractor_singular"]
                         if TARGET_COND[1] == "S" else target_row["distractor_plural"])

        # Get ALL actual token positions from this prompt
        all_pos, tok_strings, role_map = get_all_positions(
            tokenizer, target_prompt, subject, distractor)
        n_tokens = len(all_pos)

        # Clean baseline
        base_D = get_D_score(model, tokenizer, device,
                              target_prompt, is_id, are_id)

        for layer_idx in tqdm(layers, desc="  layers", leave=False, unit="L"):
            if layer_idx not in directions:
                continue
            real_dir = directions[layer_idx]

            for pos_idx, tok_pos in enumerate(all_pos):
                d_s = steered_D(
                    model, tokenizer, device,
                    target_prompt,
                    layer_idx, tok_pos,
                    real_dir, ALPHA,
                    is_id, are_id,
                )

                effect        = d_s - base_D
                # Positive signed_effect = correct plural push (D should drop)
                signed_effect = -effect  # alpha>0, plural push lowers D

                tok_str  = tok_strings[pos_idx] if pos_idx < len(tok_strings) else "?"
                role     = role_map.get(tok_pos, f"pos{tok_pos}")

                rows.append({
                    "model_tag":      model_tag,
                    "item_id":        item_id,
                    "layer":          layer_idx,
                    "is_layer_0":     layer_idx == 0,
                    "token_pos":      tok_pos,
                    "token_string":   tok_str,
                    "role":           role,
                    "n_tokens":       n_tokens,
                    "alpha":          ALPHA,
                    "base_D":         base_D,
                    "d_steered":      d_s,
                    "effect":         effect,
                    "signed_effect":  signed_effect,
                })

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    return pd.DataFrame(rows)


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_mean_ci(values, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED, ci=95):
    rng  = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    lo    = np.percentile(boots, (100 - ci) / 2)
    hi    = np.percentile(boots, 100 - (100 - ci) / 2)
    return float(vals.mean()), float(lo), float(hi)


# ============================================================
# SUMMARISE
# ============================================================

def summarise(raw_df, model_tag):

    # ── Position profile: mean over layers 1+ ────────────────────────────
    profile_rows = []
    df_main = raw_df[raw_df["layer"] > 0]

    for (tok_pos, role), grp in df_main.groupby(["token_pos", "role"]):
        per_item  = grp.groupby("item_id")["signed_effect"].mean()
        m, lo, hi = bootstrap_mean_ci(per_item.values)
        peak_layer = int(grp.groupby("layer")["signed_effect"].mean().idxmax())
        # representative token string (most common across items)
        tok_str = grp["token_string"].mode()[0] if len(grp) > 0 else "?"
        profile_rows.append({
            "model_tag":           model_tag,
            "token_pos":           tok_pos,
            "token_string":        tok_str,
            "role":                role,
            "layers":              "1+",
            "signed_effect_mean":  m,
            "signed_effect_ci_lo": lo,
            "signed_effect_ci_hi": hi,
            "peak_layer":          peak_layer,
            "n_items":             grp["item_id"].nunique(),
        })
    profile_df = pd.DataFrame(profile_rows).sort_values("token_pos")

    # ── Per (token_pos, layer) summary ───────────────────────────────────
    summary_rows = []
    for (tok_pos, role, layer), grp in raw_df.groupby(
            ["token_pos", "role", "layer"]):
        per_item  = grp.groupby("item_id")["signed_effect"].mean()
        m, lo, hi = bootstrap_mean_ci(per_item.values)
        tok_str   = grp["token_string"].mode()[0]
        summary_rows.append({
            "model_tag":           model_tag,
            "token_pos":           tok_pos,
            "token_string":        tok_str,
            "role":                role,
            "layer":               layer,
            "is_layer_0":          layer == 0,
            "signed_effect_mean":  m,
            "signed_effect_ci_lo": lo,
            "signed_effect_ci_hi": hi,
            "n_items":             grp["item_id"].nunique(),
        })
    summary_df = pd.DataFrame(summary_rows)

    # ── Layer 0 ───────────────────────────────────────────────────────────
    df_l0  = raw_df[raw_df["layer"] == 0]
    l0_rows = []
    for (tok_pos, role), grp in df_l0.groupby(["token_pos", "role"]):
        per_item  = grp.groupby("item_id")["signed_effect"].mean()
        m, lo, hi = bootstrap_mean_ci(per_item.values)
        tok_str   = grp["token_string"].mode()[0]
        l0_rows.append({
            "model_tag":           model_tag,
            "token_pos":           tok_pos,
            "token_string":        tok_str,
            "role":                role,
            "layer":               0,
            "signed_effect_mean":  m,
            "signed_effect_ci_lo": lo,
            "signed_effect_ci_hi": hi,
        })
    layer0_df = pd.DataFrame(l0_rows).sort_values("token_pos")

    return profile_df, summary_df, layer0_df


# ============================================================
# PLOTS
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "figure.dpi": 150,
})


def _bar_color(role: str, base_color: str) -> str:
    if role == "subject":    return "#1a6faf"
    if role == "distractor": return "#e07b00"
    if role == "final":      return "#d62728"
    return base_color


def plot_position_profile(profile_df, layer0_df, model_tag):
    """
    Bar chart: signed_effect by token position (layers 1+).
    Subject=blue, distractor=orange, final=red, other=model color.
    Layer 0 in separate panel below.
    """
    name   = MODEL_DISPLAY.get(model_tag, model_tag)
    mcolor = MODEL_COLORS.get(model_tag, "#555")

    profile_df = profile_df.sort_values("token_pos")
    x      = np.arange(len(profile_df))
    labels = [f"{r['role']}\n({r['token_string']})"
              for _, r in profile_df.iterrows()]
    means  = profile_df["signed_effect_mean"].values
    lo_err = means - profile_df["signed_effect_ci_lo"].values
    hi_err = profile_df["signed_effect_ci_hi"].values - means
    colors = [_bar_color(r["role"], mcolor)
              for _, r in profile_df.iterrows()]

    fig, (ax_main, ax_l0) = plt.subplots(
        2, 1, figsize=(11, 7),
        gridspec_kw={"height_ratios": [3, 1]})

    ax_main.bar(x, means, 0.65, color=colors, alpha=0.85,
                edgecolor="white",
                yerr=[lo_err, hi_err], capsize=4,
                error_kw={"elinewidth": 1.3, "ecolor": "0.3"})
    ax_main.axhline(0, color="black", linewidth=0.9)
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(labels, fontsize=9)
    ax_main.set_ylabel("Signed effect (layers 1+)\npositive = correct plural push",
                        fontsize=10)
    ax_main.set_title(
        f"Causal steering effect by token position — {name}\n"
        f"Subject-number direction  |  α=+{ALPHA}  |  target=SP\n"
        f"Blue=subject  Orange=distractor  Red=final",
        fontsize=11, fontweight="bold")

    # Layer 0 panel
    if len(layer0_df) > 0:
        l0 = layer0_df.sort_values("token_pos")
        l0_means = l0["signed_effect_mean"].values
        l0_colors = [_bar_color(r["role"], mcolor) for _, r in l0.iterrows()]
        ax_l0.bar(x, l0_means, 0.65, color=l0_colors,
                  alpha=0.45, edgecolor="white", hatch="///")
        ax_l0.axhline(0, color="black", linewidth=0.8)
        ax_l0.set_xticks(x)
        ax_l0.set_xticklabels(labels, fontsize=8)
        ax_l0.set_ylabel("Layer 0\n(embedding)", fontsize=9)
        ax_l0.set_facecolor("#fffde7")
        ax_l0.set_title("Layer 0 — embedding-level (reported separately)",
                         fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"plot_allpos_profile_{model_tag}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_heatmap(summary_df, model_tag):
    """Heatmap: token_pos × layer, signed_effect. Layers 1+ only."""
    name   = MODEL_DISPLAY.get(model_tag, model_tag)
    sub    = summary_df[~summary_df["is_layer_0"]].copy()

    positions = sorted(sub["token_pos"].unique())
    layers    = sorted(sub["layer"].unique())
    pos_label = {int(r["token_pos"]): f"{r['role']}\n({r['token_string']})"
                 for _, r in sub.drop_duplicates("token_pos").iterrows()}

    matrix  = np.full((len(positions), len(layers)), np.nan)
    pos_idx = {p: i for i, p in enumerate(positions)}
    lay_idx = {l: i for i, l in enumerate(layers)}

    for _, row in sub.iterrows():
        matrix[pos_idx[row["token_pos"]], lay_idx[row["layer"]]] = \
            row["signed_effect_mean"]

    vmax = np.nanpercentile(np.abs(matrix), 95)
    fig, ax = plt.subplots(figsize=(max(10, len(layers)//2), 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, origin="upper")

    ax.set_yticks(range(len(positions)))
    ax.set_yticklabels([pos_label.get(p, f"pos{p}") for p in positions],
                        fontsize=9)

    step = max(1, len(layers) // 10)
    ax.set_xticks(range(0, len(layers), step))
    ax.set_xticklabels(layers[::step], fontsize=8)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Token position", fontsize=10)
    ax.set_title(f"Steering effect heatmap — {name}\n"
                  f"Subject-number direction  |  α=+{ALPHA}  |  layers 1+",
                  fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Signed effect")
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"plot_allpos_heatmap_{model_tag}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_all_models_comparison(all_profiles: dict):
    """
    Side-by-side position profiles for all three models.
    Key figure: Qwen sharp subject peak vs Llama gradual rise.
    """
    tags = [t for t in ["phi2", "llama32_3b", "qwen25_3b"]
            if t in all_profiles]
    if len(tags) < 2:
        return

    fig, axes = plt.subplots(1, len(tags),
                              figsize=(5.5 * len(tags), 5),
                              sharey=False)
    if len(tags) == 1:
        axes = [axes]

    for ax, tag in zip(axes, tags):
        prof   = all_profiles[tag].sort_values("token_pos")
        name   = MODEL_DISPLAY.get(tag, tag)
        mcolor = MODEL_COLORS.get(tag, "#555")
        x      = np.arange(len(prof))
        labels = [f"{r['role']}\n({r['token_string']})"
                  for _, r in prof.iterrows()]
        means  = prof["signed_effect_mean"].values
        lo_err = means - prof["signed_effect_ci_lo"].values
        hi_err = prof["signed_effect_ci_hi"].values - means
        colors = [_bar_color(r["role"], mcolor) for _, r in prof.iterrows()]

        ax.bar(x, means, 0.65, color=colors, alpha=0.82,
               edgecolor="white",
               yerr=[lo_err, hi_err], capsize=3,
               error_kw={"elinewidth": 1.2, "ecolor": "0.3"})
        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        ax.set_title(name, fontsize=12, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Signed effect (layers 1+)", fontsize=9)

    fig.suptitle(
        "Causal steering effect by token position — all models\n"
        "Subject-number direction  |  α=+1  |  target=SP\n"
        "Blue=subject  Orange=distractor  Red=final",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plot_allmodels_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# PAPER NUMBERS
# ============================================================

def write_paper_numbers(profile_df, layer0_df, model_tag):
    name  = MODEL_DISPLAY.get(model_tag, model_tag)
    lines = []
    lines.append(f"ALL-POSITION STEERING PAPER NUMBERS — {name}")
    lines.append("=" * 70)
    lines.append(f"Subject-number direction | α=+{ALPHA} | layers 1+")
    lines.append(f"Target: {TARGET_COND} (singular subject + plural distractor)")
    lines.append("")
    lines.append(f"{'Pos':>4}  {'Role':<12}  {'Token':<12}  "
                  f"{'Effect':>9}  {'CI_lo':>9}  {'CI_hi':>9}  {'PeakL':>6}")
    lines.append("-" * 70)

    for _, r in profile_df.sort_values("token_pos").iterrows():
        lines.append(
            f"{int(r['token_pos']):>4}  {str(r['role']):<12}  "
            f"{str(r['token_string']):<12}  "
            f"{r['signed_effect_mean']:>9.4f}  "
            f"{r['signed_effect_ci_lo']:>9.4f}  "
            f"{r['signed_effect_ci_hi']:>9.4f}  "
            f"{int(r['peak_layer']):>6}")

    # Key ratio: subject vs final
    idx = profile_df.set_index("role")
    lines.append("")
    lines.append("Key ratios (subject / final):")
    if "subject" in idx.index and "final" in idx.index:
        s = float(idx.loc["subject", "signed_effect_mean"]) if "subject" in idx.index else None
        f = float(idx.loc["final",   "signed_effect_mean"]) if "final"   in idx.index else None
        if s is not None and f is not None and f != 0:
            lines.append(f"  subject={s:.4f}  final={f:.4f}  ratio={s/f:.3f}x")
            lines.append(f"  Pattern: {'SOURCE-LOCALIZED (subj>final)' if s>f else 'FINAL-TOKEN INTEGRATION (final>=subj)'}")

    lines.append("")
    lines.append("Layer 0 (embedding-level, separate):")
    for _, r in layer0_df.sort_values("token_pos").iterrows():
        lines.append(
            f"  pos{int(r['token_pos'])}  {str(r['role']):<12}  "
            f"{r['signed_effect_mean']:.4f} "
            f"[{r['signed_effect_ci_lo']:.4f},{r['signed_effect_ci_hi']:.4f}]")

    txt = "\n".join(lines)
    path = os.path.join(OUTPUT_DIR, f"test3_paper_numbers_{model_tag}.txt")
    with open(path, "w") as f:
        f.write(txt)
    print(f"Saved: {path}")
    print(txt)


# ============================================================
# MAIN
# ============================================================

def main():
    torch.set_grad_enabled(False)

    # ── GPU array support ─────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_idx", type=int, default=None,
                        help="0=phi2  1=llama  2=qwen. "
                             "Falls back to SLURM_ARRAY_TASK_ID, "
                             "then runs all models if neither set.")
    args = parser.parse_args()

    task_id = args.model_idx
    if task_id is None:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"]) \
                  if "SLURM_ARRAY_TASK_ID" in os.environ else None

    if task_id is not None:
        if task_id < 0 or task_id >= len(MODELS):
            raise ValueError(f"model_idx {task_id} out of range 0–{len(MODELS)-1}")
        models_to_run = [MODELS[task_id]]
        print(f"\nGPU array mode: task {task_id} → {models_to_run[0]['tag']}")
    else:
        models_to_run = MODELS
        print("\nRunning all models sequentially.")

    df       = load_clean_items()
    item_map = build_item_condition_map(df)

    # Collect profiles for cross-model plot
    # Load any already-finished profiles from disk first
    all_profiles = {}
    for tag in ["phi2", "llama32_3b", "qwen25_3b"]:
        p = os.path.join(OUTPUT_DIR,
                         f"test3_allpos_position_profile_{tag}.csv")
        if os.path.exists(p):
            all_profiles[tag] = pd.read_csv(p)
            print(f"  Loaded existing profile: {p}")

    for model_cfg in models_to_run:
        model_id   = model_cfg["model_id"]
        model_tag  = model_cfg["tag"]
        force_fp32 = model_cfg.get("force_fp32", False)

        model, tokenizer, device, n_layers = load_model(model_id, force_fp32)
        is_id, are_id = get_verb_ids(tokenizer)
        print(f"  is_id={is_id}  are_id={are_id}")

        directions = load_directions(model_tag)

        raw_df = run_allpos_steering(
            model, tokenizer, device, n_layers,
            item_map, directions, is_id, are_id, model_tag)

        raw_df.to_csv(
            os.path.join(OUTPUT_DIR, f"test3_allpos_raw_{model_tag}.csv"),
            index=False)
        print(f"Saved raw: {len(raw_df)} rows")

        profile_df, summary_df, layer0_df = summarise(raw_df, model_tag)

        profile_df.to_csv(
            os.path.join(OUTPUT_DIR,
                         f"test3_allpos_position_profile_{model_tag}.csv"),
            index=False)
        summary_df.to_csv(
            os.path.join(OUTPUT_DIR,
                         f"test3_allpos_summary_by_layer_{model_tag}.csv"),
            index=False)
        layer0_df.to_csv(
            os.path.join(OUTPUT_DIR,
                         f"test3_layer0_{model_tag}.csv"),
            index=False)

        plot_position_profile(profile_df, layer0_df, model_tag)
        plot_heatmap(summary_df, model_tag)
        try:
            write_paper_numbers(profile_df, layer0_df, model_tag)
        except Exception as e:
            print(f"[WARN] write_paper_numbers failed for {model_tag}: {e}")
            print("[WARN] Continuing because raw/profile/summary files are already saved.")

        all_profiles[model_tag] = profile_df
        unload_model(model, tokenizer)

    # Cross-model figure (only if all 3 profiles available)
    if len(all_profiles) >= 2:
        plot_all_models_comparison(all_profiles)

    print(f"\nDONE — outputs in: {OUTPUT_DIR}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f:<60} {sz:>9,} bytes")


if __name__ == "__main__":
    main()