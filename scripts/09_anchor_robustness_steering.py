"""
12_anchor_robustness_steering.py

Anchor-template robustness steering check.

Purpose:
  Test whether the agreement-control direction learned on the main
  "in question" template still works on alternate anchor/template prompts.

This check intentionally holds the auxiliary pair fixed to is/are so that the
only tested intervention is anchor/template variation. Auxiliary-pair robustness
is tested separately with the was/were steering, held-out, all-position, and
temporal analyses.

Runs ONE model per SLURM array task:
  0 = Phi-2
  1 = Llama-3.2-3B
  2 = Qwen2.5-3B

Train direction on:
  agreement_dataset_outputs/clean_items_all_models_intersection.csv

Test direction on:
  anchor robustness CSV. Edit ANCHOR_TEST_CSV below if needed.

Core measurement:
  d_subject[layer] = mean(hidden at subject token | plural subject)
                     - mean(hidden at subject token | singular subject)

  Apply this direction to anchor-variant prompts and measure change in:
      D = logit(is) - logit(are)

  Since the direction is plural - singular, +alpha should push toward plural "are",
  so D should decrease. Therefore:
      signed_effect = -(D_steered - D_clean)
  Positive signed_effect means steering works in the expected plural direction.

Recommended use in paper:
  Appendix / robustness only. This checks that the steering effect is not tied
  specifically to the phrase "in question".
"""

import os
import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIG
# ============================================================

MAIN_TRAIN_CSV = Path("data/agreement/clean_items_all_models_intersection.csv")
ANCHOR_TEST_CSV = Path("data/anchor_robustness/agreement_anchor_robustness_eval_balanced.csv")

# Fallback candidates, used if ANCHOR_TEST_CSV does not exist.
ANCHOR_CSV_CANDIDATES = [
    Path("anchor_robustness_outputs/agreement_anchor_robustness.csv"),
    Path("anchor_robustness_outputs/agreement_items_anchor_robust.csv"),
    Path("anchor_robustness_outputs/agreement_items_robust.csv"),
    Path("anchor_robustness_outputs/anchor_robustness_items.csv"),
    Path("agreement_anchor_robustness.csv"),
    Path("agreement_items_robust.csv"),
]

OUT_DIR = Path("results/anchor_robustness/is_are")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Run one model per array task:
#   SLURM_ARRAY_TASK_ID=0 -> Phi-2
#   SLURM_ARRAY_TASK_ID=1 -> Llama-3.2-3B
#   SLURM_ARRAY_TASK_ID=2 -> Qwen2.5-3B
#
# Default auxiliary pair is is/are:
#   AUX_PAIR = "be_present"
#   SINGULAR_V = "is"
#   PLURAL_V = "are"

MODELS = [
    {"model_id": "microsoft/phi-2", "tag": "phi2", "force_fp32": True},
    {"model_id": "meta-llama/Llama-3.2-3B", "tag": "llama32_3b", "force_fp32": False},
    {"model_id": "Qwen/Qwen2.5-3B", "tag": "qwen25_3b", "force_fp32": False},
]

LOCAL_FILES_ONLY = True
TRUST_REMOTE_CODE = True

# Reported anchor robustness intentionally holds the auxiliary pair fixed to
# is/are. This isolates template/anchor variation: the question is whether the
# subject/final localization pattern survives changing the anchor phrase, not
# whether it survives changing the auxiliary. Auxiliary-pair robustness is tested
# separately by the was/were steering, held-out, all-position, and temporal runs.
AUX_PAIR = "be_present"
SINGULAR_V = "is"
PLURAL_V = "are"

# Test the hard attraction condition by default.
# SP = singular subject + plural distractor. +plural direction should push toward "are".
TEST_CONDITIONS = ["SS", "SP", "PS", "PP"]

# Small robustness check: subject and final are enough.
# Do not run full all-position steering for anchor robustness.
POSITIONS = ["subject", "distractor", "final"]
ALPHA = 1.0

# Runtime limits. Set to None to use all rows.
MAX_TRAIN_ROWS = 1200          # balanced over singular/plural where possible
MAX_TEST_ROWS_PER_ANCHOR = 200 # per anchor/template and condition
SEED = 42

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 123

MODEL_DISPLAY = {
    "phi2": "Phi-2",
    "llama32_3b": "Llama-3.2-3B",
    "qwen25_3b": "Qwen2.5-3B",
}


# ============================================================
# ARRAY MODEL SELECTION
# ============================================================

def get_model_cfg():
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if task_id < 0 or task_id >= len(MODELS):
        raise ValueError(f"Bad SLURM_ARRAY_TASK_ID={task_id}; expected 0,1,2")

    cfg = MODELS[task_id]
    print("=" * 80)
    print("SLURM ARRAY MODEL SELECTION")
    print("=" * 80)
    print(f"SLURM_ARRAY_TASK_ID = {task_id}")
    print(f"model_id = {cfg['model_id']}")
    print(f"tag      = {cfg['tag']}")
    print(f"fp32     = {cfg.get('force_fp32', False)}")
    return cfg


# ============================================================
# DATA HELPERS
# ============================================================

def find_anchor_csv():
    if ANCHOR_TEST_CSV.exists():
        return ANCHOR_TEST_CSV
    for p in ANCHOR_CSV_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find anchor robustness CSV. Edit ANCHOR_TEST_CSV at the top of this script.\n"
        f"Tried: {[str(p) for p in [ANCHOR_TEST_CSV] + ANCHOR_CSV_CANDIDATES]}"
    )


def _pick_col(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None


def _normalise_num(x):
    s = str(x).strip().lower()
    if s in {"s", "sg", "sing", "singular", "0"}:
        return "sg"
    if s in {"p", "pl", "plural", "1"}:
        return "pl"
    raise ValueError(f"Cannot normalise number label: {x!r}")


def normalise_dataset(df, role):
    """
    Make main and anchor CSVs use the same minimal schema:
      aux_pair, condition, prompt, subject, subject_num, group_id, anchor_template

    This is intentionally tolerant because earlier scripts used slightly different names.
    """
    df = df.copy()

    # aux/verb pair column
    aux_col = _pick_col(df, ["aux_pair", "verb_pair", "pair_id_verb", "verb_pair_id"])
    if aux_col is None:
        raise ValueError(f"{role}: cannot find aux/verb pair column. Columns: {list(df.columns)}")
    df["aux_pair_norm"] = df[aux_col].astype(str)

    # condition column
    cond_col = _pick_col(df, ["condition", "cond"])
    if cond_col is None:
        raise ValueError(f"{role}: cannot find condition column. Columns: {list(df.columns)}")
    df["condition_norm"] = df[cond_col].astype(str)

    # prompt column
    prompt_col = _pick_col(df, ["prompt", "text", "input", "prefix"])
    if prompt_col is None:
        raise ValueError(f"{role}: cannot find prompt column. Columns: {list(df.columns)}")
    df["prompt_norm"] = df[prompt_col].astype(str)

    # subject number
    subj_num_col = _pick_col(df, ["subject_num", "subject_number", "subj_num", "subj_number"])
    if subj_num_col is not None:
        df["subject_num_norm"] = df[subj_num_col].map(_normalise_num)
    else:
        # Infer from condition first character.
        df["subject_num_norm"] = df["condition_norm"].str[0].map(_normalise_num)

    # subject word. Prefer explicit subject column.
    subj_col = _pick_col(df, ["subject", "subj", "subject_word", "subj_word"])
    if subj_col is not None:
        df["subject_norm"] = df[subj_col].astype(str)
    else:
        sg_col = _pick_col(df, ["subject_singular", "subj_sg", "subject_sg"])
        pl_col = _pick_col(df, ["subject_plural", "subj_pl", "subject_pl"])
        if sg_col is None or pl_col is None:
            raise ValueError(
                f"{role}: cannot find subject column or singular/plural subject columns. "
                f"Columns: {list(df.columns)}"
            )
        df["subject_norm"] = np.where(
            df["subject_num_norm"] == "sg",
            df[sg_col].astype(str),
            df[pl_col].astype(str),
        )

    # distractor word. Prefer explicit distractor column.
    dist_col = _pick_col(df, ["distractor", "dist", "distractor_word", "dist_word"])
    if dist_col is not None:
        df["distractor_norm"] = df[dist_col].astype(str)
    else:
        dist_sg_col = _pick_col(df, ["distractor_singular", "dist_sg", "distractor_sg"])
        dist_pl_col = _pick_col(df, ["distractor_plural", "dist_pl", "distractor_pl"])

        if dist_sg_col is None or dist_pl_col is None:
            raise ValueError(
                f"{role}: cannot find distractor column or singular/plural distractor columns. "
                f"Columns: {list(df.columns)}"
            )

        # Infer distractor number from second character of condition:
        # SS/SP/PS/PP -> condition[1]
        distractor_num = df["condition_norm"].str[1].map(_normalise_num)

        df["distractor_norm"] = np.where(
            distractor_num == "sg",
            df[dist_sg_col].astype(str),
            df[dist_pl_col].astype(str),
        )

    # group id for item-level bootstrap.
    gid_col = _pick_col(df, ["group_id", "item_id", "pair_id", "base_id", "example_id"])
    if gid_col is not None:
        df["group_id_norm"] = df[gid_col].astype(str)
    else:
        df["group_id_norm"] = [f"{role}_{i:06d}" for i in range(len(df))]

    # anchor/template label. If absent, create a single label.
    anchor_col = _pick_col(df, ["anchor", "anchor_template", "template", "template_id", "variant", "filler", "anchor_id"])
    if anchor_col is not None:
        df["anchor_template_norm"] = df[anchor_col].astype(str)
    else:
        df["anchor_template_norm"] = "anchor_variant"

    out = pd.DataFrame({
        "aux_pair": df["aux_pair_norm"],
        "condition": df["condition_norm"],
        "prompt": df["prompt_norm"],
        "subject": df["subject_norm"],
        "distractor": df["distractor_norm"],
        "subject_num": df["subject_num_norm"],
        "group_id": df["group_id_norm"],
        "anchor_template": df["anchor_template_norm"],
    })

    # Keep main aux pair only.
    out = out[out["aux_pair"] == AUX_PAIR].copy()
    if out.empty:
        raise ValueError(f"{role}: no rows after filtering aux_pair={AUX_PAIR}")

    return out


def balanced_sample_train(df):
    if MAX_TRAIN_ROWS is None or len(df) <= MAX_TRAIN_ROWS:
        return df.copy()

    rng = np.random.default_rng(SEED)
    parts = []
    per_num = max(1, MAX_TRAIN_ROWS // 2)

    for num, sub in df.groupby("subject_num"):
        take = min(len(sub), per_num)
        idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
        parts.append(sub.loc[idx])

    sampled = pd.concat(parts, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def sample_test(df):
    df = df[df["condition"].isin(TEST_CONDITIONS)].copy()
    if df.empty:
        raise ValueError(f"No test rows for TEST_CONDITIONS={TEST_CONDITIONS}")

    if MAX_TEST_ROWS_PER_ANCHOR is None:
        return df.reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    parts = []
    for (anchor, cond), sub in df.groupby(["anchor_template", "condition"]):
        take = min(len(sub), MAX_TEST_ROWS_PER_ANCHOR)
        idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
        parts.append(sub.loc[idx])

    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def load_data():
    if not MAIN_TRAIN_CSV.exists():
        raise FileNotFoundError(f"Missing main train CSV: {MAIN_TRAIN_CSV}")

    anchor_csv = find_anchor_csv()

    train_raw = pd.read_csv(MAIN_TRAIN_CSV)
    test_raw = pd.read_csv(anchor_csv)

    train = normalise_dataset(train_raw, role="train")
    test = normalise_dataset(test_raw, role="anchor_test")

    train = balanced_sample_train(train)
    test = sample_test(test)

    print("=" * 80)
    print("LOADED DATA")
    print("=" * 80)
    print(f"Train CSV: {MAIN_TRAIN_CSV}")
    print(f"Anchor CSV: {anchor_csv}")
    print(f"Train rows after filtering/sampling: {len(train)}")
    print(f"Test rows after filtering/sampling:  {len(test)}")
    print("\nTrain subject_num counts:")
    print(train["subject_num"].value_counts())
    print("\nTest counts:")
    print(test.groupby(["anchor_template", "condition"]).size())

    return train, test, anchor_csv


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_id, force_fp32=False):
    print("\n" + "=" * 80)
    print(f"LOADING MODEL: {model_id}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if force_fp32 else (
        torch.float16 if device.type == "cuda" else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=dtype,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="eager",
            **load_kwargs,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    model.eval()
    model.to(device)

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    n_layers = len(model.model.layers)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Layers: {n_layers}")

    return model, tokenizer, device, n_layers


def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# TOKEN HELPERS
# ============================================================

def get_verb_ids(tokenizer):
    sg = tokenizer.encode(" " + SINGULAR_V, add_special_tokens=False)
    pl = tokenizer.encode(" " + PLURAL_V, add_special_tokens=False)
    if len(sg) != 1 or len(pl) != 1:
        raise ValueError(f"Verb tokenization not single-token: {SINGULAR_V}={sg}, {PLURAL_V}={pl}")
    return sg[0], pl[0]


def locate_word_end(tokenizer, prompt, word):
    """Locate final subtoken position of word in prompt. Returns -1 if not found."""
    ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    candidates = [
        tokenizer.encode(" " + str(word), add_special_tokens=False),
        tokenizer.encode(str(word), add_special_tokens=False),
    ]

    for wids in candidates:
        n = len(wids)
        if n == 0:
            continue
        for i in range(len(ids) - n + 1):
            if ids[i:i + n] == wids:
                return i + n - 1
    return -1


def get_positions(tokenizer, row):
    prompt = row["prompt"]
    subject = row["subject"]
    distractor = row["distractor"]

    ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]

    return {
        "subject": locate_word_end(tokenizer, prompt, subject),
        "distractor": locate_word_end(tokenizer, prompt, distractor),
        "final": len(ids) - 1,
    }


# ============================================================
# FORWARD HELPERS
# ============================================================

@torch.no_grad()
def get_hidden_states(model, tokenizer, device, prompt):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    return [h.detach().float().squeeze(0).cpu() for h in out.hidden_states[1:]]


@torch.no_grad()
def get_D_score(model, tokenizer, device, prompt, sg_id, pl_id):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    out = model(**enc, use_cache=False)
    logits = out.logits.float()
    final = int(enc["attention_mask"].sum().item() - 1)
    return float((logits[0, final, sg_id] - logits[0, final, pl_id]).item())


# ============================================================
# DIRECTION CONSTRUCTION
# ============================================================

def build_subject_directions(train_df, model, tokenizer, device, n_layers, model_tag):
    """
    d_subject[layer] = mean(hidden at subject token | plural subject)
                       - mean(hidden at subject token | singular subject)
    Direction is learned from the main template dataset.
    """
    plural = [[] for _ in range(n_layers)]
    singular = [[] for _ in range(n_layers)]

    print("\n" + "=" * 80)
    print(f"BUILDING MAIN-TEMPLATE SUBJECT DIRECTIONS: {model_tag}")
    print("=" * 80)

    missed = 0
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="direction train rows"):
        pos = get_positions(tokenizer, row)
        s_pos = pos["subject"]
        if s_pos < 0:
            missed += 1
            continue

        hs = get_hidden_states(model, tokenizer, device, row["prompt"])
        is_plural = row["subject_num"] == "pl"

        for layer_idx, h in enumerate(hs):
            vec = h[s_pos].cpu()
            if is_plural:
                plural[layer_idx].append(vec)
            else:
                singular[layer_idx].append(vec)

    if missed:
        print(f"[WARN] Missed subject token in {missed} train rows")

    directions = {}
    for layer_idx in range(n_layers):
        if len(plural[layer_idx]) == 0 or len(singular[layer_idx]) == 0:
            raise RuntimeError(f"No vectors for layer {layer_idx}")
        p = torch.stack(plural[layer_idx]).mean(0)
        s = torch.stack(singular[layer_idx]).mean(0)
        directions[layer_idx] = p - s

    print("Directions built.")
    return directions


def build_random_directions(directions, seed=77):
    rng = np.random.default_rng(seed)
    random_dirs = {}
    for layer_idx, d in directions.items():
        norm = d.norm().item()
        v = torch.tensor(rng.standard_normal(d.shape[0]), dtype=torch.float32)
        v = v / v.norm() * norm
        random_dirs[layer_idx] = v
    return random_dirs


# ============================================================
# STEERING HOOK
# ============================================================

class SteeringHook:
    def __init__(self, model, layer_idx, token_pos, direction, alpha):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.token_pos = int(token_pos)
        self.direction = direction.detach().float().cpu()
        self.alpha = float(alpha)
        self.handle = None

    def __enter__(self):
        layer = self.model.model.layers[self.layer_idx]
        token_pos = self.token_pos
        direction = self.direction
        alpha = self.alpha

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                new_hidden = hidden.clone()
                new_hidden[0, token_pos, :] += alpha * direction.to(
                    device=new_hidden.device,
                    dtype=new_hidden.dtype,
                )
                return (new_hidden,) + output[1:]

            hidden = output
            new_hidden = hidden.clone()
            new_hidden[0, token_pos, :] += alpha * direction.to(
                device=new_hidden.device,
                dtype=new_hidden.dtype,
            )
            return new_hidden

        self.handle = layer.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def steered_D_score(model, tokenizer, device, prompt, layer_idx, token_pos, direction, alpha, sg_id, pl_id):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    final = int(enc["attention_mask"].sum().item() - 1)
    with SteeringHook(model, layer_idx, token_pos, direction, alpha):
        out = model(**enc, use_cache=False)
    logits = out.logits.float()
    return float((logits[0, final, sg_id] - logits[0, final, pl_id]).item())


# ============================================================
# STEERING LOOP
# ============================================================

def run_anchor_steering(test_df, model, tokenizer, device, n_layers, directions, random_dirs, sg_id, pl_id, model_tag):
    rows = []
    layers = list(range(n_layers))

    print("\n" + "=" * 80)
    print(f"RUNNING ANCHOR ROBUSTNESS STEERING: {model_tag}")
    print("=" * 80)
    print(f"Test rows: {len(test_df)} | layers: {n_layers} | positions: {POSITIONS}")

    missed = 0
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"{model_tag} anchor test"):
        prompt = row["prompt"]
        pos = get_positions(tokenizer, row)
        base_D = get_D_score(model, tokenizer, device, prompt, sg_id, pl_id)

        for position in POSITIONS:
            tok_pos = pos[position]
            if tok_pos < 0:
                missed += 1
                continue

            for layer_idx in layers:
                real_dir = directions[layer_idx]
                rand_dir = random_dirs[layer_idx]

                d_real = steered_D_score(
                    model, tokenizer, device,
                    prompt, layer_idx, tok_pos,
                    real_dir, ALPHA,
                    sg_id, pl_id,
                )
                d_rand = steered_D_score(
                    model, tokenizer, device,
                    prompt, layer_idx, tok_pos,
                    rand_dir, ALPHA,
                    sg_id, pl_id,
                )

                real_effect = d_real - base_D
                rand_effect = d_rand - base_D

                # Direction is plural - singular. +alpha should push plural "are",
                # so D=is-are should decrease. Positive signed effect = expected direction.
                signed_real = -real_effect
                signed_rand = -rand_effect

                rows.append({
                    "model_tag": model_tag,
                    "group_id": row["group_id"],
                    "anchor_template": row["anchor_template"],
                    "condition": row["condition"],
                    "position": position,
                    "layer": layer_idx,
                    "alpha": ALPHA,
                    "prompt": prompt,
                    "base_D": base_D,
                    "d_steered_real": d_real,
                    "d_steered_rand": d_rand,
                    "real_effect": real_effect,
                    "rand_effect": rand_effect,
                    "signed_real": signed_real,
                    "signed_rand": signed_rand,
                    "real_vs_rand": signed_real - abs(signed_rand),
                })

    if missed:
        print(f"[WARN] Missed token positions in {missed} test position checks")

    return pd.DataFrame(rows)


# ============================================================
# SUMMARIES
# ============================================================

def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarize(raw, model_tag):
    rows = []
    main = raw[raw["layer"] > 0].copy()

    # Overall by position
    for pos, grp in main.groupby("position"):
        per_item = grp.groupby("group_id")["signed_real"].mean()
        m, lo, hi = bootstrap_ci(per_item.values)
        per_item_r = grp.groupby("group_id")["signed_rand"].mean()
        rm, rlo, rhi = bootstrap_ci(per_item_r.values)
        layer_means = grp.groupby("layer")["signed_real"].mean()
        peak_layer = int(layer_means.idxmax())

        rows.append({
            "model_tag": model_tag,
            "summary_level": "overall",
            "anchor_template": "ALL",
            "condition": "+".join(TEST_CONDITIONS),
            "position": pos,
            "layers": "1+",
            "real_mean": m,
            "real_ci_lo": lo,
            "real_ci_hi": hi,
            "rand_mean": rm,
            "rand_ci_lo": rlo,
            "rand_ci_hi": rhi,
            "real_vs_rand": m - abs(rm),
            "peak_layer": peak_layer,
            "n_items": grp["group_id"].nunique(),
        })

    # By anchor/template and position
    for (anchor, cond, pos), grp in main.groupby(["anchor_template", "condition", "position"]):
        per_item = grp.groupby("group_id")["signed_real"].mean()
        m, lo, hi = bootstrap_ci(per_item.values)
        per_item_r = grp.groupby("group_id")["signed_rand"].mean()
        rm, rlo, rhi = bootstrap_ci(per_item_r.values)
        layer_means = grp.groupby("layer")["signed_real"].mean()
        peak_layer = int(layer_means.idxmax())

        rows.append({
            "model_tag": model_tag,
            "summary_level": "by_anchor",
            "anchor_template": anchor,
            "condition": cond,
            "position": pos,
            "layers": "1+",
            "real_mean": m,
            "real_ci_lo": lo,
            "real_ci_hi": hi,
            "rand_mean": rm,
            "rand_ci_lo": rlo,
            "rand_ci_hi": rhi,
            "real_vs_rand": m - abs(rm),
            "peak_layer": peak_layer,
            "n_items": grp["group_id"].nunique(),
        })

    summary = pd.DataFrame(rows)

    # Subject/final ratio for overall summary if both positions exist.
    ratio_rows = []
    overall = summary[summary["summary_level"] == "overall"].set_index("position")
    if "subject" in overall.index and "final" in overall.index:
        subj = float(overall.loc["subject", "real_mean"])
        final = float(overall.loc["final", "real_mean"])
        ratio = subj / final if abs(final) > 1e-8 else np.nan
        ratio_rows.append({
            "model_tag": model_tag,
            "anchor_template": "ALL",
            "condition": "+".join(TEST_CONDITIONS),
            "subject_effect": subj,
            "final_effect": final,
            "subject_final_ratio": ratio,
            "pattern": "source-localized" if ratio > 1 else "final-integrated",
        })

    ratios = pd.DataFrame(ratio_rows)
    return summary, ratios


def write_paper_numbers(summary, ratios, model_tag):
    lines = []
    lines.append(f"ANCHOR ROBUSTNESS STEERING — {MODEL_DISPLAY.get(model_tag, model_tag)}")
    lines.append("=" * 80)
    lines.append("Direction trained on main in-question dataset; tested on anchor variants")
    lines.append(f"Target conditions: {TEST_CONDITIONS} | alpha=+1 | layers 1+")
    lines.append("")
    lines.append("Overall effects:")
    lines.append(f"{'Position':<10} {'Effect':>10} {'CI_lo':>10} {'CI_hi':>10} {'Rand':>10} {'PeakL':>6} {'n':>6}")
    lines.append("-" * 80)

    overall = summary[summary["summary_level"] == "overall"]
    for pos in POSITIONS:
        r = overall[overall["position"] == pos]
        if r.empty:
            continue
        r = r.iloc[0]
        lines.append(
            f"{pos:<10} {r['real_mean']:>10.4f} {r['real_ci_lo']:>10.4f} {r['real_ci_hi']:>10.4f} "
            f"{r['rand_mean']:>10.4f} {int(r['peak_layer']):>6d} {int(r['n_items']):>6d}"
        )

    if not ratios.empty:
        r = ratios.iloc[0]
        lines.append("")
        lines.append(
            f"Subject/final ratio: {r['subject_final_ratio']:.3f} "
            f"({r['pattern']}; subject={r['subject_effect']:.4f}, final={r['final_effect']:.4f})"
        )

    path = OUT_DIR / f"anchor_steering_paper_numbers_{model_tag}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    torch.set_grad_enabled(False)

    cfg = get_model_cfg()
    model_id = cfg["model_id"]
    model_tag = cfg["tag"]
    force_fp32 = cfg.get("force_fp32", False)

    train_df, test_df, anchor_csv = load_data()

    # Save exact sampled data used, for reproducibility.
    train_df.to_csv(OUT_DIR / f"anchor_steering_train_sample_{model_tag}.csv", index=False)
    test_df.to_csv(OUT_DIR / f"anchor_steering_test_sample_{model_tag}.csv", index=False)

    model, tokenizer, device, n_layers = load_model(model_id, force_fp32)
    sg_id, pl_id = get_verb_ids(tokenizer)
    print(f"Token IDs: {SINGULAR_V}={sg_id}, {PLURAL_V}={pl_id}")

    directions = build_subject_directions(
        train_df=train_df,
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_layers=n_layers,
        model_tag=model_tag,
    )

    dir_path = OUT_DIR / f"main_template_subject_directions_{model_tag}.pkl"
    with open(dir_path, "wb") as f:
        pickle.dump({"directions": directions, "train_csv": str(MAIN_TRAIN_CSV)}, f)
    print(f"Saved directions: {dir_path}")

    random_dirs = build_random_directions(directions)

    raw = run_anchor_steering(
        test_df=test_df,
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_layers=n_layers,
        directions=directions,
        random_dirs=random_dirs,
        sg_id=sg_id,
        pl_id=pl_id,
        model_tag=model_tag,
    )

    raw_path = OUT_DIR / f"anchor_steering_raw_{model_tag}.csv"
    raw.to_csv(raw_path, index=False)
    print(f"Saved raw: {raw_path} rows={len(raw)}")

    summary, ratios = summarize(raw, model_tag)

    summary_path = OUT_DIR / f"anchor_steering_summary_{model_tag}.csv"
    ratios_path = OUT_DIR / f"anchor_steering_ratios_{model_tag}.csv"
    summary.to_csv(summary_path, index=False)
    ratios.to_csv(ratios_path, index=False)

    print(f"Saved summary: {summary_path}")
    print(f"Saved ratios: {ratios_path}")

    write_paper_numbers(summary, ratios, model_tag)

    unload_model(model, tokenizer)
    print("\nDONE")


if __name__ == "__main__":
    main()
