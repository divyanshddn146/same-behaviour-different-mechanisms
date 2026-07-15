"""
03_activation_patching_agreement.py
===================================
Activation patching for controlled agreement prompts.

This script compares interventions on the true subject and distractor positions
across three decoder-only models. It runs subject patches, distractor patches,
self-patch and same-number controls, crossed distractor patches, and noise
controls. Phi-2 is evaluated in float32 for numerical stability; Llama and
Qwen use float16 when CUDA is available.

Notes:
- Runs all 3 models (Phi-2, Llama-3.2-3B, Qwen2.5-3B).
- Default causal verb pair: be_present = is/are.
- Past-tense replication: be_past = was/were by changing only the CONFIG block
  and the preferred clean-item filename in load_clean_items().
- Patches END token of subject/distractor span, not start token.
- Signs follow D = logit(SINGULAR_V) - logit(PLURAL_V).
  Default paper run: logit(is) - logit(are).
  Past-tense replication: logit(was) - logit(were).
- Layer 0 is saved and reported separately (embedding-level replacement).

Inputs expected:
- agreement_dataset_outputs/clean_items_all_models_intersection.csv
- optionally behavioral_analysis_outputs/clean_be_present_all_models_intersection.csv
- for was/were replication: behavioral_analysis_outputs/clean_be_past_all_models_intersection.csv

Outputs:
Default is/are run:
- exp2_patching_outputs/patching_raw_by_item_layer_{model_tag}.csv
- exp2_patching_outputs/patching_summary_by_layer_{model_tag}.csv
- exp2_patching_outputs/patching_bootstrap_summary_{model_tag}.csv
- exp2_patching_outputs/patching_layer0_separate_{model_tag}.csv
- exp2_patching_outputs/plot_patching_layerwise_{model_tag}.png
- exp2_patching_outputs/plot_patching_summary_{model_tag}.png
- exp2_patching_outputs/patching_paper_numbers_{model_tag}.txt

was/were replication:
- same filenames under exp2_patching_outputs_was_were/
"""

import os
import gc
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

DATASET_DIR  = "agreement_dataset_outputs"
BEHAV_DIR    = "behavioral_analysis_outputs"
OUTPUT_DIR   = "exp2_patching_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOCAL_FILES_ONLY  = True
TRUST_REMOTE_CODE = True

# Run all three models.
# IMPORTANT:
# - Phi-2 must use float32 for numerical stability; fp16 produced NaN logits
#   in our setup.
# - Llama/Qwen use float16.
MODELS = [
    {
        "model_id": "microsoft/phi-2",
        "tag": "phi2",
        "dtype": "float32",
    },
    {
        "model_id": "meta-llama/Llama-3.2-3B",
        "tag": "llama32_3b",
        "dtype": "float16",
    },
    {
        "model_id": "Qwen/Qwen2.5-3B",
        "tag": "qwen25_3b",
        "dtype": "float16",
    },
]

# Main causal patching verb pair.
VERB_PAIR   = "be_present"
SINGULAR_V  = "is"
PLURAL_V    = "are"

# ------------------------------------------------------------------
# To reproduce the was/were patching replication, copy this script and
# change the CONFIG values below and the preferred clean-item filename:
#
# OUTPUT_DIR  = "exp2_patching_outputs_was_were"
# VERB_PAIR   = "be_past"
# SINGULAR_V  = "was"
# PLURAL_V    = "were"
#
# Also change the preferred clean-item file in load_clean_items()
# from:
#   clean_be_present_all_models_intersection.csv
# to:
#   clean_be_past_all_models_intersection.csv
#
# No patching logic changes are required for the was/were replication.
# ------------------------------------------------------------------

# Bootstrap for summaries only.
N_BOOTSTRAP    = 2000
BOOTSTRAP_SEED = 42

# Noise control.
N_NOISE_SEEDS  = 3

# Optional quick debug. Keep None for final full run.
MAX_ITEMS = None

MODEL_DISPLAY = {
    "phi2":       "Phi-2",
    "llama32_3b": "Llama-3.2-3B",
    "qwen25_3b":  "Qwen2.5-3B",
}

# ============================================================
# DATA LOADING
# ============================================================

def load_clean_items() -> pd.DataFrame:
    """Load the selected verb-pair intersection dataset."""
    # Default paper run uses the present-tense clean intersection.
    # For the was/were replication, change this filename to:
    # "clean_be_past_all_models_intersection.csv"
    path = os.path.join(BEHAV_DIR, "clean_be_present_all_models_intersection.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        path2 = os.path.join(DATASET_DIR, "clean_items_all_models_intersection.csv")
        if not os.path.exists(path2):
            raise FileNotFoundError(
                f"Missing {path2}. Run make_dataset.py and behavioral_analysis.py first."
            )
        df = pd.read_csv(path2)

    df = df[df["verb_pair"] == VERB_PAIR].copy()
    df = df.drop_duplicates(subset=["item_id", "condition"])

    if MAX_ITEMS is not None:
        keep = sorted(df["item_id"].unique())[:MAX_ITEMS]
        df = df[df["item_id"].isin(keep)].copy()

    print(f"Loaded {df['item_id'].nunique()} clean base items for patching.", flush=True)
    return df


def build_item_condition_map(df: pd.DataFrame) -> dict:
    """
    Returns dict: item_id -> {condition -> row_dict}.
    Keeps only items with SS/SP/PS/PP.
    """
    item_map = {}
    for item_id, grp in df.groupby("item_id"):
        item_map[item_id] = {}
        for _, row in grp.iterrows():
            item_map[item_id][row["condition"]] = row.to_dict()

    complete = {k: v for k, v in item_map.items()
                if all(c in v for c in ["SS", "SP", "PS", "PP"])}
    print(f"Items with all 4 conditions: {len(complete)}", flush=True)
    return complete


# ============================================================
# MODEL LOADING
# ============================================================

def safe_cuda() -> bool:
    try:
        if torch.cuda.is_available():
            torch.zeros(1).cuda()
            return True
    except RuntimeError as e:
        print(f"CUDA not usable: {e}", flush=True)
    return False


def dtype_from_name(dtype_name: str):
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_name}. Use float32 or float16.")


def load_model(model_id: str, dtype_name: str):
    print(f"\n{'='*80}\nLOADING: {model_id} | dtype={dtype_name}\n{'='*80}", flush=True)

    use_cuda = safe_cuda()
    device   = torch.device("cuda" if use_cuda else "cpu")
    dtype    = dtype_from_name(dtype_name) if use_cuda else torch.float32

    print(f"  device={device}  dtype={dtype}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map="auto" is kept for GPU memory safety on smaller cards.
    # Hooks use hidden.device, so patch vectors are moved to the correct device.
    base_kw = dict(
        local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    if use_cuda:
        base_kw["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="eager",
            dtype=dtype,
            **base_kw,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="eager",
            torch_dtype=dtype,
            **base_kw,
        )
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            **base_kw,
        )

    model.eval()

    if not use_cuda:
        model.to(device)

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError("This script expects decoder layers at model.model.layers.")

    n_layers = len(model.model.layers)
    n_heads  = getattr(model.config, "num_attention_heads", "?")
    print(f"  Layers={n_layers}  Heads={n_heads}", flush=True)
    return model, tokenizer, device, n_layers


def unload_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    print("  Unloaded.", flush=True)


# ============================================================
# TOKEN HELPERS
# ============================================================

def get_verb_ids(tokenizer) -> tuple[int, int]:
    singular_ids = tokenizer.encode(" " + SINGULAR_V, add_special_tokens=False)
    plural_ids   = tokenizer.encode(" " + PLURAL_V,   add_special_tokens=False)
    assert len(singular_ids) == 1 and len(plural_ids) == 1, \
        f"Verb not single-token: {SINGULAR_V}={singular_ids} {PLURAL_V}={plural_ids}"
    return singular_ids[0], plural_ids[0]


def locate_token(tokenizer, prompt: str, word: str) -> int:
    """
    Return END position of word in tokenized prompt.
    This patches the final subtoken of the noun span.
    -1 if not found.
    """
    ids = tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"][0].tolist()

    for prefix in (" ", ""):
        wids = tokenizer.encode(prefix + word, add_special_tokens=False)
        n = len(wids)
        for i in range(len(ids) - n + 1):
            if ids[i:i+n] == wids:
                return i + n - 1

    return -1


# ============================================================
# FORWARD + HIDDEN STATE EXTRACTION
# ============================================================

def move_batch_to_device(enc, device):
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def get_hidden_states(model, tokenizer, device, prompt: str):
    """
    Returns hidden states after each transformer layer:
      hidden[layer_idx] = output of model.model.layers[layer_idx]
    """
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    enc = move_batch_to_device(enc, device)

    out = model(**enc, output_hidden_states=True)

    # hidden_states[0] = embeddings
    # hidden_states[1] = after layer 0
    hidden = [h.detach().float().cpu() for h in out.hidden_states[1:]]

    return hidden, enc["input_ids"].detach().cpu(), out.logits.detach().float().cpu()


@torch.no_grad()
def get_D_score(model, tokenizer, device, prompt: str,
                singular_id: int, plural_id: int) -> float:
    """Clean D = logit(SINGULAR_V) - logit(PLURAL_V) at final token."""
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    enc = move_batch_to_device(enc, device)

    out = model(**enc)
    logits = out.logits.float()

    final = int(enc["attention_mask"][0].sum().item()) - 1
    val = logits[0, final, singular_id] - logits[0, final, plural_id]

    if torch.isnan(val):
        return float("nan")
    return float(val.item())


# ============================================================
# PATCHING ENGINE
# ============================================================

class HiddenStatePatcher:
    """
    Patches the residual stream at a specific (layer, token_position)
    by replacing the hidden state with a donor vector.
    """

    def __init__(self, model, layer_idx: int, token_pos: int, donor_vec: torch.Tensor):
        self.model     = model
        self.layer_idx = layer_idx
        self.token_pos = token_pos
        self.donor_vec = donor_vec.detach().float().cpu()
        self.handle    = None

    def __enter__(self):
        layer = self.model.model.layers[self.layer_idx]
        token_pos = self.token_pos
        donor_vec = self.donor_vec

        def hook(module, inputs, output):
            # HF Llama/Qwen/Phi layers usually return tuple(hidden_states, ...)
            if isinstance(output, tuple):
                hidden = output[0]
                new_hidden = hidden.clone()
                new_hidden[0, token_pos, :] = donor_vec.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                return (new_hidden,) + output[1:]

            hidden = output
            new_hidden = hidden.clone()
            new_hidden[0, token_pos, :] = donor_vec.to(
                device=hidden.device,
                dtype=hidden.dtype,
            )
            return new_hidden

        self.handle = layer.register_forward_hook(hook)
        return self

    def __exit__(self, *args):
        if self.handle:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def patched_D_score(model, tokenizer, device,
                    recipient_prompt: str,
                    donor_vec: torch.Tensor,
                    patch_token_pos: int,
                    patch_layer: int,
                    singular_id: int, plural_id: int) -> float:
    """
    Run recipient_prompt with donor_vec patched into
    (patch_layer, patch_token_pos). Return D = logit(SINGULAR_V) - logit(PLURAL_V).
    """
    enc = tokenizer(recipient_prompt, return_tensors="pt", add_special_tokens=True)
    enc = move_batch_to_device(enc, device)

    final_pos = int(enc["attention_mask"][0].sum().item()) - 1

    with HiddenStatePatcher(model, patch_layer, patch_token_pos, donor_vec):
        out = model(**enc)
        logits = out.logits.float()

    val = logits[0, final_pos, singular_id] - logits[0, final_pos, plural_id]

    if torch.isnan(val):
        return float("nan")
    return float(val.item())


# ============================================================
# PATCHING EXPERIMENT DEFINITIONS
# ============================================================
#
# D = logit(SINGULAR_V) - logit(PLURAL_V)
# Positive D/effect = more singular-biased.
# Negative D/effect = more plural-biased.
#
# expected_sign:
#   +1 if patch should increase D
#   -1 if patch should decrease D
#    0 for self-patch controls

PATCH_EXPERIMENTS = [
    # Subject patching
    {
        "name":           "subj_sg_to_pl",
        "group":          "subject_patch",
        "donor_cond":     "SS",
        "recipient_cond": "PS",
        "patch_token":    "subject",
        "expected_sign":  +1,
        "description":    "Patch singular-subject repr into plural-subject prompt",
    },
    {
        "name":           "subj_pl_to_sg",
        "group":          "subject_patch",
        "donor_cond":     "PS",
        "recipient_cond": "SS",
        "patch_token":    "subject",
        "expected_sign":  -1,
        "description":    "Patch plural-subject repr into singular-subject prompt",
    },

    # Same-subject distractor patching
    {
        "name":           "dist_sg_to_pl",
        "group":          "distractor_patch",
        "donor_cond":     "SS",
        "recipient_cond": "SP",
        "patch_token":    "distractor",
        "expected_sign":  +1,
        "description":    "Patch singular-distractor repr into plural-distractor prompt",
    },
    {
        "name":           "dist_pl_to_sg",
        "group":          "distractor_patch",
        "donor_cond":     "PP",
        "recipient_cond": "PS",
        "patch_token":    "distractor",
        "expected_sign":  -1,
        "description":    "Patch plural-distractor repr into singular-distractor prompt",
    },

    # Self-patch sanity controls
    {
        "name":           "self_patch_SS",
        "group":          "self_patch",
        "donor_cond":     "SS",
        "recipient_cond": "SS",
        "patch_token":    "subject",
        "expected_sign":  0,
        "description":    "Self-patch SS subject",
    },
    {
        "name":           "self_patch_SP",
        "group":          "self_patch",
        "donor_cond":     "SP",
        "recipient_cond": "SP",
        "patch_token":    "distractor",
        "expected_sign":  0,
        "description":    "Self-patch SP distractor",
    },

    # Same-number / lexical controls
    {
        "name":           "same_num_SS_to_PP_subj",
        "group":          "same_number_control",
        "donor_cond":     "SS",
        "recipient_cond": "PP",
        "patch_token":    "subject",
        "expected_sign":  +1,
        "description":    "SS subject into PP",
    },
    {
        "name":           "same_num_SP_to_PS_dist",
        "group":          "same_number_control",
        "donor_cond":     "SP",
        "recipient_cond": "PS",
        "patch_token":    "distractor",
        "expected_sign":  -1,
        "description":    "SP distractor into PS",
    },

    # Crossed distractor patching
    {
        "name":           "crossed_dist_SP_to_PS",
        "group":          "crossed_distractor_patch",
        "donor_cond":     "SP",
        "recipient_cond": "PS",
        "patch_token":    "distractor",
        "expected_sign":  -1,
        "description":    "SP distractor into PS; crossed condition",
    },
    {
        "name":           "crossed_dist_SS_to_PP",
        "group":          "crossed_distractor_patch",
        "donor_cond":     "SS",
        "recipient_cond": "PP",
        "patch_token":    "distractor",
        "expected_sign":  +1,
        "description":    "SS distractor into PP; crossed condition",
    },
]


# ============================================================
# NOISE PATCH CONTROL
# ============================================================

@torch.no_grad()
def noise_patched_D_score(model, tokenizer, device,
                           recipient_prompt: str,
                           clean_hidden: torch.Tensor,
                           patch_token_pos: int,
                           patch_layer: int,
                           singular_id: int, plural_id: int,
                           noise_seed: int) -> float:
    """
    Replace hidden state at (layer, token_pos) with Gaussian noise
    matched to clean_hidden mean/std.
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(noise_seed)

    clean_cpu = clean_hidden.detach().float().cpu()
    mean_val = clean_cpu.mean().item()
    std_val = clean_cpu.std().item()

    noise_vec = torch.normal(
        mean=mean_val,
        std=std_val,
        size=clean_cpu.shape,
        generator=rng,
    ).float()

    return patched_D_score(
        model, tokenizer, device,
        recipient_prompt, noise_vec,
        patch_token_pos, patch_layer,
        singular_id, plural_id,
    )


# ============================================================
# MAIN PATCHING LOOP
# ============================================================

def get_word_for(row: dict, cond: str, token_type: str) -> str:
    if token_type == "subject":
        return row["subject_singular"] if cond[0] == "S" else row["subject_plural"]
    if token_type == "distractor":
        return row["distractor_singular"] if cond[1] == "S" else row["distractor_plural"]
    raise ValueError(f"Unknown token_type={token_type}")


def run_patching_for_model(model, tokenizer, device, n_layers: int,
                            item_map: dict, singular_id: int, plural_id: int,
                            model_tag: str) -> pd.DataFrame:
    all_rows = []
    items = list(item_map.items())
    n_items = len(items)
    layers = list(range(n_layers))

    print(f"\n{'='*80}", flush=True)
    print(f"PATCHING [{model_tag}] | {n_items} items | {n_layers} layers | "
          f"{len(PATCH_EXPERIMENTS)} experiments | noise seeds={N_NOISE_SEEDS}", flush=True)
    print(f"{'='*80}", flush=True)

    for item_idx, (item_id, conds) in enumerate(
        tqdm(items, desc=f"[{model_tag}] items", unit="item")
    ):
        # Clean D for all conditions
        clean_D = {}
        for cond, row in conds.items():
            clean_D[cond] = get_D_score(
                model, tokenizer, device, row["prompt"], singular_id, plural_id
            )

        if any(math.isnan(v) for v in clean_D.values()):
            tqdm.write(f"WARNING: NaN clean_D for {model_tag} item {item_id}; skipping item.")
            continue

        # Hidden states for all conditions
        cond_hidden = {}
        for cond, row in conds.items():
            hs, _, _ = get_hidden_states(model, tokenizer, device, row["prompt"])
            cond_hidden[cond] = hs

        for exp in PATCH_EXPERIMENTS:
            donor_cond     = exp["donor_cond"]
            recipient_cond = exp["recipient_cond"]
            patch_token    = exp["patch_token"]
            expected_sign  = exp["expected_sign"]

            donor_row = conds[donor_cond]
            recipient_row = conds[recipient_cond]

            donor_word = get_word_for(donor_row, donor_cond, patch_token)
            recipient_word = get_word_for(recipient_row, recipient_cond, patch_token)

            donor_tok_pos = locate_token(tokenizer, donor_row["prompt"], donor_word)
            rec_tok_pos = locate_token(tokenizer, recipient_row["prompt"], recipient_word)

            if donor_tok_pos < 0 or rec_tok_pos < 0:
                tqdm.write(
                    f"WARNING: token not found for {exp['name']} item={item_id} "
                    f"donor_word={donor_word} rec_word={recipient_word}"
                )
                continue

            for layer_idx in layers:
                donor_vec = cond_hidden[donor_cond][layer_idx][0, donor_tok_pos, :]

                p_D = patched_D_score(
                    model, tokenizer, device,
                    recipient_row["prompt"],
                    donor_vec,
                    rec_tok_pos,
                    layer_idx,
                    singular_id, plural_id,
                )

                if math.isnan(p_D):
                    tqdm.write(f"WARNING: NaN patched_D for {model_tag} item={item_id} exp={exp['name']} layer={layer_idx}")
                    continue

                effect = p_D - clean_D[recipient_cond]
                signed_effect = effect * expected_sign if expected_sign != 0 else effect

                noise_effects = []
                if N_NOISE_SEEDS > 0:
                    rec_vec = cond_hidden[recipient_cond][layer_idx][0, rec_tok_pos, :]
                    for ns in range(N_NOISE_SEEDS):
                        n_D = noise_patched_D_score(
                            model, tokenizer, device,
                            recipient_row["prompt"],
                            rec_vec,
                            rec_tok_pos, layer_idx,
                            singular_id, plural_id,
                            noise_seed=1000 * layer_idx + ns,
                        )
                        if not math.isnan(n_D):
                            noise_effects.append(n_D - clean_D[recipient_cond])

                noise_effect_mean = float(np.mean(noise_effects)) if noise_effects else 0.0

                all_rows.append({
                    "model_tag":          model_tag,
                    "item_id":            item_id,
                    "exp_name":           exp["name"],
                    "exp_group":          exp["group"],
                    "donor_cond":         donor_cond,
                    "recipient_cond":     recipient_cond,
                    "patch_token":        patch_token,
                    "expected_sign":      expected_sign,
                    "layer":              layer_idx,
                    "is_layer_0":         layer_idx == 0,
                    "donor_token_pos":    donor_tok_pos,
                    "recipient_token_pos":rec_tok_pos,
                    "clean_D_donor":      clean_D[donor_cond],
                    "clean_D_recipient":  clean_D[recipient_cond],
                    "patched_D":          p_D,
                    "effect":             effect,
                    "signed_effect":      signed_effect,
                    "noise_effect_mean":  noise_effect_mean,
                    "real_minus_noise":   signed_effect - abs(noise_effect_mean),
                })

        # Free hidden states per item.
        del cond_hidden
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    return pd.DataFrame(all_rows)


# ============================================================
# BOOTSTRAP CI
# ============================================================

def bootstrap_mean_ci(values, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED, ci=95):
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return float(vals.mean()), float(lo), float(hi)


# ============================================================
# SUMMARIZE RESULTS
# ============================================================

def summarize_patching(raw_df: pd.DataFrame, model_tag: str):
    by_layer_rows = []
    for (exp_name, exp_group, layer), grp in raw_df.groupby(
        ["exp_name", "exp_group", "layer"]
    ):
        se = grp["signed_effect"].values
        ne = grp["noise_effect_mean"].values

        m, lo, hi = bootstrap_mean_ci(se)
        nm, nlo, nhi = bootstrap_mean_ci(np.abs(ne))

        by_layer_rows.append({
            "model_tag":              model_tag,
            "exp_name":               exp_name,
            "exp_group":              exp_group,
            "layer":                  layer,
            "is_layer_0":             layer == 0,
            "signed_effect_mean":     m,
            "signed_effect_ci_lo":    lo,
            "signed_effect_ci_hi":    hi,
            "noise_effect_abs_mean":  nm,
            "noise_effect_abs_ci_lo": nlo,
            "noise_effect_abs_ci_hi": nhi,
            "n_items":                grp["item_id"].nunique(),
        })

    by_layer = pd.DataFrame(by_layer_rows)

    boot_rows = []
    df_no0 = raw_df[raw_df["layer"] > 0].copy()

    for (exp_name, exp_group), grp in df_no0.groupby(["exp_name", "exp_group"]):
        per_item = grp.groupby("item_id")["signed_effect"].mean()
        per_item_noise = grp.groupby("item_id")["noise_effect_mean"].apply(
            lambda x: np.abs(x).mean()
        )

        m, lo, hi = bootstrap_mean_ci(per_item.values)
        nm, nlo, nhi = bootstrap_mean_ci(per_item_noise.values)

        layer_means = grp.groupby("layer")["signed_effect"].mean()
        peak_layer = int(layer_means.idxmax())
        peak_val = float(layer_means.max())

        boot_rows.append({
            "model_tag":              model_tag,
            "exp_name":               exp_name,
            "exp_group":              exp_group,
            "layers_included":        "1+",
            "signed_effect_mean":     m,
            "signed_effect_ci_lo":    lo,
            "signed_effect_ci_hi":    hi,
            "noise_effect_abs_mean":  nm,
            "noise_effect_abs_ci_lo": nlo,
            "noise_effect_abs_ci_hi": nhi,
            "real_minus_noise":       m - nm,
            "peak_layer":             peak_layer,
            "peak_effect":            peak_val,
            "n_items":                grp["item_id"].nunique(),
        })

    boot_summary = pd.DataFrame(boot_rows)

    df_l0 = raw_df[raw_df["layer"] == 0].copy()
    l0_rows = []
    for (exp_name, exp_group), grp in df_l0.groupby(["exp_name", "exp_group"]):
        per_item = grp.groupby("item_id")["signed_effect"].mean()
        m, lo, hi = bootstrap_mean_ci(per_item.values)
        l0_rows.append({
            "model_tag":           model_tag,
            "exp_name":            exp_name,
            "exp_group":           exp_group,
            "layer":               0,
            "signed_effect_mean":  m,
            "signed_effect_ci_lo": lo,
            "signed_effect_ci_hi": hi,
            "note": "Layer 0 reflects embedding-level token replacement; report separately.",
        })

    layer0_df = pd.DataFrame(l0_rows)
    return by_layer, boot_summary, layer0_df


# ============================================================
# PLOTS
# ============================================================

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "figure.dpi":         150,
})

GROUP_COLORS = {
    "subject_patch":            "#1a6faf",
    "distractor_patch":         "#cc3311",
    "self_patch":               "#aaaaaa",
    "same_number_control":      "#ee7733",
    "crossed_distractor_patch": "#9467bd",
}

GROUP_LABELS = {
    "subject_patch":            "Subject patch",
    "distractor_patch":         "Distractor patch",
    "self_patch":               "Self-patch",
    "same_number_control":      "Same-number control",
    "crossed_distractor_patch": "Crossed distractor patch",
}

EXP_STYLE = {
    "subj_sg_to_pl":           {"ls": "-",  "marker": "o"},
    "subj_pl_to_sg":           {"ls": "--", "marker": "s"},
    "dist_sg_to_pl":           {"ls": "-",  "marker": "^"},
    "dist_pl_to_sg":           {"ls": "--", "marker": "v"},
    "self_patch_SS":           {"ls": ":",  "marker": "x"},
    "self_patch_SP":           {"ls": ":",  "marker": "x"},
    "same_num_SS_to_PP_subj":  {"ls": "-.", "marker": "D"},
    "same_num_SP_to_PS_dist":  {"ls": "-.", "marker": "P"},
    "crossed_dist_SP_to_PS":   {"ls": "-",  "marker": "h"},
    "crossed_dist_SS_to_PP":   {"ls": "--", "marker": "H"},
}


def plot_layerwise(by_layer: pd.DataFrame, model_tag: str, n_layers: int):
    groups_to_plot = [
        "subject_patch",
        "distractor_patch",
        "crossed_distractor_patch",
        "same_number_control",
        "self_patch",
    ]

    n_panels = len(groups_to_plot)
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        sharex=True,
    )
    axes_flat = axes.flatten()

    for ax, group in zip(axes_flat, groups_to_plot):
        grp_df = by_layer[by_layer["exp_group"] == group]
        exps = grp_df["exp_name"].unique()

        for exp in exps:
            edf = grp_df[grp_df["exp_name"] == exp].sort_values("layer")
            if edf.empty:
                continue

            style = EXP_STYLE.get(exp, {"ls": "-", "marker": "o"})
            color = GROUP_COLORS.get(group, "#333")

            ax.plot(
                edf["layer"], edf["signed_effect_mean"],
                linestyle=style["ls"],
                marker=style["marker"],
                markersize=4,
                color=color,
                alpha=0.85,
                label=exp,
                linewidth=1.4,
            )
            ax.fill_between(
                edf["layer"],
                edf["signed_effect_ci_lo"],
                edf["signed_effect_ci_hi"],
                alpha=0.12,
                color=color,
            )

        ax.axvspan(-0.5, 0.5, color="gold", alpha=0.25, label="Layer 0")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(GROUP_LABELS.get(group, group), fontsize=11, fontweight="bold")
        ax.set_ylabel("Signed effect (logit units)", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    for ax in axes_flat[(nrows - 1) * ncols:]:
        if ax.get_visible():
            ax.set_xlabel("Layer", fontsize=10)

    fig.suptitle(
        f"Activation patching — layer curves [{MODEL_DISPLAY.get(model_tag, model_tag)}]",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"plot_patching_layerwise_{model_tag}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}", flush=True)


def plot_summary_bars(boot_summary: pd.DataFrame, layer0_df: pd.DataFrame,
                      model_tag: str):
    main_exps = [e["name"] for e in PATCH_EXPERIMENTS]
    groups = {e["name"]: e["group"] for e in PATCH_EXPERIMENTS}

    boot_idx = boot_summary.set_index("exp_name")
    l0_idx = layer0_df.set_index("exp_name") if not layer0_df.empty else pd.DataFrame()

    fig, (ax_main, ax_l0) = plt.subplots(
        2, 1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    x = np.arange(len(main_exps))
    bar_w = 0.55
    colors = [GROUP_COLORS.get(groups.get(e, ""), "#666") for e in main_exps]

    means = [boot_idx.loc[e, "signed_effect_mean"] if e in boot_idx.index else 0.0
             for e in main_exps]
    lo_err = [means[i] - boot_idx.loc[e, "signed_effect_ci_lo"] if e in boot_idx.index else 0.0
              for i, e in enumerate(main_exps)]
    hi_err = [boot_idx.loc[e, "signed_effect_ci_hi"] - means[i] if e in boot_idx.index else 0.0
              for i, e in enumerate(main_exps)]
    noise = [boot_idx.loc[e, "noise_effect_abs_mean"] if e in boot_idx.index else 0.0
             for e in main_exps]

    ax_main.bar(
        x, means, bar_w,
        color=colors,
        alpha=0.82,
        edgecolor="white",
        yerr=[lo_err, hi_err],
        capsize=4,
        error_kw={"elinewidth": 1.3, "ecolor": "0.25"},
    )

    for xi, nv in zip(x, noise):
        ax_main.plot(
            [xi - bar_w/2, xi + bar_w/2],
            [nv, nv],
            color="black",
            linewidth=1.5,
            linestyle="--",
            alpha=0.6,
        )

    ax_main.axhline(0, color="black", linewidth=0.9)
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(main_exps, rotation=35, ha="right", fontsize=9)
    ax_main.set_ylabel("Signed effect — layers 1+ (logit units)", fontsize=10)
    ax_main.set_title(
        f"Patching effects (layers 1+) [{MODEL_DISPLAY.get(model_tag, model_tag)}]\n"
        "Dashed line = noise baseline",
        fontsize=11,
        fontweight="bold",
    )

    legend_patches = [
        mpatches.Patch(color=GROUP_COLORS[g], label=GROUP_LABELS.get(g, g), alpha=0.8)
        for g in ["subject_patch", "distractor_patch", "crossed_distractor_patch",
                  "self_patch", "same_number_control"]
    ]
    ax_main.legend(handles=legend_patches, fontsize=9, loc="upper right")

    if not l0_idx.empty:
        means_l0 = [l0_idx.loc[e, "signed_effect_mean"] if e in l0_idx.index else 0.0
                    for e in main_exps]
        lo_l0 = [means_l0[i] - l0_idx.loc[e, "signed_effect_ci_lo"] if e in l0_idx.index else 0.0
                 for i, e in enumerate(main_exps)]
        hi_l0 = [l0_idx.loc[e, "signed_effect_ci_hi"] - means_l0[i] if e in l0_idx.index else 0.0
                 for i, e in enumerate(main_exps)]

        ax_l0.bar(
            x, means_l0, bar_w,
            color=colors,
            alpha=0.5,
            edgecolor="white",
            hatch="///",
            yerr=[lo_l0, hi_l0],
            capsize=3,
            error_kw={"elinewidth": 1.1, "ecolor": "0.4"},
        )

    ax_l0.axhline(0, color="black", linewidth=0.8)
    ax_l0.set_xticks(x)
    ax_l0.set_xticklabels(main_exps, rotation=35, ha="right", fontsize=8)
    ax_l0.set_ylabel("Layer 0\nembedding", fontsize=9)
    ax_l0.set_facecolor("#fffde7")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"plot_patching_summary_{model_tag}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}", flush=True)


# ============================================================
# PAPER NUMBERS
# ============================================================

def write_paper_numbers(boot_summary: pd.DataFrame, layer0_df: pd.DataFrame,
                         model_tag: str):
    lines = []
    lines.append(f"PATCHING PAPER NUMBERS — {MODEL_DISPLAY.get(model_tag, model_tag)}")
    lines.append("=" * 80)
    lines.append("Layers 1+ (main results) [bootstrap 95% CI over items]")
    lines.append("")
    lines.append(f"{'Experiment':<30} {'Effect':>10} {'CI_lo':>10} {'CI_hi':>10} "
                 f"{'Noise':>10} {'Real-Noise':>12} {'Peak_L':>8}")
    lines.append("─" * 95)

    group_order = [
        "subject_patch",
        "distractor_patch",
        "crossed_distractor_patch",
        "self_patch",
        "same_number_control",
    ]

    for group in group_order:
        sub = boot_summary[boot_summary["exp_group"] == group]
        if sub.empty:
            continue
        lines.append(f"\n[{group}]")
        for _, r in sub.iterrows():
            lines.append(
                f"{r['exp_name']:<30} {r['signed_effect_mean']:>10.4f} "
                f"{r['signed_effect_ci_lo']:>10.4f} {r['signed_effect_ci_hi']:>10.4f} "
                f"{r['noise_effect_abs_mean']:>10.4f} {r['real_minus_noise']:>12.4f} "
                f"{int(r['peak_layer']):>8d}"
            )

    lines.append("")
    lines.append("─" * 95)
    lines.append("Layer 0 (reported separately — embedding-level replacement)")
    lines.append("")

    if not layer0_df.empty:
        for _, r in layer0_df.iterrows():
            lines.append(
                f"{r['exp_name']:<30} {r['signed_effect_mean']:>10.4f} "
                f"[{r['signed_effect_ci_lo']:.4f}, {r['signed_effect_ci_hi']:.4f}]"
            )

    lines.append("")
    lines.append("─" * 95)
    lines.append("Key comparisons (layers 1+):")

    idx = boot_summary.set_index("exp_name")
    for e1, e2, label in [
        ("subj_sg_to_pl", "dist_sg_to_pl", "Subject patch vs distractor patch (sg->pl)"),
        ("subj_pl_to_sg", "dist_pl_to_sg", "Subject patch vs distractor patch (pl->sg)"),
    ]:
        if e1 in idx.index and e2 in idx.index:
            v1 = float(idx.loc[e1, "signed_effect_mean"])
            v2 = float(idx.loc[e2, "signed_effect_mean"])
            lines.append(f"{label}")
            if abs(v2) > 1e-9:
                lines.append(f"  {e1}: {v1:.4f} | {e2}: {v2:.4f} | ratio: {v1/v2:.2f}x")
            else:
                lines.append(f"  {e1}: {v1:.4f} | {e2}: {v2:.4f}")

    path = os.path.join(OUTPUT_DIR, f"patching_paper_numbers_{model_tag}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved: {path}", flush=True)
    print("\n".join(lines), flush=True)


# ============================================================
# MAIN
# ============================================================

def main():
    torch.set_grad_enabled(False)

    print("\n" + "="*80, flush=True)
    print("LOADING CLEAN DATASET", flush=True)
    print("="*80, flush=True)

    df = load_clean_items()
    item_map = build_item_condition_map(df)

    for model_cfg in MODELS:
        model_id  = model_cfg["model_id"]
        model_tag = model_cfg["tag"]
        dtype_name = model_cfg.get("dtype", "float16")

        model, tokenizer, device, n_layers = load_model(model_id, dtype_name)
        singular_id, plural_id = get_verb_ids(tokenizer)

        print(f"  {SINGULAR_V}_id={singular_id}  {PLURAL_V}_id={plural_id}", flush=True)

        raw_df = run_patching_for_model(
            model, tokenizer, device, n_layers,
            item_map, singular_id, plural_id, model_tag,
        )

        raw_path = os.path.join(OUTPUT_DIR, f"patching_raw_by_item_layer_{model_tag}.csv")
        raw_df.to_csv(raw_path, index=False)
        print(f"\nSaved raw: {raw_path} ({len(raw_df)} rows)", flush=True)

        if raw_df.empty:
            print(f"WARNING: raw_df empty for {model_tag}; skipping summaries.", flush=True)
            unload_model(model, tokenizer)
            continue

        by_layer, boot_summary, layer0_df = summarize_patching(raw_df, model_tag)

        by_layer_path = os.path.join(OUTPUT_DIR, f"patching_summary_by_layer_{model_tag}.csv")
        boot_path = os.path.join(OUTPUT_DIR, f"patching_bootstrap_summary_{model_tag}.csv")
        layer0_path = os.path.join(OUTPUT_DIR, f"patching_layer0_separate_{model_tag}.csv")

        by_layer.to_csv(by_layer_path, index=False)
        boot_summary.to_csv(boot_path, index=False)
        layer0_df.to_csv(layer0_path, index=False)

        print(f"Saved: {by_layer_path}", flush=True)
        print(f"Saved: {boot_path}", flush=True)
        print(f"Saved: {layer0_path}", flush=True)

        plot_layerwise(by_layer, model_tag, n_layers)
        plot_summary_bars(boot_summary, layer0_df, model_tag)
        write_paper_numbers(boot_summary, layer0_df, model_tag)

        unload_model(model, tokenizer)

    print("\n" + "="*80, flush=True)
    print("DONE — outputs in:", OUTPUT_DIR, flush=True)
    print("="*80, flush=True)

    for f in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, f)
        try:
            sz = os.path.getsize(path)
            print(f"  {f:<60} {sz:>9,} bytes", flush=True)
        except OSError:
            pass


if __name__ == "__main__":
    main()
