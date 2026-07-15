"""
04_activation_steering_agreement.py
===================================
Activation steering for controlled agreement prompts.

This script builds agreement-control directions (subject and distractor number
directions) and tests whether injecting them changes auxiliary-verb logits.
Real steering is compared against random matched-norm directions and
shuffled-label directions.

Core question
-------------
Where does number information become decision-controlling?
  At the subject token?
  At the distractor token?
  At the final prediction token?

Direction construction
----------------------
d_subject    = mean(subject hidden | plural-subject)
             − mean(subject hidden | singular-subject)

d_distractor = mean(distractor hidden | plural-distractor)
             − mean(distractor hidden | singular-distractor)

Built per-layer from the selected VERB_PAIR items.
Default paper run: be_present (is/are).
Past-tense replication: be_past (was/were) by changing only the CONFIG block.

Steering
--------
For each prompt, layer, position, direction, alpha:
  h_patched = h_clean + α · direction

Positions: subject token | distractor token | final token
Alpha:     [-1, 1]  (opposite-sign check)
D = logit(SINGULAR_V) − logit(PLURAL_V)
Default paper run: logit(is) − logit(are).
Past-tense replication: logit(was) − logit(were).

Controls
--------
1. Clean no-steering baseline D_clean for each item
2. Random direction  — same norm as real direction, random orientation
3. Shuffled-label direction — direction built with permuted plural/singular labels
4. Opposite-sign check — +d and −d should have opposite effects
5. Layer 0 reported separately

Bootstrap
---------
All effects bootstrapped over item_id (not item×layer pairs).

Outputs
-------
steering_directions_{model_tag}.pkl         ← saved directions per layer
steering_raw_{model_tag}.csv               ← full raw results
steering_summary_by_layer_{model_tag}.csv  ← mean effect per layer
steering_bootstrap_{model_tag}.csv         ← CIs (layers 1+)
steering_layer0_{model_tag}.csv            ← layer 0 separate
steering_alpha_sweep_{model_tag}.csv       ← alpha dose-response
plot_steering_by_layer_{model_tag}.png     ← Figure: layerwise curves
plot_steering_real_vs_controls_{model_tag}.png  ← Figure: real vs random/shuffled
plot_steering_alpha_sweep_{model_tag}.png  ← Figure: dose-response (final token)
steering_paper_numbers_{model_tag}.txt     ← copy-paste numbers
"""

import os
import gc
import pickle
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

LOCAL_FILES_ONLY  = True
TRUST_REMOTE_CODE = True

OUTPUT_DIR = "exp3_steering_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
    {"model_id": "microsoft/phi-2",          "tag": "phi2",       "force_fp32": True},
    {"model_id": "meta-llama/Llama-3.2-3B",  "tag": "llama32_3b", "force_fp32": False},
    {"model_id": "Qwen/Qwen2.5-3B",          "tag": "qwen25_3b",  "force_fp32": False},
]

VERB_PAIR  = "be_present"
SINGULAR_V = "is"
PLURAL_V   = "are"

# ------------------------------------------------------------------
# To reproduce the was/were replication, copy this script and change
# the CONFIG values below and the preferred clean-item filename:
#
# OUTPUT_DIR = "exp3_steering_outputs_was_were"
# VERB_PAIR  = "be_past"
# SINGULAR_V = "was"
# PLURAL_V   = "were"
#
# Also change the preferred clean-item file in load_clean_items()
# from:
#   clean_be_present_all_models_intersection.csv
# to:
#   clean_be_past_all_models_intersection.csv
#
# No steering logic changes are required for the was/were replication.
# ------------------------------------------------------------------

# Paper config: use both signs for alpha orientation check.
# The main table reports alpha = +1; alpha = -1 is used for the sign check.
ALPHAS     = [-1.0, 1.0]
MAIN_ALPHA = 1.0

# Paper config: compare where the subject-number direction is causally usable.
# These are the three positions reported in Table 2: subject, distractor, final.
POSITIONS  = ["subject", "distractor", "final"]

# Paper config: Table 2 uses the subject-number direction only.
# The code can also build distractor-number directions, but they are not needed
# for the main paper result.
DIRECTIONS = ["subject"]

# Keep controls minimal so code does not break, but runtime is much lower
N_SHUFFLES       = 1
SHUFFLE_SEED     = 99

N_RANDOM_DIRS    = 1
RANDOM_DIR_SEED  = 77

# Bootstrap
N_BOOTSTRAP    = 1000
BOOTSTRAP_SEED = 42

MODEL_DISPLAY = {
    "phi2": "Phi-2",
    "llama32_3b": "Llama-3.2-3B",
    "qwen25_3b": "Qwen2.5-3B",
}

# ============================================================
# DATA LOADING
# ============================================================

def load_clean_items() -> pd.DataFrame:
    # Default paper run uses the present-tense clean intersection.
    # For the was/were replication, change this filename to:
    # "clean_be_past_all_models_intersection.csv"
    path = os.path.join(BEHAV_DIR, "clean_be_present_all_models_intersection.csv")
    if not os.path.exists(path):
        path = os.path.join(DATASET_DIR, "clean_items_all_models_intersection.csv")
    df = pd.read_csv(path)
    df = df[df["verb_pair"] == VERB_PAIR].copy()
    df = df.drop_duplicates(subset=["item_id", "condition"])
    print(f"Loaded {df['item_id'].nunique()} clean base items for steering.")
    return df


def build_item_condition_map(df: pd.DataFrame) -> dict:
    item_map = {}
    for item_id, grp in df.groupby("item_id"):
        item_map[item_id] = {}
        for _, row in grp.iterrows():
            item_map[item_id][row["condition"]] = row.to_dict()
    complete = {k: v for k, v in item_map.items() if len(v) == 4}
    print(f"Items with all 4 conditions: {len(complete)}")
    return complete


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
    print(f"\n{'='*80}\nLOADING: {model_id}\n{'='*80}")
    use_cuda = safe_cuda()
    device   = torch.device("cuda" if use_cuda else "cpu")

    # Phi-2 must use float32 for numerical stability: fp16 can produce NaN
    # logits on some GPU configurations. All other models use fp16 on CUDA
    # for speed.
    if force_fp32:
        dtype = torch.float32
        print(f"  device={device}  dtype=float32  (force_fp32=True, Phi-2 safe mode)")
    else:
        dtype = torch.float16 if use_cuda else torch.float32
        print(f"  device={device}  dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kw = dict(
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=dtype,
)
    # Never use device_map with force_fp32 (Phi-2 on CPU is safer)
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
# TOKEN HELPERS
# ============================================================

def get_verb_ids(tokenizer) -> tuple:
    is_id  = tokenizer.encode(" " + SINGULAR_V, add_special_tokens=False)
    are_id = tokenizer.encode(" " + PLURAL_V,   add_special_tokens=False)
    assert len(is_id) == 1 and len(are_id) == 1, \
        f"Verb not single token: is={is_id} are={are_id}"
    return is_id[0], are_id[0]


def locate_token(tokenizer, prompt: str, word: str) -> int:
    """
    Return the position of the LAST subtoken of word in the tokenized prompt.
    Using the final subtoken ensures we read/steer the fully-formed noun
    representation rather than an incomplete prefix subtoken.
    Returns -1 if not found.
    """
    ids = tokenizer(prompt, add_special_tokens=True,
                    return_tensors="pt")["input_ids"][0].tolist()
    for prefix in (" ", ""):
        wids = tokenizer.encode(prefix + word, add_special_tokens=False)
        n = len(wids)
        for i in range(len(ids) - n + 1):
            if ids[i:i+n] == wids:
                return i + n - 1   # ← end token, not start token
    return -1


def get_token_positions(tokenizer, conds: dict) -> dict:
    """
    Returns dict: condition → {subject_pos, distractor_pos, final_pos}
    """
    pos_map = {}
    for cond, row in conds.items():
        prompt = row["prompt"]
        subj   = row["subject_singular"] if cond[0] == "S" else row["subject_plural"]
        dist   = row["distractor_singular"] if cond[1] == "S" else row["distractor_plural"]
        ids    = tokenizer(prompt, add_special_tokens=True,
                           return_tensors="pt")["input_ids"][0]
        pos_map[cond] = {
            "subject_pos":    locate_token(tokenizer, prompt, subj),
            "distractor_pos": locate_token(tokenizer, prompt, dist),
            "final_pos":      len(ids) - 1,
        }
    return pos_map


# ============================================================
# HIDDEN STATE EXTRACTION
# ============================================================

@torch.no_grad()
def get_hidden_states(model, tokenizer, device, prompt: str):
    """Returns list of [seq, hidden] float tensors, one per layer (0-indexed)."""
    enc = tokenizer(prompt, return_tensors="pt",
                    add_special_tokens=True).to(device)
    out = model(**enc, output_hidden_states=True)
    return [h.detach().float().squeeze(0) for h in out.hidden_states[1:]]


@torch.no_grad()
def get_D_score(model, tokenizer, device, prompt: str,
                is_id: int, are_id: int) -> float:
    enc    = tokenizer(prompt, return_tensors="pt",
                       add_special_tokens=True).to(device)
    out    = model(**enc)
    logits = out.logits.float()
    final  = enc["attention_mask"].sum() - 1
    return (logits[0, final, is_id] - logits[0, final, are_id]).item()


# ============================================================
# DIRECTION CONSTRUCTION
# ============================================================

def build_directions(item_map: dict, model, tokenizer, device,
                     n_layers: int, model_tag: str) -> dict:
    """
    Build per-layer subject-number and distractor-number directions.

    d_subject[layer]    = mean(subject hidden | plural-subject conditions PS,PP)
                        − mean(subject hidden | singular-subject conditions SS,SP)

    d_distractor[layer] = mean(distractor hidden | plural-distractor conditions SP,PP)
                        − mean(distractor hidden | singular-distractor conditions SS,PS)

    Returns:
        directions[direction_name][layer] = tensor [hidden_dim]
    """
    # Accumulators: direction × layer → list of vectors
    accum = {
        "subject_plural":      [[] for _ in range(n_layers)],
        "subject_singular":    [[] for _ in range(n_layers)],
        "distractor_plural":   [[] for _ in range(n_layers)],
        "distractor_singular": [[] for _ in range(n_layers)],
    }

    print(f"\n  Building directions for {model_tag}...")
    for item_id, conds in tqdm(item_map.items(),
                                desc="  direction collection", unit="item"):
        pos_map = get_token_positions(tokenizer, conds)

        for cond, row in conds.items():
            hs      = get_hidden_states(model, tokenizer, device, row["prompt"])
            p_map   = pos_map[cond]
            s_pos   = p_map["subject_pos"]
            d_pos   = p_map["distractor_pos"]

            if s_pos < 0 or d_pos < 0:
                continue

            subject_is_plural    = (cond[0] == "P")
            distractor_is_plural = (cond[1] == "P")

            for layer_idx, h in enumerate(hs):
                sv = h[s_pos].cpu()
                dv = h[d_pos].cpu()

                if subject_is_plural:
                    accum["subject_plural"][layer_idx].append(sv)
                else:
                    accum["subject_singular"][layer_idx].append(sv)

                if distractor_is_plural:
                    accum["distractor_plural"][layer_idx].append(dv)
                else:
                    accum["distractor_singular"][layer_idx].append(dv)

    # Compute directions
    directions = {"subject": {}, "distractor": {}}
    for layer_idx in range(n_layers):
        sp = torch.stack(accum["subject_plural"][layer_idx]).mean(0)
        ss = torch.stack(accum["subject_singular"][layer_idx]).mean(0)
        dp = torch.stack(accum["distractor_plural"][layer_idx]).mean(0)
        ds = torch.stack(accum["distractor_singular"][layer_idx]).mean(0)

        directions["subject"][layer_idx]    = sp - ss   # plural − singular
        directions["distractor"][layer_idx] = dp - ds

    print(f"  Directions built for {n_layers} layers.")
    return directions, accum


def build_shuffled_directions(accum: dict, n_layers: int,
                               n_shuffles: int = N_SHUFFLES,
                               seed: int = SHUFFLE_SEED) -> dict:
    """
    Build N_SHUFFLES fake directions by permuting plural/singular labels,
    then average them. One per (subject|distractor) × layer.
    """
    rng = np.random.default_rng(seed)
    shuffled = {"subject": {}, "distractor": {}}

    for layer_idx in range(n_layers):
        sp = accum["subject_plural"][layer_idx]
        ss = accum["subject_singular"][layer_idx]
        dp = accum["distractor_plural"][layer_idx]
        ds = accum["distractor_singular"][layer_idx]

        # Subject shuffled direction
        all_subj = sp + ss
        if len(all_subj) >= 2:
            shuf_dirs = []
            for _ in range(n_shuffles):
                perm = rng.permutation(len(all_subj))
                half = len(all_subj) // 2
                fake_pl = torch.stack([all_subj[i] for i in perm[:half]]).mean(0)
                fake_sg = torch.stack([all_subj[i] for i in perm[half:]]).mean(0)
                shuf_dirs.append(fake_pl - fake_sg)
            shuffled["subject"][layer_idx] = torch.stack(shuf_dirs).mean(0)
        else:
            shuffled["subject"][layer_idx] = torch.zeros_like(
                accum["subject_plural"][layer_idx][0])

        # Distractor shuffled direction
        all_dist = dp + ds
        if len(all_dist) >= 2:
            shuf_dirs = []
            for _ in range(n_shuffles):
                perm = rng.permutation(len(all_dist))
                half = len(all_dist) // 2
                fake_pl = torch.stack([all_dist[i] for i in perm[:half]]).mean(0)
                fake_sg = torch.stack([all_dist[i] for i in perm[half:]]).mean(0)
                shuf_dirs.append(fake_pl - fake_sg)
            shuffled["distractor"][layer_idx] = torch.stack(shuf_dirs).mean(0)
        else:
            shuffled["distractor"][layer_idx] = torch.zeros_like(
                accum["distractor_plural"][layer_idx][0])

    return shuffled


def build_random_directions(directions: dict, n_layers: int,
                              n_random: int = N_RANDOM_DIRS,
                              seed: int = RANDOM_DIR_SEED) -> dict:
    """
    For each real direction, build N_RANDOM random vectors with the same L2 norm,
    then average their absolute effects (norm is preserved; orientation is random).
    Returns one averaged random direction per (source × layer).
    """
    rng = np.random.default_rng(seed)
    random_dirs = {"subject": {}, "distractor": {}}

    for src in ["subject", "distractor"]:
        for layer_idx in range(n_layers):
            real_dir  = directions[src][layer_idx]
            norm      = real_dir.norm().item()
            hidden_dim = real_dir.shape[0]

            rand_vecs = []
            for _ in range(n_random):
                v = torch.tensor(
                    rng.standard_normal(hidden_dim), dtype=torch.float32)
                v = v / v.norm() * norm   # same norm, random orientation
                rand_vecs.append(v)

            # Store all random vecs; we'll average effects at steering time
            random_dirs[src][layer_idx] = rand_vecs  # list of tensors

    return random_dirs


# ============================================================
# STEERING ENGINE
# ============================================================

class SteeringHook:
    """
    Add a direction vector (scaled by alpha) to the residual stream
    at (layer, token_pos) via a forward hook.

    Handles both:
    - tuple output: (hidden_states, ...)
    - tensor output: hidden_states

    Also handles hidden shape:
    - [batch, seq, hidden]
    - [seq, hidden]
    """
    def __init__(self, model, layer_idx: int, token_pos: int,
                 direction: torch.Tensor, alpha: float):
        self.model     = model
        self.layer_idx = layer_idx
        self.token_pos = int(token_pos)
        self.direction = direction.detach().float().cpu()
        self.alpha     = float(alpha)
        self.handle    = None

    def __enter__(self):
        layer      = self.model.model.layers[self.layer_idx]
        token_pos  = self.token_pos
        direction  = self.direction
        alpha      = self.alpha

        def hook(module, input, output):
            # Case 1: HF decoder layer returns tuple(hidden_states, ...)
            if isinstance(output, tuple):
                hidden = output[0]
                new_hidden = hidden.clone()

                if new_hidden.dim() == 3:
                    # [batch, seq, hidden]
                    new_hidden[0, token_pos, :] += alpha * direction.to(
                        device=new_hidden.device,
                        dtype=new_hidden.dtype,
                    )
                elif new_hidden.dim() == 2:
                    # [seq, hidden]
                    new_hidden[token_pos, :] += alpha * direction.to(
                        device=new_hidden.device,
                        dtype=new_hidden.dtype,
                    )
                else:
                    raise RuntimeError(f"Unexpected hidden dim in tuple output: {new_hidden.shape}")

                return (new_hidden,) + output[1:]

            # Case 2: decoder layer directly returns hidden_states tensor
            hidden = output
            new_hidden = hidden.clone()

            if new_hidden.dim() == 3:
                # [batch, seq, hidden]
                new_hidden[0, token_pos, :] += alpha * direction.to(
                    device=new_hidden.device,
                    dtype=new_hidden.dtype,
                )
            elif new_hidden.dim() == 2:
                # [seq, hidden]
                new_hidden[token_pos, :] += alpha * direction.to(
                    device=new_hidden.device,
                    dtype=new_hidden.dtype,
                )
            else:
                raise RuntimeError(f"Unexpected tensor output shape: {new_hidden.shape}")

            return new_hidden

        self.handle = layer.register_forward_hook(hook)
        return self

    def __exit__(self, *args):
        if self.handle:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def steered_D_score(model, tokenizer, device,
                    prompt: str,
                    steer_layer: int, steer_pos: int,
                    direction: torch.Tensor, alpha: float,
                    is_id: int, are_id: int) -> float:
    enc = tokenizer(prompt, return_tensors="pt",
                    add_special_tokens=True).to(device)
    final_pos = enc["attention_mask"].sum().item() - 1

    with SteeringHook(model, steer_layer, steer_pos, direction, alpha):
        out    = model(**enc)
        logits = out.logits.float()

    return (logits[0, final_pos, is_id] - logits[0, final_pos, are_id]).item()


# ============================================================
# MAIN STEERING LOOP
# ============================================================

def run_steering(model, tokenizer, device, n_layers: int,
                 item_map: dict, is_id: int, are_id: int,
                 directions: dict, shuffled_dirs: dict,
                 random_dirs: dict, model_tag: str) -> pd.DataFrame:
    """
    For each item × direction × position × alpha × layer:
      measure D_steered − D_clean.

    Also runs random and shuffled controls at the same positions/layers.
    Opposite-sign check is implicit: α = +1 and α = −1 are both in ALPHAS.
    """
    rows   = []
    items  = list(item_map.items())
    layers = list(range(n_layers))

    print(f"\n{'='*80}")
    print(f"STEERING  [{model_tag}]  |  {len(items)} items  "
          f"|  {n_layers} layers  |  {len(DIRECTIONS)} directions  "
          f"|  {len(POSITIONS)} positions  |  {len(ALPHAS)} alphas")
    print(f"{'='*80}")

    for item_id, conds in tqdm(items, desc=f"[{model_tag}] items", unit="item"):

        pos_map = get_token_positions(tokenizer, conds)

        # ── Clean D scores for all 4 conditions ──────────────────────────────
        clean_D = {}
        for cond, row in conds.items():
            clean_D[cond] = get_D_score(
                model, tokenizer, device, row["prompt"], is_id, are_id)

        # ── Steering target: SP (singular subject + plural distractor) ─────────
        # SP is the agreement-attraction condition: the model sees a plural
        # distractor and may be biased toward "are" despite a singular subject.
        # Steering on SP directly tests whether we can push/pull the model
        # in this attraction case, which is the strongest test for the
        # final-token integration theory.
        # ─────────────────────────────────────────────────────────────────────

        target_cond = "SP"   # attraction condition: singular subject + plural distractor
        target_row  = conds[target_cond]
        target_pos  = pos_map[target_cond]

        base_D = clean_D[target_cond]

        for dir_name in DIRECTIONS:
            for position in POSITIONS:

                # Resolve token position
                if position == "subject":
                    tok_pos = target_pos["subject_pos"]
                elif position == "distractor":
                    tok_pos = target_pos["distractor_pos"]
                else:   # final
                    tok_pos = target_pos["final_pos"]

                if tok_pos < 0:
                    tqdm.write(f"  WARNING: {position} token not found "
                               f"item={item_id} cond={target_cond}")
                    continue

                for layer_idx in tqdm(layers, desc=f"  {dir_name}/{position}",
                                       leave=False, unit="layer"):

                    real_dir = directions[dir_name][layer_idx]

                    for alpha in ALPHAS:

                        # ── Real direction ────────────────────────────────────
                        d_steered = steered_D_score(
                            model, tokenizer, device,
                            target_row["prompt"],
                            layer_idx, tok_pos,
                            real_dir, alpha,
                            is_id, are_id,
                        )
                        real_effect = d_steered - base_D
                        # For plural push (alpha > 0): expect real_effect < 0
                        # signed_effect = −real_effect so positive = correct dir
                        signed_real = -real_effect * np.sign(alpha) if alpha != 0 else 0.0

                        # ── Shuffled-label direction ──────────────────────────
                        shuf_dir     = shuffled_dirs[dir_name][layer_idx]
                        d_shuf       = steered_D_score(
                            model, tokenizer, device,
                            target_row["prompt"],
                            layer_idx, tok_pos,
                            shuf_dir, alpha,
                            is_id, are_id,
                        )
                        shuf_effect  = d_shuf - base_D
                        signed_shuf  = -shuf_effect * np.sign(alpha) if alpha != 0 else 0.0

                        # ── Random directions (averaged) ──────────────────────
                        rand_vecs    = random_dirs[dir_name][layer_idx]
                        rand_effects = []
                        for rv in rand_vecs:
                            d_rand = steered_D_score(
                                model, tokenizer, device,
                                target_row["prompt"],
                                layer_idx, tok_pos,
                                rv, alpha,
                                is_id, are_id,
                            )
                            rand_effects.append(d_rand - base_D)
                        rand_effect_mean = float(np.mean(rand_effects))
                        signed_rand      = (-rand_effect_mean * np.sign(alpha)
                                            if alpha != 0 else 0.0)

                        rows.append({
                            "model_tag":      model_tag,
                            "item_id":        item_id,
                            "direction":      dir_name,
                            "position":       position,
                            "layer":          layer_idx,
                            "is_layer_0":     layer_idx == 0,
                            "alpha":          alpha,
                            "base_D":         base_D,
                            "d_steered_real": d_steered,
                            "d_steered_shuf": d_shuf,
                            "d_steered_rand": d_steered + rand_effect_mean - real_effect,
                            "real_effect":    real_effect,
                            "shuf_effect":    shuf_effect,
                            "rand_effect":    rand_effect_mean,
                            "signed_real":    signed_real,
                            "signed_shuf":    signed_shuf,
                            "signed_rand":    signed_rand,
                            "real_vs_shuf":   signed_real - abs(signed_shuf),
                            "real_vs_rand":   signed_real - abs(signed_rand),
                        })

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    return pd.DataFrame(rows)


# ============================================================
# BOOTSTRAP CI
# ============================================================

def bootstrap_mean_ci(values, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED, ci=95):
    rng  = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    lo    = np.percentile(boots, (100 - ci) / 2)
    hi    = np.percentile(boots, 100 - (100 - ci) / 2)
    return float(vals.mean()), float(lo), float(hi)


# ============================================================
# SUMMARISE
# ============================================================

def summarise(raw_df: pd.DataFrame, model_tag: str):
    """
    Returns:
      by_layer     — mean signed_real per (direction, position, layer, alpha)
      bootstrap_df — CI over items, layers 1+, main alpha ±1
      layer0_df    — layer 0 separate
      alpha_df     — alpha sweep at final token, layers 1+ averaged
    """

    # ── Per-layer summary ────────────────────────────────────────────────────
    by_layer_rows = []
    grp_cols = ["direction", "position", "layer", "alpha"]
    for keys, grp in raw_df.groupby(grp_cols):
        dir_name, position, layer, alpha = keys
        per_item = grp.groupby("item_id")["signed_real"].mean()
        m, lo, hi = bootstrap_mean_ci(per_item.values)
        ps = grp.groupby("item_id")["signed_shuf"].mean()
        sm, slo, shi = bootstrap_mean_ci(ps.values)
        pr = grp.groupby("item_id")["signed_rand"].mean()
        rm, rlo, rhi = bootstrap_mean_ci(pr.values)
        by_layer_rows.append({
            "model_tag": model_tag,
            "direction": dir_name, "position": position,
            "layer": layer, "is_layer_0": layer == 0, "alpha": alpha,
            "real_mean": m, "real_ci_lo": lo, "real_ci_hi": hi,
            "shuf_mean": sm, "shuf_ci_lo": slo, "shuf_ci_hi": shi,
            "rand_mean": rm, "rand_ci_lo": rlo, "rand_ci_hi": rhi,
            "n_items": grp["item_id"].nunique(),
        })
    by_layer = pd.DataFrame(by_layer_rows)

    # ── Bootstrap summary: layers 1+, main alpha ─────────────────────────────
    df_main = raw_df[(raw_df["layer"] > 0) & (raw_df["alpha"] == MAIN_ALPHA)]
    boot_rows = []
    for (dir_name, position), grp in df_main.groupby(["direction", "position"]):
        per_item  = grp.groupby("item_id")["signed_real"].mean()
        m, lo, hi = bootstrap_mean_ci(per_item.values)
        ps        = grp.groupby("item_id")["signed_shuf"].mean()
        sm, slo, shi = bootstrap_mean_ci(ps.values)
        pr        = grp.groupby("item_id")["signed_rand"].mean()
        rm, rlo, rhi = bootstrap_mean_ci(pr.values)
        peak_layer = int(
            grp.groupby("layer")["signed_real"].mean().idxmax())
        boot_rows.append({
            "model_tag": model_tag,
            "direction": dir_name, "position": position,
            "alpha": MAIN_ALPHA, "layers": "1+",
            "real_mean": m, "real_ci_lo": lo, "real_ci_hi": hi,
            "shuf_mean": sm, "shuf_ci_lo": slo, "shuf_ci_hi": shi,
            "rand_mean": rm, "rand_ci_lo": rlo, "rand_ci_hi": rhi,
            "real_vs_shuf": m - abs(sm),
            "real_vs_rand": m - abs(rm),
            "peak_layer": peak_layer,
            "n_items": grp["item_id"].nunique(),
        })
    bootstrap_df = pd.DataFrame(boot_rows)

    # ── Layer 0 separate ──────────────────────────────────────────────────────
    df_l0 = raw_df[(raw_df["layer"] == 0) & (raw_df["alpha"] == MAIN_ALPHA)]
    l0_rows = []
    for (dir_name, position), grp in df_l0.groupby(["direction", "position"]):
        per_item  = grp.groupby("item_id")["signed_real"].mean()
        m, lo, hi = bootstrap_mean_ci(per_item.values)
        l0_rows.append({
            "model_tag": model_tag, "direction": dir_name,
            "position": position, "layer": 0, "alpha": MAIN_ALPHA,
            "real_mean": m, "real_ci_lo": lo, "real_ci_hi": hi,
            "note": "Layer 0: embedding-level. Report separately.",
        })
    layer0_df = pd.DataFrame(l0_rows)

    # ── Alpha sweep: final token, layers 1+ ──────────────────────────────────
    df_alpha = raw_df[(raw_df["layer"] > 0) & (raw_df["position"] == "final")]
    alpha_rows = []
    for (dir_name, alpha), grp in df_alpha.groupby(["direction", "alpha"]):
        per_item  = grp.groupby("item_id")["real_effect"].mean()
        m, lo, hi = bootstrap_mean_ci(per_item.values)
        alpha_rows.append({
            "model_tag": model_tag, "direction": dir_name,
            "position": "final", "alpha": alpha,
            "effect_mean": m, "effect_ci_lo": lo, "effect_ci_hi": hi,
            "n_items": grp["item_id"].nunique(),
        })
    alpha_df = pd.DataFrame(alpha_rows)

    return by_layer, bootstrap_df, layer0_df, alpha_df


# ============================================================
# PLOTS
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "figure.dpi": 150,
})

POS_COLORS = {
    "subject":    "#1a6faf",
    "distractor": "#cc3311",
    "final":      "#2ca02c",
}
POS_LABELS = {
    "subject":    "Subject token",
    "distractor": "Distractor token",
    "final":      "Final token",
}
DIR_LABELS = {
    "subject":    "Subject-number direction",
    "distractor": "Distractor-number direction",
}


def plot_steering_by_layer(by_layer: pd.DataFrame, model_tag: str, n_layers: int):
    """
    One panel per direction (subject / distractor).
    Lines: subject-token / distractor-token / final-token steering.
    Main alpha only.
    """
    dirs = [d for d in DIRECTIONS if d in by_layer["direction"].unique()]
    fig, axes = plt.subplots(1, len(dirs),
                              figsize=(6.5 * len(dirs), 5), sharey=False)
    if len(dirs) == 1:
        axes = [axes]

    sub_main = by_layer[by_layer["alpha"] == MAIN_ALPHA]

    for ax, dir_name in zip(axes, dirs):
        sub = sub_main[sub_main["direction"] == dir_name]

        for pos in POSITIONS:
            pdf = sub[sub["position"] == pos].sort_values("layer")
            if pdf.empty:
                continue
            ax.plot(pdf["layer"], pdf["real_mean"],
                    color=POS_COLORS[pos], label=POS_LABELS[pos],
                    linewidth=1.8, marker="o", markersize=3)
            ax.fill_between(pdf["layer"],
                            pdf["real_ci_lo"], pdf["real_ci_hi"],
                            alpha=0.13, color=POS_COLORS[pos])

        ax.axvspan(-0.5, 0.5, color="gold", alpha=0.25, label="Layer 0 (embedding)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{DIR_LABELS[dir_name]}\nα = +{MAIN_ALPHA}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Signed effect (positive = correct direction)", fontsize=9)
        ax.legend(fontsize=9)

    fig.suptitle(f"Steering effect by layer and token position"
                 f"  [{MODEL_DISPLAY.get(model_tag, model_tag)}]",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"plot_steering_by_layer_{model_tag}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_real_vs_controls(bootstrap_df: pd.DataFrame, model_tag: str):
    """
    Bar chart: real vs shuffled vs random at main alpha, layers 1+.
    One panel per direction; x-axis = position.
    """
    dirs = [d for d in DIRECTIONS if d in bootstrap_df["direction"].unique()]
    fig, axes = plt.subplots(1, len(dirs),
                              figsize=(6.5 * len(dirs), 5), sharey=True)
    if len(dirs) == 1:
        axes = [axes]

    x      = np.arange(len(POSITIONS))
    bar_w  = 0.25

    for ax, dir_name in zip(axes, dirs):
        sub = bootstrap_df[bootstrap_df["direction"] == dir_name]
        sub = sub.set_index("position").reindex(POSITIONS)

        def safe(col, default=0.0):
            return sub[col].fillna(default).values

        real_m  = safe("real_mean")
        real_lo = real_m - safe("real_ci_lo")
        real_hi = safe("real_ci_hi") - real_m

        shuf_m  = safe("shuf_mean")
        shuf_lo = np.abs(shuf_m) - (np.abs(shuf_m) - safe("shuf_ci_lo", 0.0))
        shuf_hi = safe("shuf_ci_hi", 0.0) - np.abs(shuf_m)

        rand_m  = safe("rand_mean")

        ax.bar(x - bar_w, real_m, bar_w, label="Real direction",
               color="#1a6faf", alpha=0.85, edgecolor="white",
               yerr=[real_lo, real_hi], capsize=4,
               error_kw={"elinewidth": 1.3, "ecolor": "0.3"})
        ax.bar(x,          np.abs(shuf_m), bar_w, label="Shuffled-label direction",
               color="#ee7733", alpha=0.75, edgecolor="white",
               error_kw={"elinewidth": 1.2, "ecolor": "0.4"})
        ax.bar(x + bar_w,  np.abs(rand_m), bar_w, label="Random direction",
               color="#999999", alpha=0.70, edgecolor="white",
               error_kw={"elinewidth": 1.2, "ecolor": "0.4"})

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([POS_LABELS[p] for p in POSITIONS], fontsize=10)
        ax.set_title(f"{DIR_LABELS[dir_name]}\nα = +{MAIN_ALPHA}, layers 1+",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Signed effect (logit units)", fontsize=10)
        ax.legend(fontsize=9)

    fig.suptitle(f"Real vs shuffled vs random direction"
                 f"  [{MODEL_DISPLAY.get(model_tag, model_tag)}]",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR,
                        f"plot_steering_real_vs_controls_{model_tag}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_alpha_sweep(alpha_df: pd.DataFrame, model_tag: str):
    """
    Dose-response: x=alpha, y=D change at final token.
    Separate line per direction.
    """
    dirs = [d for d in DIRECTIONS if d in alpha_df["direction"].unique()]
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for dir_name in dirs:
        sub = alpha_df[alpha_df["direction"] == dir_name].sort_values("alpha")
        color = "#1a6faf" if dir_name == "subject" else "#cc3311"
        ax.plot(sub["alpha"], sub["effect_mean"],
                marker="o", linewidth=1.8, color=color,
                label=DIR_LABELS[dir_name])
        ax.fill_between(sub["alpha"],
                        sub["effect_ci_lo"], sub["effect_ci_hi"],
                        alpha=0.12, color=color)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Alpha (steering strength)", fontsize=11)
    ax.set_ylabel("ΔD = D_steered − D_clean (logit units)", fontsize=10)
    ax.set_title(f"Dose-response at final token, layers 1+"
                 f"  [{MODEL_DISPLAY.get(model_tag, model_tag)}]",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"plot_steering_alpha_sweep_{model_tag}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# PAPER NUMBERS
# ============================================================

def write_paper_numbers(bootstrap_df: pd.DataFrame, layer0_df: pd.DataFrame,
                         model_tag: str):
    lines = []
    tag   = MODEL_DISPLAY.get(model_tag, model_tag)
    lines.append(f"STEERING PAPER NUMBERS — {tag}")
    lines.append("=" * 72)
    lines.append(f"Alpha = +{MAIN_ALPHA}  |  Layers 1+  |  Bootstrap 95% CI over items")
    lines.append("")
    lines.append(f"{'Direction':<12} {'Position':<12} "
                 f"{'Real':>8} {'CI_lo':>8} {'CI_hi':>8} "
                 f"{'Shuf':>8} {'Rand':>8} "
                 f"{'R-S':>8} {'R-R':>8} {'Peak_L':>7}")
    lines.append("─" * 90)

    for dir_name in DIRECTIONS:
        sub = bootstrap_df[bootstrap_df["direction"] == dir_name]
        for pos in POSITIONS:
            r = sub[sub["position"] == pos]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(
                f"{dir_name:<12} {pos:<12} "
                f"{r['real_mean']:>8.4f} {r['real_ci_lo']:>8.4f} {r['real_ci_hi']:>8.4f} "
                f"{r['shuf_mean']:>8.4f} {r['rand_mean']:>8.4f} "
                f"{r['real_vs_shuf']:>8.4f} {r['real_vs_rand']:>8.4f} "
                f"{int(r['peak_layer']):>7d}"
            )
        lines.append("")

    lines.append("─" * 90)
    lines.append("Layer 0 (embedding-level, reported separately):")
    lines.append("")
    if not layer0_df.empty:
        for _, r in layer0_df.iterrows():
            lines.append(
                f"  {r['direction']:<12} {r['position']:<12} "
                f"real={r['real_mean']:.4f} "
                f"[{r['real_ci_lo']:.4f}, {r['real_ci_hi']:.4f}]"
            )

    lines.append("")
    lines.append("─" * 90)
    lines.append("Key comparison (subject-number direction, final vs distractor token):")
    idx = bootstrap_df[bootstrap_df["direction"] == "subject"].set_index("position")
    for pos in ["final", "distractor", "subject"]:
        if pos in idx.index:
            r = idx.loc[pos]
            lines.append(f"  final-token real = {r['real_mean']:.4f}" if pos == "final"
                          else f"  {pos}-token real = {r['real_mean']:.4f}")

    lines.append("")
    lines.append("Opposite-sign check (α = +1 vs α = −1, final token summary in alpha_sweep CSV):")
    lines.append("  See alpha_sweep CSV — effect(+α) should ≈ −effect(−α)")

    path = os.path.join(OUTPUT_DIR, f"steering_paper_numbers_{model_tag}.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {path}")
    print("\n".join(lines))


# ============================================================
# MAIN
# ============================================================

def main():
    torch.set_grad_enabled(False)

    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    df       = load_clean_items()
    item_map = build_item_condition_map(df)

    for model_cfg in MODELS:
        model_id   = model_cfg["model_id"]
        model_tag  = model_cfg["tag"]
        force_fp32 = model_cfg.get("force_fp32", False)

        model, tokenizer, device, n_layers = load_model(model_id, force_fp32)
        is_id, are_id = get_verb_ids(tokenizer)
        print(f"  is_id={is_id}  are_id={are_id}")

        # ── Build directions ──────────────────────────────────────────────────
        directions, accum = build_directions(
            item_map, model, tokenizer, device, n_layers, model_tag)

        # Save directions
        dir_path = os.path.join(OUTPUT_DIR,
                                f"steering_directions_{model_tag}.pkl")
        with open(dir_path, "wb") as f:
            pickle.dump({"directions": directions}, f)
        print(f"  Directions saved: {dir_path}")

        shuffled_dirs = build_shuffled_directions(accum, n_layers)
        random_dirs   = build_random_directions(directions, n_layers)

        # ── Run steering ──────────────────────────────────────────────────────
        raw_df = run_steering(
            model, tokenizer, device, n_layers,
            item_map, is_id, are_id,
            directions, shuffled_dirs, random_dirs,
            model_tag,
        )

        raw_path = os.path.join(OUTPUT_DIR, f"steering_raw_{model_tag}.csv")
        raw_df.to_csv(raw_path, index=False)
        print(f"\nSaved raw: {raw_path}  ({len(raw_df)} rows)")

        # ── Summarise ─────────────────────────────────────────────────────────
        by_layer, bootstrap_df, layer0_df, alpha_df = summarise(raw_df, model_tag)

        by_layer.to_csv(
            os.path.join(OUTPUT_DIR,
                         f"steering_summary_by_layer_{model_tag}.csv"), index=False)
        bootstrap_df.to_csv(
            os.path.join(OUTPUT_DIR,
                         f"steering_bootstrap_{model_tag}.csv"), index=False)
        layer0_df.to_csv(
            os.path.join(OUTPUT_DIR,
                         f"steering_layer0_{model_tag}.csv"), index=False)
        alpha_df.to_csv(
            os.path.join(OUTPUT_DIR,
                         f"steering_alpha_sweep_{model_tag}.csv"), index=False)

        # ── Plots ─────────────────────────────────────────────────────────────
        plot_steering_by_layer(by_layer, model_tag, n_layers)
        plot_real_vs_controls(bootstrap_df, model_tag)
        plot_alpha_sweep(alpha_df, model_tag)

        # ── Paper numbers ─────────────────────────────────────────────────────
        write_paper_numbers(bootstrap_df, layer0_df, model_tag)

        unload_model(model, tokenizer)

    print("\n" + "="*80)
    print("DONE — outputs in:", OUTPUT_DIR)
    print("="*80)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f:<65} {sz:>9,} bytes")


if __name__ == "__main__":
    main()