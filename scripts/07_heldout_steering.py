"""
robust_split_steering_phi2.py

Robustness F:
Build subject-number steering direction on 50% of items,
test steering on held-out 50% items.

Purpose:
Check whether Exp3 steering direction generalises to held-out items,
rather than being fitted to the same examples used for evaluation.

Model:
  microsoft/phi-2 only

Direction:
  d_subject[layer] = mean(subject hidden | plural subject)
                   - mean(subject hidden | singular subject)

Train/test:
  split by item_id, not by rows/conditions.

Target condition:
  SP = singular subject + plural distractor
  This is the attraction condition.

Metric:
  D = logit(is) - logit(are)
  alpha +1 should push toward plural/"are", decreasing D.
  signed_effect = -effect * sign(alpha)
  positive signed_effect = correct directional steering.

Outputs:
  robust_split_phi2_raw.csv
  robust_split_phi2_summary.csv
  robust_split_phi2_alpha_sweep.csv
  robust_split_phi2_paper_numbers.txt
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

DATASET_DIR = Path("data/agreement")
BEHAV_DIR = Path("data/agreement")

# Default example: Phi-2, is/are.
# This same script was run for all three models and both verb pairs by changing
# only the CONFIG block: MODEL_ID, MODEL_TAG, OUTPUT_DIR, VERB_PAIR,
# SINGULAR_V, and PLURAL_V.

OUTPUT_DIR = Path("results/heldout_steering/is_are")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# To reproduce each run, change only CONFIG values.
#
# is/are:
#   VERB_PAIR = "be_present"
#   SINGULAR_V = "is"
#   PLURAL_V = "are"
#   OUTPUT_DIR = Path("results/heldout_steering/is_are")
#
# was/were:
#   VERB_PAIR = "be_past"
#   SINGULAR_V = "was"
#   PLURAL_V = "were"
#   OUTPUT_DIR = Path("results/heldout_steering/was_were")
#
# Models:
#   Phi-2:
#     MODEL_ID = "microsoft/phi-2"
#     MODEL_TAG = "phi2"
#   Llama-3.2-3B:
#     MODEL_ID = "meta-llama/Llama-3.2-3B"
#     MODEL_TAG = "llama32_3b"
#   Qwen2.5-3B:
#     MODEL_ID = "Qwen/Qwen2.5-3B"
#     MODEL_TAG = "qwen25_3b"

MODEL_ID = "microsoft/phi-2"
MODEL_TAG = "phi2"

LOCAL_FILES_ONLY = True
TRUST_REMOTE_CODE = True

VERB_PAIR = "be_present"
SINGULAR_V = "is"
PLURAL_V = "are"

TARGET_COND = "SP"

ALPHAS = [-1.0, 1.0]
POSITIONS = ["subject", "distractor", "final"]

SPLIT_SEED = 123
TRAIN_FRAC = 0.5

RANDOM_SEED = 77
N_RANDOM_DIRS = 3

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42


# ============================================================
# DATA
# ============================================================

def load_clean_items() -> pd.DataFrame:
    path = BEHAV_DIR / "clean_be_present_all_models_intersection.csv"
    if not path.exists():
        path = DATASET_DIR / "clean_items_all_models_intersection.csv"

    df = pd.read_csv(path)
    df = df[df["verb_pair"] == VERB_PAIR].copy()
    df = df.drop_duplicates(subset=["item_id", "condition"])
    print(f"Loaded {df['item_id'].nunique()} clean base items.")
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


def split_items(item_map: dict):
    rng = np.random.default_rng(SPLIT_SEED)
    item_ids = np.array(sorted(item_map.keys()))
    rng.shuffle(item_ids)

    n_train = int(round(len(item_ids) * TRAIN_FRAC))
    train_ids = set(item_ids[:n_train])
    test_ids = set(item_ids[n_train:])

    train_map = {k: item_map[k] for k in train_ids}
    test_map = {k: item_map[k] for k in test_ids}

    print(f"Train items: {len(train_map)}")
    print(f"Test items:  {len(test_map)}")

    return train_map, test_map


# ============================================================
# MODEL
# ============================================================

def safe_cuda() -> bool:
    try:
        if torch.cuda.is_available():
            torch.zeros(1).cuda()
            return True
    except RuntimeError:
        return False
    return False


def load_model():
    print(f"\n{'='*80}")
    print(f"LOADING: {MODEL_ID}")
    print(f"{'='*80}")

    use_cuda = safe_cuda()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Important: Phi-2 on P100 should use float32 to avoid fp16 NaNs.
    dtype = torch.float32
    print(f"device={device}, dtype=float32")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=dtype,
    )

    model.eval()
    model.to(device)

    n_layers = len(model.model.layers)
    print(f"Layers={n_layers}")

    return model, tokenizer, device, n_layers


def unload_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model unloaded.")


# ============================================================
# TOKEN HELPERS
# ============================================================

def get_verb_ids(tokenizer):
    is_id = tokenizer.encode(" " + SINGULAR_V, add_special_tokens=False)
    are_id = tokenizer.encode(" " + PLURAL_V, add_special_tokens=False)

    assert len(is_id) == 1, f"' is' not single token: {is_id}"
    assert len(are_id) == 1, f"' are' not single token: {are_id}"

    return is_id[0], are_id[0]


def locate_token(tokenizer, prompt: str, word: str) -> int:
    """
    Return final subtoken position of word.
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


def get_token_positions(tokenizer, conds: dict):
    pos_map = {}

    for cond, row in conds.items():
        prompt = row["prompt"]

        subj = row["subject_singular"] if cond[0] == "S" else row["subject_plural"]
        dist = row["distractor_singular"] if cond[1] == "S" else row["distractor_plural"]

        ids = tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
        )["input_ids"][0]

        pos_map[cond] = {
            "subject": locate_token(tokenizer, prompt, subj),
            "distractor": locate_token(tokenizer, prompt, dist),
            "final": len(ids) - 1,
        }

    return pos_map


# ============================================================
# HIDDEN STATES AND LOGITS
# ============================================================

@torch.no_grad()
def get_hidden_states(model, tokenizer, device, prompt: str):
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(device)

    out = model(
        **enc,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )

    return [h.detach().float().squeeze(0) for h in out.hidden_states[1:]]


@torch.no_grad()
def get_D_score(model, tokenizer, device, prompt: str, is_id: int, are_id: int):
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(device)

    out = model(**enc, return_dict=True, use_cache=False)
    logits = out.logits.float()

    final_pos = enc["attention_mask"].sum().item() - 1
    return (logits[0, final_pos, is_id] - logits[0, final_pos, are_id]).item()


# ============================================================
# TRAIN-SPLIT DIRECTION
# ============================================================

def build_subject_direction_train_split(train_map, model, tokenizer, device, n_layers):
    """
    Build subject-number direction from TRAIN items only.

    d_subject[layer] = mean(subject hidden | plural subject)
                     - mean(subject hidden | singular subject)
    """
    accum = {
        "plural": [[] for _ in range(n_layers)],
        "singular": [[] for _ in range(n_layers)],
    }

    print("\nBuilding subject-number directions from TRAIN split...")

    for item_id, conds in tqdm(train_map.items(), desc="train direction", unit="item"):
        pos_map = get_token_positions(tokenizer, conds)

        for cond, row in conds.items():
            prompt = row["prompt"]
            subj_pos = pos_map[cond]["subject"]

            if subj_pos < 0:
                continue

            hs = get_hidden_states(model, tokenizer, device, prompt)
            is_plural_subject = cond[0] == "P"

            for layer_idx, h in enumerate(hs):
                vec = h[subj_pos].cpu()

                if is_plural_subject:
                    accum["plural"][layer_idx].append(vec)
                else:
                    accum["singular"][layer_idx].append(vec)

    directions = {}

    for layer_idx in range(n_layers):
        plural = torch.stack(accum["plural"][layer_idx]).mean(0)
        singular = torch.stack(accum["singular"][layer_idx]).mean(0)
        directions[layer_idx] = plural - singular

    path = OUTPUT_DIR / "robust_split_phi2_subject_directions_train.pkl"
    with open(path, "wb") as f:
        pickle.dump(
            {
                "directions": directions,
                "train_items": list(train_map.keys()),
                "split_seed": SPLIT_SEED,
                "train_frac": TRAIN_FRAC,
            },
            f,
        )

    print(f"Saved train-split directions: {path}")
    return directions


def build_random_dirs(directions, n_layers):
    rng = np.random.default_rng(RANDOM_SEED)
    random_dirs = {}

    for layer_idx in range(n_layers):
        real = directions[layer_idx]
        norm = real.norm().item()
        hidden_dim = real.shape[0]

        vecs = []
        for _ in range(N_RANDOM_DIRS):
            v = torch.tensor(rng.standard_normal(hidden_dim), dtype=torch.float32)
            v = v / v.norm() * norm
            vecs.append(v)

        random_dirs[layer_idx] = vecs

    return random_dirs


# ============================================================
# STEERING HOOK
# ============================================================

class SteeringHook:
    """
    Robust hook for Phi-2/GPT-style layer outputs.
    Handles tuple output and tensor output,
    and handles [batch, seq, hidden] or [seq, hidden].
    """

    def __init__(self, model, layer_idx: int, token_pos: int, direction: torch.Tensor, alpha: float):
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

                if new_hidden.dim() == 3:
                    new_hidden[0, token_pos, :] += alpha * direction.to(
                        device=new_hidden.device,
                        dtype=new_hidden.dtype,
                    )
                elif new_hidden.dim() == 2:
                    new_hidden[token_pos, :] += alpha * direction.to(
                        device=new_hidden.device,
                        dtype=new_hidden.dtype,
                    )
                else:
                    raise RuntimeError(f"Unexpected tuple hidden shape: {new_hidden.shape}")

                return (new_hidden,) + output[1:]

            hidden = output
            new_hidden = hidden.clone()

            if new_hidden.dim() == 3:
                new_hidden[0, token_pos, :] += alpha * direction.to(
                    device=new_hidden.device,
                    dtype=new_hidden.dtype,
                )
            elif new_hidden.dim() == 2:
                new_hidden[token_pos, :] += alpha * direction.to(
                    device=new_hidden.device,
                    dtype=new_hidden.dtype,
                )
            else:
                raise RuntimeError(f"Unexpected tensor hidden shape: {new_hidden.shape}")

            return new_hidden

        self.handle = layer.register_forward_hook(hook)
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def steered_D_score(model, tokenizer, device, prompt, layer_idx, token_pos, direction, alpha, is_id, are_id):
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(device)

    final_pos = enc["attention_mask"].sum().item() - 1

    with SteeringHook(model, layer_idx, token_pos, direction, alpha):
        out = model(**enc, return_dict=True, use_cache=False)
        logits = out.logits.float()

    return (logits[0, final_pos, is_id] - logits[0, final_pos, are_id]).item()


# ============================================================
# TEST-SPLIT STEERING
# ============================================================

def run_heldout_steering(test_map, model, tokenizer, device, n_layers, directions, random_dirs, is_id, are_id):
    rows = []

    print("\nRunning held-out steering on TEST split...")

    for item_id, conds in tqdm(test_map.items(), desc="test steering", unit="item"):
        pos_map = get_token_positions(tokenizer, conds)

        target_row = conds[TARGET_COND]
        target_prompt = target_row["prompt"]
        target_pos = pos_map[TARGET_COND]

        base_D = get_D_score(model, tokenizer, device, target_prompt, is_id, are_id)

        for position in POSITIONS:
            tok_pos = target_pos[position]

            if tok_pos < 0:
                continue

            for layer_idx in range(n_layers):
                real_dir = directions[layer_idx]

                for alpha in ALPHAS:
                    d_real = steered_D_score(
                        model, tokenizer, device,
                        target_prompt,
                        layer_idx,
                        tok_pos,
                        real_dir,
                        alpha,
                        is_id,
                        are_id,
                    )

                    real_effect = d_real - base_D
                    signed_real = -real_effect * np.sign(alpha)

                    rand_effects = []
                    for rv in random_dirs[layer_idx]:
                        d_rand = steered_D_score(
                            model, tokenizer, device,
                            target_prompt,
                            layer_idx,
                            tok_pos,
                            rv,
                            alpha,
                            is_id,
                            are_id,
                        )
                        rand_effects.append(d_rand - base_D)

                    rand_effect = float(np.mean(rand_effects))
                    signed_rand = -rand_effect * np.sign(alpha)

                    rows.append(
                        {
                            "model_tag": MODEL_TAG,
                            "item_id": item_id,
                            "split": "test",
                            "direction": "subject_train_split",
                            "position": position,
                            "layer": layer_idx,
                            "is_layer_0": layer_idx == 0,
                            "alpha": alpha,
                            "base_D": base_D,
                            "d_steered_real": d_real,
                            "real_effect": real_effect,
                            "signed_real": signed_real,
                            "rand_effect": rand_effect,
                            "signed_rand": signed_rand,
                            "real_vs_rand": signed_real - abs(signed_rand),
                        }
                    )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw = pd.DataFrame(rows)
    raw_path = OUTPUT_DIR / "robust_split_phi2_raw.csv"
    raw.to_csv(raw_path, index=False)

    print(f"Saved raw: {raw_path} ({len(raw)} rows)")
    return raw


# ============================================================
# SUMMARY
# ============================================================

def bootstrap_mean_ci(values, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED, ci=95):
    vals = np.asarray(values, dtype=float)

    if len(vals) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)

    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)

    return float(vals.mean()), float(lo), float(hi)


def summarise(raw):
    # Main summary: alpha +1, layers 1+
    main = raw[(raw["layer"] > 0) & (raw["alpha"] == 1.0)].copy()

    rows = []
    for position, grp in main.groupby("position"):
        per_item_real = grp.groupby("item_id")["signed_real"].mean()
        real_m, real_lo, real_hi = bootstrap_mean_ci(per_item_real.values)

        per_item_rand = grp.groupby("item_id")["signed_rand"].mean()
        rand_m, rand_lo, rand_hi = bootstrap_mean_ci(per_item_rand.values)

        peak_layer = int(grp.groupby("layer")["signed_real"].mean().idxmax())

        rows.append(
            {
                "model_tag": MODEL_TAG,
                "direction": "subject_train_split",
                "position": position,
                "alpha": 1.0,
                "layers": "1+",
                "real_mean": real_m,
                "real_ci_lo": real_lo,
                "real_ci_hi": real_hi,
                "rand_mean": rand_m,
                "rand_ci_lo": rand_lo,
                "rand_ci_hi": rand_hi,
                "real_vs_rand": real_m - abs(rand_m),
                "peak_layer": peak_layer,
                "n_items": grp["item_id"].nunique(),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = OUTPUT_DIR / "robust_split_phi2_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Alpha sweep at final token, layers 1+
    alpha_rows = []
    final_df = raw[(raw["layer"] > 0) & (raw["position"] == "final")].copy()

    for alpha, grp in final_df.groupby("alpha"):
        per_item_effect = grp.groupby("item_id")["real_effect"].mean()
        m, lo, hi = bootstrap_mean_ci(per_item_effect.values)

        alpha_rows.append(
            {
                "model_tag": MODEL_TAG,
                "direction": "subject_train_split",
                "position": "final",
                "alpha": alpha,
                "effect_mean": m,
                "effect_ci_lo": lo,
                "effect_ci_hi": hi,
                "n_items": grp["item_id"].nunique(),
            }
        )

    alpha_df = pd.DataFrame(alpha_rows)
    alpha_path = OUTPUT_DIR / "robust_split_phi2_alpha_sweep.csv"
    alpha_df.to_csv(alpha_path, index=False)
    print(f"Saved alpha sweep: {alpha_path}")

    return summary, alpha_df


def write_paper_numbers(summary, alpha_df):
    lines = []

    lines.append("ROBUSTNESS F — HELD-OUT STEERING — PHI-2")
    lines.append("=" * 80)
    lines.append("Direction built on 50% train items; steering tested on held-out 50%.")
    lines.append("Direction: subject-number direction only.")
    lines.append("Target condition: SP.")
    lines.append("Alpha = +1.0, layers 1+, bootstrap 95% CI over held-out items.")
    lines.append("")
    lines.append(f"{'Position':<12} {'Real':>9} {'CI_lo':>9} {'CI_hi':>9} {'Rand':>9} {'R-Rand':>9} {'Peak_L':>7} {'N':>5}")
    lines.append("-" * 80)

    order = ["subject", "distractor", "final"]
    idx = summary.set_index("position")

    for pos in order:
        if pos not in idx.index:
            continue
        r = idx.loc[pos]
        lines.append(
            f"{pos:<12} "
            f"{r['real_mean']:>9.4f} {r['real_ci_lo']:>9.4f} {r['real_ci_hi']:>9.4f} "
            f"{r['rand_mean']:>9.4f} {r['real_vs_rand']:>9.4f} "
            f"{int(r['peak_layer']):>7d} {int(r['n_items']):>5d}"
        )

    lines.append("")
    lines.append("Alpha sign check at final token:")
    lines.append("-" * 80)

    for _, r in alpha_df.sort_values("alpha").iterrows():
        lines.append(
            f"alpha={r['alpha']:+.1f}: effect={r['effect_mean']:.4f} "
            f"[{r['effect_ci_lo']:.4f}, {r['effect_ci_hi']:.4f}]"
        )

    path = OUTPUT_DIR / "robust_split_phi2_paper_numbers.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nSaved: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    torch.set_grad_enabled(False)

    df = load_clean_items()
    item_map = build_item_condition_map(df)
    train_map, test_map = split_items(item_map)

    model, tokenizer, device, n_layers = load_model()
    is_id, are_id = get_verb_ids(tokenizer)

    print(f"is_id={is_id}, are_id={are_id}")

    directions = build_subject_direction_train_split(
        train_map=train_map,
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_layers=n_layers,
    )

    random_dirs = build_random_dirs(directions, n_layers)

    raw = run_heldout_steering(
        test_map=test_map,
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_layers=n_layers,
        directions=directions,
        random_dirs=random_dirs,
        is_id=is_id,
        are_id=are_id,
    )

    summary, alpha_df = summarise(raw)
    write_paper_numbers(summary, alpha_df)

    unload_model(model, tokenizer)

    print("\nDONE.")
    print(f"Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()