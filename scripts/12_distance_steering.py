"""
steering_distance_adverbial_array.py

Distance steering experiment for the validated adverbial distance dataset.

Runs ONE model per SLURM array task:
  0 = Phi-2
  1 = Llama-3.2-3B
  2 = Qwen2.5-3B

Dataset:
  data/distance_adverbial_filtered/
    agreement_distance_adverbial_llama_clean_margin_gt0_balanced.csv

Core measurement:
  Build subject-number direction = plural-subject hidden - singular-subject hidden.
  Test steering on SP condition across distance buckets:
    short / medium / long / extra_long

Positions:
  subject token
  final token

Metric:
  D = logit(is) - logit(are)
  alpha=+1 subject direction should push plural "are", so D should decrease.
  signed_effect = -(D_steered - D_clean)
  Positive signed_effect = steering works in expected plural direction.
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

INPUT_CSV = Path(
    "data/distance_adverbial_filtered/"
    "agreement_distance_adverbial_llama_clean_margin_gt0_balanced.csv"
)

# Default: is/are distance steering.
OUT_DIR = Path("results/distance_stress/steering/is_are")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {"model_id": "microsoft/phi-2", "tag": "phi2", "force_fp32": True},
    {"model_id": "meta-llama/Llama-3.2-3B", "tag": "llama32_3b", "force_fp32": False},
    {"model_id": "Qwen/Qwen2.5-3B", "tag": "qwen25_3b", "force_fp32": False},
]

LOCAL_FILES_ONLY = True
TRUST_REMOTE_CODE = True

# Default reported run: is/are distance steering.
# For was/were, change only:
#   OUT_DIR = Path("results/distance_stress/steering/was_were")
#   AUX_PAIR = "be_past"
#   SINGULAR_V = "was"
#   PLURAL_V = "were"
AUX_PAIR = "be_present"
SINGULAR_V = "is"
PLURAL_V = "are"

ALPHA = 1.0
POSITIONS = ["subject", "final"]

# Keep runtime sane. This samples complete pair_id groups per distance bucket.
# 100 means about 100 SP prompts per distance = 400 total target prompts per model.
MAX_GROUPS_PER_DISTANCE = 100
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
# DATA
# ============================================================

def load_distance_items():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing dataset: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    df = df[df["aux_pair"] == AUX_PAIR].copy()

    required = [
        "pair_id", "distance_bucket", "condition", "prompt",
        "subject", "subject_num", "distractor", "distractor_num",
        "correct_verb", "wrong_verb",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    print("=" * 80)
    print("LOADED DATASET")
    print("=" * 80)
    print(f"CSV: {INPUT_CSV}")
    print(f"Rows after aux_pair={AUX_PAIR}: {len(df)}")
    print("\nCounts:")
    print(df.groupby(["distance_bucket", "condition"]).size())

    # Create a group id: one pair_id at one distance bucket should have SS/SP/PS/PP.
    df["group_id"] = (
        df["aux_pair"].astype(str)
        + "__" + df["distance_bucket"].astype(str)
        + "__" + df["pair_id"].astype(str)
    )

    # Keep only complete 4-condition groups.
    complete_groups = []
    for gid, grp in df.groupby("group_id"):
        if set(grp["condition"]) == {"SS", "SP", "PS", "PP"}:
            complete_groups.append(gid)

    df = df[df["group_id"].isin(complete_groups)].copy()

    # Sample equal number of complete groups per distance bucket.
    rng = np.random.default_rng(SEED)
    selected_groups = []

    for dist, sub in df[["distance_bucket", "group_id"]].drop_duplicates().groupby("distance_bucket"):
        gids = sub["group_id"].tolist()
        gids = sorted(gids)
        if MAX_GROUPS_PER_DISTANCE is not None and len(gids) > MAX_GROUPS_PER_DISTANCE:
            gids = list(rng.choice(gids, size=MAX_GROUPS_PER_DISTANCE, replace=False))
        selected_groups.extend(gids)

    df = df[df["group_id"].isin(selected_groups)].copy()

    print("\nAfter complete-group filtering/sampling:")
    print(f"Rows: {len(df)}")
    print(f"Groups: {df['group_id'].nunique()}")
    print(df.groupby(["distance_bucket", "condition"]).size())

    item_map = {}
    for gid, grp in df.groupby("group_id"):
        conds = {}
        for _, row in grp.iterrows():
            conds[row["condition"]] = row.to_dict()
        if set(conds.keys()) == {"SS", "SP", "PS", "PP"}:
            item_map[gid] = conds

    print(f"\nComplete item groups for steering: {len(item_map)}")
    return df, item_map


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
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **load_kwargs,
        )

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
    """
    Locate final subtoken position of word in prompt.
    Returns -1 if not found.
    """
    ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]

    candidates = [
        tokenizer.encode(" " + word, add_special_tokens=False),
        tokenizer.encode(word, add_special_tokens=False),
    ]

    for wids in candidates:
        n = len(wids)
        if n == 0:
            continue
        for i in range(len(ids) - n + 1):
            if ids[i:i+n] == wids:
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

def build_subject_directions(item_map, model, tokenizer, device, n_layers, model_tag):
    """
    d_subject[layer] = mean(hidden at subject token | plural subject)
                       - mean(hidden at subject token | singular subject)
    Uses all distance buckets together.
    """
    plural = [[] for _ in range(n_layers)]
    singular = [[] for _ in range(n_layers)]

    print("\n" + "=" * 80)
    print(f"BUILDING SUBJECT DIRECTIONS: {model_tag}")
    print("=" * 80)

    for gid, conds in tqdm(item_map.items(), desc="direction items"):
        for cond, row in conds.items():
            pos = get_positions(tokenizer, row)
            s_pos = pos["subject"]

            if s_pos < 0:
                tqdm.write(f"[WARN] subject not found: gid={gid} cond={cond} subject={row['subject']}")
                continue

            hs = get_hidden_states(model, tokenizer, device, row["prompt"])
            is_plural = row["subject_num"] == "pl"

            for layer_idx, h in enumerate(hs):
                vec = h[s_pos].cpu()
                if is_plural:
                    plural[layer_idx].append(vec)
                else:
                    singular[layer_idx].append(vec)

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

def run_distance_steering(item_map, model, tokenizer, device, n_layers, directions, random_dirs, sg_id, pl_id, model_tag):
    rows = []

    layers = list(range(n_layers))

    print("\n" + "=" * 80)
    print(f"RUNNING DISTANCE STEERING: {model_tag}")
    print("=" * 80)
    print(f"Items: {len(item_map)} | layers: {n_layers} | positions: {POSITIONS}")

    for gid, conds in tqdm(item_map.items(), desc=f"{model_tag} groups"):
        # We test SP, the attraction condition.
        if "SP" not in conds:
            continue

        row = conds["SP"]
        prompt = row["prompt"]
        distance_bucket = row["distance_bucket"]

        pos = get_positions(tokenizer, row)
        base_D = get_D_score(model, tokenizer, device, prompt, sg_id, pl_id)

        for position in POSITIONS:
            tok_pos = pos[position]
            if tok_pos < 0:
                tqdm.write(f"[WARN] position not found: gid={gid} pos={position}")
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

                # Direction is plural - singular. On SP, +alpha should push plural "are",
                # so D=is-are should decrease. Positive signed effect means expected direction.
                signed_real = -real_effect
                signed_rand = -rand_effect

                rows.append({
                    "model_tag": model_tag,
                    "group_id": gid,
                    "pair_id": row["pair_id"],
                    "distance_bucket": distance_bucket,
                    "condition": "SP",
                    "position": position,
                    "layer": layer_idx,
                    "is_layer_0": layer_idx == 0,
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

    # layers 1+ summary per distance and position
    main = raw[raw["layer"] > 0].copy()

    for (dist, pos), grp in main.groupby(["distance_bucket", "position"]):
        per_item = grp.groupby("group_id")["signed_real"].mean()
        m, lo, hi = bootstrap_ci(per_item.values)

        per_item_r = grp.groupby("group_id")["signed_rand"].mean()
        rm, rlo, rhi = bootstrap_ci(per_item_r.values)

        layer_means = grp.groupby("layer")["signed_real"].mean()
        peak_layer = int(layer_means.idxmax())

        rows.append({
            "model_tag": model_tag,
            "distance_bucket": dist,
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

    # subject/final ratio
    ratio_rows = []
    for dist, grp in summary.groupby("distance_bucket"):
        idx = grp.set_index("position")
        if "subject" in idx.index and "final" in idx.index:
            subj = float(idx.loc["subject", "real_mean"])
            final = float(idx.loc["final", "real_mean"])
            ratio = subj / final if abs(final) > 1e-8 else np.nan
            ratio_rows.append({
                "model_tag": model_tag,
                "distance_bucket": dist,
                "subject_effect": subj,
                "final_effect": final,
                "subject_final_ratio": ratio,
                "pattern": "source-localized" if ratio > 1 else "final-integrated",
            })

    ratios = pd.DataFrame(ratio_rows)

    return summary, ratios


def write_paper_numbers(summary, ratios, model_tag):
    lines = []
    lines.append(f"DISTANCE STEERING NUMBERS — {MODEL_DISPLAY.get(model_tag, model_tag)}")
    lines.append("=" * 80)
    lines.append("Subject-number direction | Target: SP | alpha=+1 | layers 1+")
    lines.append("")
    lines.append("Effects by distance:")
    lines.append("")
    lines.append(f"{'Distance':<12} {'Position':<10} {'Effect':>10} {'CI_lo':>10} {'CI_hi':>10} {'Rand':>10} {'PeakL':>6}")
    lines.append("-" * 80)

    order = ["short", "medium", "long", "extra_long"]
    for dist in order:
        sub = summary[summary["distance_bucket"] == dist]
        for pos in POSITIONS:
            r = sub[sub["position"] == pos]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(
                f"{dist:<12} {pos:<10} "
                f"{r['real_mean']:>10.4f} {r['real_ci_lo']:>10.4f} {r['real_ci_hi']:>10.4f} "
                f"{r['rand_mean']:>10.4f} {int(r['peak_layer']):>6d}"
            )

    lines.append("")
    lines.append("Subject/final ratios:")
    lines.append("")
    lines.append(f"{'Distance':<12} {'Subject':>10} {'Final':>10} {'Ratio':>10} {'Pattern':>18}")
    lines.append("-" * 80)

    for dist in order:
        r = ratios[ratios["distance_bucket"] == dist]
        if r.empty:
            continue
        r = r.iloc[0]
        lines.append(
            f"{dist:<12} {r['subject_effect']:>10.4f} {r['final_effect']:>10.4f} "
            f"{r['subject_final_ratio']:>10.3f} {r['pattern']:>18}"
        )

    path = OUT_DIR / f"distance_steering_paper_numbers_{model_tag}.txt"
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

    df, item_map = load_distance_items()

    model, tokenizer, device, n_layers = load_model(model_id, force_fp32)

    sg_id, pl_id = get_verb_ids(tokenizer)
    print(f"Token IDs: {SINGULAR_V}={sg_id}, {PLURAL_V}={pl_id}")

    directions = build_subject_directions(
        item_map=item_map,
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_layers=n_layers,
        model_tag=model_tag,
    )

    dir_path = OUT_DIR / f"subject_directions_{model_tag}.pkl"
    with open(dir_path, "wb") as f:
        pickle.dump({"directions": directions}, f)
    print(f"Saved directions: {dir_path}")

    random_dirs = build_random_directions(directions)

    raw = run_distance_steering(
        item_map=item_map,
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

    raw_path = OUT_DIR / f"distance_steering_raw_{model_tag}.csv"
    raw.to_csv(raw_path, index=False)
    print(f"Saved raw: {raw_path} rows={len(raw)}")

    summary, ratios = summarize(raw, model_tag)

    summary_path = OUT_DIR / f"distance_steering_summary_{model_tag}.csv"
    ratios_path = OUT_DIR / f"distance_steering_ratios_{model_tag}.csv"

    summary.to_csv(summary_path, index=False)
    ratios.to_csv(ratios_path, index=False)

    print(f"Saved summary: {summary_path}")
    print(f"Saved ratios: {ratios_path}")

    write_paper_numbers(summary, ratios, model_tag)

    unload_model(model, tokenizer)

    print("\nDONE")


if __name__ == "__main__":
    main()