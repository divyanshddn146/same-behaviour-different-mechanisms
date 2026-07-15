"""
12_distance_allpos_steering.py

All-position distance/adverbial steering profile for the agreement project.

Runs ONE model per SLURM array task:
  SLURM_ARRAY_TASK_ID=0 -> Phi-2
  SLURM_ARRAY_TASK_ID=1 -> Llama-3.2-3B
  SLURM_ARRAY_TASK_ID=2 -> Qwen2.5-3B

Default experiment:
  aux_pair: be_present (is/are)
  condition: SP only
  distances tested: medium, long, extra_long
  positions tested: ALL real prompt token positions

Main purpose:
  Check whether the distance attenuation effect can be explained by
  agreement-control steering being distributed over intermediate/filler tokens.

Metric:
  D = logit(singular auxiliary) - logit(plural auxiliary)
  subject direction = mean(plural-subject hidden) - mean(singular-subject hidden)
  +alpha should push SP prompts toward plural, so D should decrease.
  signed_real = -(D_steered_real - D_base)
  Positive signed_real = movement in expected plural direction.

Recommended first run:
  python scripts/13_distance_allpos_steering.py \
    --input_csv data/distance_adverbial_filtered/agreement_distance_adverbial_llama_clean_margin_gt0_balanced.csv \
    --output_dir results/distance_stress/all_position/is_are \
    --aux_pair be_present --singular_v is --plural_v are \
    --test_distances medium,long,extra_long \
    --condition SP \
    --max_groups_per_distance 100

Then submit via the provided SLURM array file.
"""

import argparse
import gc
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Model config
# -----------------------------

MODELS = [
    {"model_id": "microsoft/phi-2", "tag": "phi2", "force_fp32": True},
    {"model_id": "meta-llama/Llama-3.2-3B", "tag": "llama32_3b", "force_fp32": False},
    {"model_id": "Qwen/Qwen2.5-3B", "tag": "qwen25_3b", "force_fp32": False},
]

MODEL_DISPLAY = {
    "phi2": "Phi-2",
    "llama32_3b": "Llama-3.2-3B",
    "qwen25_3b": "Qwen2.5-3B",
}


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="All-position distance/adverbial steering profile."
    )
    p.add_argument(
        "--input_csv",
        type=str,
        default=(
            "data/distance_adverbial_filtered/"
            "agreement_distance_adverbial_llama_clean_margin_gt0_balanced.csv"
        ),
    )
    p.add_argument("--output_dir", type=str, default="results/distance_stress/all_position/is_are")
    p.add_argument("--direction_cache_dir", type=str, default="", help="Optional dir containing subject_directions_{model_tag}.pkl")
    p.add_argument("--aux_pair", type=str, default="be_present")
    p.add_argument("--singular_v", type=str, default="is")
    p.add_argument("--plural_v", type=str, default="are")
    p.add_argument("--condition", type=str, default="SP")
    p.add_argument(
        "--test_distances",
        type=str,
        default="medium,long,extra_long",
        help="Comma-separated distances to STEER on. Default excludes short because short has distractor/final overlap.",
    )
    p.add_argument(
        "--direction_distances",
        type=str,
        default="short,medium,long,extra_long",
        help="Comma-separated distances used to build/load directions. Use all by default for compatibility with previous steering runs.",
    )
    p.add_argument("--max_groups_per_distance", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--layer_stride", type=int, default=1, help="Use 1 for all layers; 2 for every other layer if runtime is high.")
    p.add_argument("--local_files_only", action="store_true", default=True)
    p.add_argument("--allow_download", action="store_true", help="Disable local_files_only.")
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument("--no_random", action="store_true", help="Skip random-direction control to save time.")
    p.add_argument("--no_raw", action="store_true", help="Do not write raw per-layer/per-token CSV.")
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap_seed", type=int, default=123)
    return p.parse_args()


def split_csv(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


# -----------------------------
# SLURM model selection
# -----------------------------

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


# -----------------------------
# Data loading
# -----------------------------

def load_distance_items(args):
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing dataset: {input_csv}")

    df = pd.read_csv(input_csv)
    required = [
        "aux_pair", "pair_id", "distance_bucket", "condition", "prompt",
        "subject", "subject_num", "distractor", "distractor_num",
        "correct_verb", "wrong_verb",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df = df[df["aux_pair"] == args.aux_pair].copy()
    direction_distances = set(split_csv(args.direction_distances))
    test_distances = set(split_csv(args.test_distances))
    all_needed_distances = direction_distances | test_distances
    df = df[df["distance_bucket"].isin(all_needed_distances)].copy()

    df["group_id"] = (
        df["aux_pair"].astype(str)
        + "__" + df["distance_bucket"].astype(str)
        + "__" + df["pair_id"].astype(str)
    )

    # Keep groups with all four agreement conditions.
    complete_groups = []
    for gid, grp in df.groupby("group_id"):
        if set(grp["condition"]) == {"SS", "SP", "PS", "PP"}:
            complete_groups.append(gid)
    df = df[df["group_id"].isin(complete_groups)].copy()

    # Sample equal complete groups per distance bucket.
    rng = np.random.default_rng(args.seed)
    selected_groups = []
    for dist, sub in df[["distance_bucket", "group_id"]].drop_duplicates().groupby("distance_bucket"):
        gids = sorted(sub["group_id"].tolist())
        if args.max_groups_per_distance is not None and args.max_groups_per_distance > 0:
            if len(gids) > args.max_groups_per_distance:
                gids = list(rng.choice(gids, size=args.max_groups_per_distance, replace=False))
        selected_groups.extend(gids)
    df = df[df["group_id"].isin(selected_groups)].copy()

    item_map = {}
    for gid, grp in df.groupby("group_id"):
        conds = {row["condition"]: row.to_dict() for _, row in grp.iterrows()}
        if set(conds.keys()) == {"SS", "SP", "PS", "PP"}:
            item_map[gid] = conds

    direction_item_map = {
        gid: conds for gid, conds in item_map.items()
        if conds[args.condition]["distance_bucket"] in direction_distances
    }
    test_item_map = {
        gid: conds for gid, conds in item_map.items()
        if conds[args.condition]["distance_bucket"] in test_distances
    }

    print("=" * 80)
    print("LOADED DATASET")
    print("=" * 80)
    print(f"CSV: {input_csv}")
    print(f"aux_pair: {args.aux_pair}")
    print(f"condition for steering: {args.condition}")
    print(f"direction distances: {sorted(direction_distances)}")
    print(f"test distances: {sorted(test_distances)}")
    print(f"Rows after filtering/sampling: {len(df)}")
    print("Counts:")
    print(df.groupby(["distance_bucket", "condition"]).size())
    print(f"Direction groups: {len(direction_item_map)}")
    print(f"Test groups: {len(test_item_map)}")

    return df, direction_item_map, test_item_map


# -----------------------------
# Model loading / architecture
# -----------------------------

def get_layers(model):
    # Llama/Qwen/Phi modern decoder-only style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # GPT-2 style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    # GPT-NeoX style
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Could not locate transformer layers for this architecture.")


def load_model(model_id, force_fp32, args):
    print("\n" + "=" * 80)
    print(f"LOADING MODEL: {model_id}")
    print("=" * 80)
    local_files_only = False if args.allow_download else args.local_files_only
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if force_fp32 else (torch.float16 if device.type == "cuda" else torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        local_files_only=local_files_only,
        trust_remote_code=args.trust_remote_code,
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
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    layers = get_layers(model)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Layers: {len(layers)}")
    return model, tokenizer, device, layers


def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -----------------------------
# Token helpers
# -----------------------------

def get_verb_ids(tokenizer, singular_v, plural_v):
    sg = tokenizer.encode(" " + singular_v, add_special_tokens=False)
    pl = tokenizer.encode(" " + plural_v, add_special_tokens=False)
    if len(sg) != 1 or len(pl) != 1:
        raise ValueError(
            f"Verb tokenization not single-token: {singular_v}={sg}, {plural_v}={pl}"
        )
    return sg[0], pl[0]


def locate_word_end(tokenizer, prompt, word):
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
            if ids[i:i+n] == wids:
                return i + n - 1
    return -1


def get_prompt_encoding(tokenizer, prompt):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"][0].tolist()
    attn = enc["attention_mask"][0].tolist()
    return ids, attn


def get_positions(tokenizer, row):
    prompt = row["prompt"]
    ids, attn = get_prompt_encoding(tokenizer, prompt)
    real_positions = [i for i, m in enumerate(attn) if int(m) == 1]
    final = real_positions[-1]
    return {
        "subject": locate_word_end(tokenizer, prompt, row["subject"]),
        "distractor": locate_word_end(tokenizer, prompt, row["distractor"]),
        "final": final,
        "seq_len": len(ids),
    }


def token_label(tokenizer, ids, pos, positions):
    tok_id = int(ids[pos])
    tok = tokenizer.decode([tok_id], skip_special_tokens=False)
    tok_clean = tok.replace("\n", "\\n")

    subject_pos = positions["subject"]
    distractor_pos = positions["distractor"]
    final_pos = positions["final"]

    is_special = tok_id in set(getattr(tokenizer, "all_special_ids", []))
    if pos == subject_pos:
        role = "subject"
    elif pos == distractor_pos and pos == final_pos:
        role = "distractor_final"
    elif pos == distractor_pos:
        role = "distractor"
    elif pos == final_pos:
        role = "final"
    elif distractor_pos >= 0 and distractor_pos < pos < final_pos:
        role = "filler_after_distractor"
    elif subject_pos >= 0 and subject_pos < pos < distractor_pos:
        role = "between_subject_distractor"
    elif pos < subject_pos:
        role = "prefix_before_subject"
    elif is_special:
        role = "special"
    else:
        role = "other"

    return {
        "token_pos": pos,
        "token_id": tok_id,
        "token_str": tok_clean,
        "role": role,
        "offset_from_final": pos - final_pos,
        "offset_from_distractor": pos - distractor_pos if distractor_pos >= 0 else np.nan,
        "offset_from_subject": pos - subject_pos if subject_pos >= 0 else np.nan,
        "is_subject": int(pos == subject_pos),
        "is_distractor": int(pos == distractor_pos),
        "is_final": int(pos == final_pos),
        "is_special": int(is_special),
    }


# -----------------------------
# Forward / steering helpers
# -----------------------------

@torch.no_grad()
def get_hidden_states(model, tokenizer, device, prompt):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    # Return hidden states after each transformer layer. hidden_states[0] is embeddings.
    return [h.detach().float().squeeze(0).cpu() for h in out.hidden_states[1:]]


@torch.no_grad()
def get_D_score(model, tokenizer, device, prompt, sg_id, pl_id):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    out = model(**enc, use_cache=False)
    logits = out.logits.float()
    final = int(enc["attention_mask"].sum().item() - 1)
    return float((logits[0, final, sg_id] - logits[0, final, pl_id]).item())


class SteeringHook:
    def __init__(self, layers, layer_idx, token_pos, direction, alpha):
        self.layers = layers
        self.layer_idx = int(layer_idx)
        self.token_pos = int(token_pos)
        self.direction = direction.detach().float().cpu()
        self.alpha = float(alpha)
        self.handle = None

    def __enter__(self):
        layer = self.layers[self.layer_idx]
        token_pos = self.token_pos
        direction = self.direction
        alpha = self.alpha

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                new_hidden = hidden.clone()
                if token_pos < new_hidden.shape[1]:
                    new_hidden[0, token_pos, :] += alpha * direction.to(
                        device=new_hidden.device,
                        dtype=new_hidden.dtype,
                    )
                return (new_hidden,) + output[1:]
            hidden = output
            new_hidden = hidden.clone()
            if token_pos < new_hidden.shape[1]:
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
def steered_D_score(model, tokenizer, device, layers, prompt, layer_idx, token_pos, direction, alpha, sg_id, pl_id):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    final = int(enc["attention_mask"].sum().item() - 1)
    with SteeringHook(layers, layer_idx, token_pos, direction, alpha):
        out = model(**enc, use_cache=False)
    logits = out.logits.float()
    return float((logits[0, final, sg_id] - logits[0, final, pl_id]).item())


# -----------------------------
# Direction construction/loading
# -----------------------------

def build_subject_directions(item_map, model, tokenizer, device, n_layers, model_tag, condition):
    """
    d_subject[layer] = mean(hidden at subject token | plural subject)
                       - mean(hidden at subject token | singular subject)

    Uses all four conditions from selected direction distances.
    """
    plural = [[] for _ in range(n_layers)]
    singular = [[] for _ in range(n_layers)]

    print("\n" + "=" * 80)
    print(f"BUILDING SUBJECT DIRECTIONS: {model_tag}")
    print("=" * 80)

    for gid, conds in tqdm(item_map.items(), desc="direction groups"):
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


def load_or_build_directions(args, model_tag, item_map, model, tokenizer, device, n_layers, output_dir):
    candidate_paths = []
    if args.direction_cache_dir:
        candidate_paths.append(Path(args.direction_cache_dir) / f"subject_directions_{model_tag}.pkl")
    candidate_paths.append(output_dir / f"subject_directions_{model_tag}.pkl")

    for path in candidate_paths:
        if path.exists():
            print(f"[directions] loading cached directions: {path}")
            with open(path, "rb") as f:
                obj = pickle.load(f)
            directions = obj["directions"] if isinstance(obj, dict) and "directions" in obj else obj
            return directions

    directions = build_subject_directions(
        item_map=item_map,
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_layers=n_layers,
        model_tag=model_tag,
        condition=args.condition,
    )
    out_path = output_dir / f"subject_directions_{model_tag}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"directions": directions}, f)
    print(f"[directions] saved: {out_path}")
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


# -----------------------------
# Main all-position loop
# -----------------------------

def iter_real_token_positions(tokenizer, prompt):
    ids, attn = get_prompt_encoding(tokenizer, prompt)
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    positions = []
    for i, (tok_id, mask) in enumerate(zip(ids, attn)):
        if int(mask) != 1:
            continue
        # Skip special BOS/EOS if present. These are not prompt words.
        if int(tok_id) in special_ids:
            continue
        positions.append(i)
    return ids, positions


def run_allpos_steering(test_item_map, model, tokenizer, device, layers, directions, random_dirs, sg_id, pl_id, model_tag, args):
    rows = []
    n_layers = len(layers)
    layer_indices = list(range(0, n_layers, max(1, int(args.layer_stride))))
    if (n_layers - 1) not in layer_indices:
        layer_indices.append(n_layers - 1)

    print("\n" + "=" * 80)
    print(f"RUNNING ALL-POSITION DISTANCE STEERING: {model_tag}")
    print("=" * 80)
    print(f"Test groups: {len(test_item_map)}")
    print(f"Layers tested: {layer_indices}")
    print(f"Random control: {not args.no_random}")

    for gid, conds in tqdm(test_item_map.items(), desc=f"{model_tag} allpos groups"):
        if args.condition not in conds:
            continue
        row = conds[args.condition]
        prompt = row["prompt"]
        distance_bucket = row["distance_bucket"]
        pair_id = row["pair_id"]

        pos_info = get_positions(tokenizer, row)
        ids, token_positions = iter_real_token_positions(tokenizer, prompt)
        base_D = get_D_score(model, tokenizer, device, prompt, sg_id, pl_id)

        # Warning for original short overlap; not used by default, but useful if user includes short.
        if distance_bucket == "short" and pos_info["distractor"] == pos_info["final"]:
            short_overlap = 1
        else:
            short_overlap = 0

        for tok_pos in token_positions:
            lab = token_label(tokenizer, ids, tok_pos, pos_info)
            for layer_idx in layer_indices:
                real_dir = directions[layer_idx]
                d_real = steered_D_score(
                    model, tokenizer, device, layers,
                    prompt, layer_idx, tok_pos, real_dir, args.alpha, sg_id, pl_id
                )
                real_effect = d_real - base_D
                signed_real = -real_effect

                d_rand = np.nan
                rand_effect = np.nan
                signed_rand = np.nan
                real_vs_rand = np.nan
                if not args.no_random:
                    rand_dir = random_dirs[layer_idx]
                    d_rand = steered_D_score(
                        model, tokenizer, device, layers,
                        prompt, layer_idx, tok_pos, rand_dir, args.alpha, sg_id, pl_id
                    )
                    rand_effect = d_rand - base_D
                    signed_rand = -rand_effect
                    real_vs_rand = signed_real - abs(signed_rand)

                rows.append({
                    "model_tag": model_tag,
                    "model_display": MODEL_DISPLAY.get(model_tag, model_tag),
                    "aux_pair": args.aux_pair,
                    "singular_v": args.singular_v,
                    "plural_v": args.plural_v,
                    "group_id": gid,
                    "pair_id": pair_id,
                    "distance_bucket": distance_bucket,
                    "condition": args.condition,
                    "layer": layer_idx,
                    "alpha": args.alpha,
                    "prompt": prompt,
                    "base_D": base_D,
                    "d_steered_real": d_real,
                    "d_steered_rand": d_rand,
                    "real_effect": real_effect,
                    "rand_effect": rand_effect,
                    "signed_real": signed_real,
                    "signed_rand": signed_rand,
                    "real_vs_rand": real_vs_rand,
                    "subject": row["subject"],
                    "distractor": row["distractor"],
                    "subject_num": row["subject_num"],
                    "distractor_num": row["distractor_num"],
                    "subject_pos": pos_info["subject"],
                    "distractor_pos": pos_info["distractor"],
                    "final_pos": pos_info["final"],
                    "short_distractor_final_overlap": short_overlap,
                    **lab,
                })

    raw = pd.DataFrame(rows)
    return raw


# -----------------------------
# Summaries
# -----------------------------

def bootstrap_ci(vals, n_bootstrap=1000, seed=123):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_bootstrap, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarize_allpos(raw, args, model_tag):
    # Ignore layer 0 for paper-style summaries, matching earlier scripts.
    main = raw[raw["layer"] > 0].copy()

    # Item-level mean first so prompts with more layers do not dominate differently.
    token_rows = []
    group_cols = [
        "distance_bucket", "role", "offset_from_final", "offset_from_distractor",
        "token_pos", "token_str",
    ]
    for keys, grp in main.groupby(group_cols, dropna=False):
        per_item = grp.groupby("group_id")["signed_real"].mean()
        m, lo, hi = bootstrap_ci(per_item.values, args.n_bootstrap, args.bootstrap_seed)
        per_item_r = grp.groupby("group_id")["signed_rand"].mean() if "signed_rand" in grp else pd.Series(dtype=float)
        rm, rlo, rhi = bootstrap_ci(per_item_r.values, args.n_bootstrap, args.bootstrap_seed)
        layer_means = grp.groupby("layer")["signed_real"].mean()
        peak_layer = int(layer_means.idxmax()) if not layer_means.empty else -1
        rec = dict(zip(group_cols, keys))
        rec.update({
            "model_tag": model_tag,
            "aux_pair": args.aux_pair,
            "layers": "1+",
            "real_mean": m,
            "real_ci_lo": lo,
            "real_ci_hi": hi,
            "rand_mean": rm,
            "rand_ci_lo": rlo,
            "rand_ci_hi": rhi,
            "real_vs_rand": m - abs(rm) if np.isfinite(rm) else np.nan,
            "peak_layer": peak_layer,
            "n_items": int(per_item.shape[0]),
            "n_rows": int(grp.shape[0]),
        })
        token_rows.append(rec)
    token_summary = pd.DataFrame(token_rows)

    # Coarse role summary: average within item/role/layer first so filler with many tokens does not dominate.
    role_rows = []
    if not main.empty:
        item_role = (
            main.groupby(["distance_bucket", "role", "group_id", "layer"], as_index=False)
            .agg(signed_real=("signed_real", "mean"), signed_rand=("signed_rand", "mean"))
        )
        for (dist, role), grp in item_role.groupby(["distance_bucket", "role"]):
            per_item = grp.groupby("group_id")["signed_real"].mean()
            m, lo, hi = bootstrap_ci(per_item.values, args.n_bootstrap, args.bootstrap_seed)
            per_item_r = grp.groupby("group_id")["signed_rand"].mean()
            rm, rlo, rhi = bootstrap_ci(per_item_r.values, args.n_bootstrap, args.bootstrap_seed)
            layer_means = grp.groupby("layer")["signed_real"].mean()
            peak_layer = int(layer_means.idxmax()) if not layer_means.empty else -1
            role_rows.append({
                "model_tag": model_tag,
                "aux_pair": args.aux_pair,
                "distance_bucket": dist,
                "role": role,
                "layers": "1+",
                "real_mean": m,
                "real_ci_lo": lo,
                "real_ci_hi": hi,
                "rand_mean": rm,
                "rand_ci_lo": rlo,
                "rand_ci_hi": rhi,
                "real_vs_rand": m - abs(rm) if np.isfinite(rm) else np.nan,
                "peak_layer": peak_layer,
                "n_items": int(per_item.shape[0]),
                "n_item_layer_rows": int(grp.shape[0]),
            })
    role_summary = pd.DataFrame(role_rows)

    # Per-layer role summary for heatmaps.
    layer_role = pd.DataFrame()
    if not main.empty:
        layer_role = (
            main.groupby(["model_tag", "aux_pair", "distance_bucket", "role", "layer"], as_index=False)
            .agg(real_mean=("signed_real", "mean"), rand_mean=("signed_rand", "mean"), n=("group_id", "nunique"))
        )

    return token_summary, role_summary, layer_role


def write_report(output_dir, model_tag, raw, token_summary, role_summary, args):
    lines = []
    lines.append(f"# All-position distance steering report — {MODEL_DISPLAY.get(model_tag, model_tag)}")
    lines.append("")
    lines.append(f"Aux pair: `{args.aux_pair}` ({args.singular_v}/{args.plural_v})")
    lines.append(f"Condition: `{args.condition}`")
    lines.append(f"Test distances: `{args.test_distances}`")
    lines.append(f"Rows: {len(raw):,}")
    lines.append("")
    lines.append("## Coarse role summary, layers 1+")
    lines.append("")
    if role_summary.empty:
        lines.append("No role summary produced.")
    else:
        show = role_summary.sort_values(["distance_bucket", "real_mean"], ascending=[True, False])
        cols = ["distance_bucket", "role", "real_mean", "real_ci_lo", "real_ci_hi", "rand_mean", "peak_layer", "n_items"]
        lines.append(show[cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `signed_real > 0` means steering moved the model toward the expected plural direction on SP prompts.")
    lines.append("- Default run excludes the original short distance from testing because short has distractor/final overlap.")
    lines.append("- `filler_after_distractor` summarizes tokens after the distractor and before the final token.")
    lines.append("- Use token-level summary to inspect which specific offsets/tokens carry steering effect.")

    path = output_dir / f"allpos_distance_report_{model_tag}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report: {path}")


# -----------------------------
# Main
# -----------------------------

def main():
    torch.set_grad_enabled(False)
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_model_cfg()
    model_id = cfg["model_id"]
    model_tag = cfg["tag"]
    force_fp32 = cfg.get("force_fp32", False)

    _, direction_item_map, test_item_map = load_distance_items(args)
    model, tokenizer, device, layers = load_model(model_id, force_fp32, args)
    n_layers = len(layers)

    sg_id, pl_id = get_verb_ids(tokenizer, args.singular_v, args.plural_v)
    print(f"Token IDs: {args.singular_v}={sg_id}, {args.plural_v}={pl_id}")

    directions = load_or_build_directions(
        args=args,
        model_tag=model_tag,
        item_map=direction_item_map,
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_layers=n_layers,
        output_dir=output_dir,
    )
    random_dirs = None if args.no_random else build_random_directions(directions)

    raw = run_allpos_steering(
        test_item_map=test_item_map,
        model=model,
        tokenizer=tokenizer,
        device=device,
        layers=layers,
        directions=directions,
        random_dirs=random_dirs,
        sg_id=sg_id,
        pl_id=pl_id,
        model_tag=model_tag,
        args=args,
    )

    if not args.no_raw:
        raw_path = output_dir / f"allpos_distance_raw_{model_tag}.csv"
        raw.to_csv(raw_path, index=False)
        print(f"Saved raw: {raw_path} rows={len(raw)}")

    token_summary, role_summary, layer_role = summarize_allpos(raw, args, model_tag)

    token_summary_path = output_dir / f"allpos_distance_token_summary_{model_tag}.csv"
    role_summary_path = output_dir / f"allpos_distance_role_summary_{model_tag}.csv"
    layer_role_path = output_dir / f"allpos_distance_layer_role_summary_{model_tag}.csv"

    token_summary.to_csv(token_summary_path, index=False)
    role_summary.to_csv(role_summary_path, index=False)
    layer_role.to_csv(layer_role_path, index=False)

    print(f"Saved token summary: {token_summary_path}")
    print(f"Saved role summary: {role_summary_path}")
    print(f"Saved layer-role summary: {layer_role_path}")

    write_report(output_dir, model_tag, raw, token_summary, role_summary, args)

    unload_model(model, tokenizer)
    print("\nDONE")


if __name__ == "__main__":
    main()
