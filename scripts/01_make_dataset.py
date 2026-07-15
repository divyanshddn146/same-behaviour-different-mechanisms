"""
make_dataset.py
===============
Agreement-attraction dataset builder for BlackboxNLP paper.

Template : The {subject} of the {distractor} in question ___
Conditions: SS, SP, PS, PP
Verb pairs: is/are, was/were, has/have, does/do
Models    : microsoft/phi-2
            meta-llama/Llama-3.2-3B
            Qwen/Qwen2.5-3B

Outputs
-------
agreement_items_raw.csv
agreement_items_manual_clean.csv
agreement_audit_{model_tag}.csv          (one per model)
clean_items_{model_tag}_margin_gt_0.csv  (one per model)
clean_items_{model_tag}_margin_gt_1.csv  (one per model)
clean_items_all_models_intersection.csv
dataset_summary.csv

Per-sentence pass/fail is printed to stdout and saved in the audit CSVs.
"""

import os
import gc
import random
import itertools
from collections import defaultdict

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "agreement_dataset_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOCAL_FILES_ONLY = True   # set False to download from HuggingFace Hub
TRUST_REMOTE_CODE = True

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

VERB_PAIRS = [
    {"pair_id": "be_present",  "singular": "is",  "plural": "are"},
    {"pair_id": "be_past",     "singular": "was",  "plural": "were"},
    {"pair_id": "have_present","singular": "has",  "plural": "have"},
    {"pair_id": "do_present",  "singular": "does", "plural": "do"},
]

CONDITIONS = ["SS", "SP", "PS", "PP"]

MARGIN_CLEAN    = 0.0   # margin > 0   → clean
MARGIN_STRONG   = 1.0   # margin > 1   → strong

RANDOM_SEED     = 42    # for reproducible shuffled sampling
MAX_BASE_ITEMS  = 400   # cap on base items after shuffle

# ============================================================
# NOUN POOL
# ============================================================

SUBJECT_PAIRS = [
    ("executive",       "executives"),
    ("manager",         "managers"),
    ("director",        "directors"),
    ("teacher",         "teachers"),
    ("student",         "students"),
    ("scientist",       "scientists"),
    ("engineer",        "engineers"),
    ("artist",          "artists"),
    ("doctor",          "doctors"),
    ("lawyer",          "lawyers"),
    ("author",          "authors"),
    ("researcher",      "researchers"),
    ("employee",        "employees"),
    ("officer",         "officers"),
    ("worker",          "workers"),
    ("candidate",       "candidates"),
    ("member",          "members"),
    ("leader",          "leaders"),
    ("representative",  "representatives"),
    ("consultant",      "consultants"),
    ("analyst",         "analysts"),
    ("editor",          "editors"),
    ("inspector",       "inspectors"),
    ("coordinator",     "coordinators"),
    ("supervisor",      "supervisors"),
    ("administrator",   "administrators"),
    ("assistant",       "assistants"),
    ("accountant",      "accountants"),
    ("architect",       "architects"),
    ("designer",        "designers"),
    ("planner",         "planners"),
    ("advisor",         "advisors"),
    ("commissioner",    "commissioners"),
    ("auditor",         "auditors"),
    ("examiner",        "examiners"),
    ("evaluator",       "evaluators"),
    ("instructor",      "instructors"),
    ("operator",        "operators"),
    ("technician",      "technicians"),
    ("developer",       "developers"),
    ("programmer",      "programmers"),
    ("reporter",        "reporters"),
    ("journalist",      "journalists"),
    ("professor",       "professors"),
    ("lecturer",        "lecturers"),
    ("trainer",         "trainers"),
    ("builder",         "builders"),
    ("driver",          "drivers"),
    ("painter",         "painters"),
    ("owner",           "owners"),
]

DISTRACTOR_PAIRS = [
    ("corporation",     "corporations"),
    ("company",         "companies"),
    ("department",      "departments"),
    ("organization",    "organizations"),
    ("agency",          "agencies"),
    ("university",      "universities"),
    ("school",          "schools"),
    ("laboratory",      "laboratories"),
    ("institution",     "institutions"),
    ("office",          "offices"),
    ("project",         "projects"),
    ("program",         "programs"),
    ("division",        "divisions"),
    ("association",     "associations"),
    ("foundation",      "foundations"),
    ("club",            "clubs"),
    ("board",           "boards"),
    ("hospital",        "hospitals"),
    ("institute",       "institutes"),
    ("committee",       "committees"),
    ("ministry",        "ministries"),
    ("bureau",          "bureaus"),
    ("branch",          "branches"),
    ("network",         "networks"),
    ("sector",          "sectors"),
    ("panel",           "panels"),
    ("council",         "councils"),
    ("forum",           "forums"),
    ("coalition",       "coalitions"),
    ("alliance",        "alliances"),
    ("consortium",      "consortiums"),
    ("federation",      "federations"),
    ("partnership",     "partnerships"),
    ("enterprise",      "enterprises"),
    ("firm",            "firms"),
    ("authority",       "authorities"),
    ("service",         "services"),
    ("system",          "systems"),
    ("facility",        "facilities"),
    ("unit",            "units"),
    ("group",           "groups"),
    ("team",            "teams"),
    ("station",         "stations"),
    ("centre",          "centres"),
    ("region",          "regions"),
    ("district",        "districts"),
    ("platform",        "platforms"),
    ("college",         "colleges"),
]
# Duplicates removed: ("sector","sectors") appeared twice, ("institute","institutes") appeared twice

# Verb continuations for full_sentence_example metadata (correct_verb fills the gap)
CONTINUATIONS = {
    "be_present":   "responsible for the decision.",
    "be_past":      "responsible for the decision.",
    "have_present": "received the report.",
    "do_present":   "not agree with the proposal.",
}

# ============================================================
# PROMPT BUILDER
# ============================================================

def make_prompt(subject: str, distractor: str) -> str:
    return f"The {subject} of the {distractor} in question"


def subject_from_condition(condition: str, subj_sg: str, subj_pl: str) -> str:
    return subj_sg if condition[0] == "S" else subj_pl


def distractor_from_condition(condition: str, dist_sg: str, dist_pl: str) -> str:
    return dist_sg if condition[1] == "S" else dist_pl


def correct_verb(condition: str, singular_verb: str, plural_verb: str) -> str:
    return singular_verb if condition[0] == "S" else plural_verb


def wrong_verb(condition: str, singular_verb: str, plural_verb: str) -> str:
    return plural_verb if condition[0] == "S" else singular_verb


# ============================================================
# DATASET GENERATION
# ============================================================
def get_dtype(dtype_name: str):
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype for P100: {dtype_name}")

def build_raw_dataset() -> pd.DataFrame:
    """
    Cross subject pairs × distractor pairs, deduplicate, randomly shuffle
    (seeded for reproducibility), cap at MAX_BASE_ITEMS, assign item_ids.

    Random shuffle ensures even subject coverage across all 400 items,
    avoiding the [:400] bias that would over-sample early subjects.
    """
    rows = []
    item_id = 0

    # Cross all subjects with all distractors
    pairs = list(itertools.product(SUBJECT_PAIRS, DISTRACTOR_PAIRS))

    # Deduplicate on (subj_sg, dist_sg) key — removes any accidental duplicates
    # that survived the distractor-list cleanup
    seen = set()
    unique_pairs = []
    for (subj_sg, subj_pl), (dist_sg, dist_pl) in pairs:
        key = (subj_sg, dist_sg)
        if key not in seen:
            seen.add(key)
            unique_pairs.append(((subj_sg, subj_pl), (dist_sg, dist_pl)))

    # ── FIX: seeded shuffle for diverse subject coverage ──────────────────────
    # [:400] without shuffle would assign ~8 distractors to each of the first
    # 50 subjects and nothing to later ones. Shuffle first.
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(unique_pairs)
    unique_pairs = unique_pairs[:MAX_BASE_ITEMS]

    n_subjects_covered = len({sg for (sg, _), _ in unique_pairs})
    n_distractors_covered = len({dsg for _, (dsg, _) in unique_pairs})
    print(f"After shuffle + cap: {len(unique_pairs)} base items | "
          f"{n_subjects_covered} unique subjects | "
          f"{n_distractors_covered} unique distractors")

    for (subj_sg, subj_pl), (dist_sg, dist_pl) in unique_pairs:
        item_tag = f"agr_{item_id:04d}"

        for vp in VERB_PAIRS:
            for cond in CONDITIONS:
                subj = subject_from_condition(cond, subj_sg, subj_pl)
                dist = distractor_from_condition(cond, dist_sg, dist_pl)
                prompt = make_prompt(subj, dist)
                cv = correct_verb(cond, vp["singular"], vp["plural"])
                wv = wrong_verb(cond,   vp["singular"], vp["plural"])

                # ── FIX: correct full_sentence_example ────────────────────────
                # Format: "{prompt} {correct_verb} {continuation}"
                # e.g. "The executive of the corporations in question is responsible..."
                continuation = CONTINUATIONS[vp["pair_id"]]
                full_sent = f"{prompt} {cv} {continuation}"

                rows.append({
                    "item_id":               item_tag,
                    "base_item_index":       item_id,
                    "template_id":           "of_in_question",
                    "verb_pair":             vp["pair_id"],
                    "subject_singular":      subj_sg,
                    "subject_plural":        subj_pl,
                    "distractor_singular":   dist_sg,
                    "distractor_plural":     dist_pl,
                    "condition":             cond,
                    "subject_number":        "singular" if cond[0] == "S" else "plural",
                    "distractor_number":     "singular" if cond[1] == "S" else "plural",
                    "prompt":                prompt,
                    "singular_verb":         vp["singular"],
                    "plural_verb":           vp["plural"],
                    "correct_verb":          cv,
                    "wrong_verb":            wv,
                    "full_sentence_example": full_sent,
                })

        item_id += 1

    df = pd.DataFrame(rows)
    print(f"Raw dataset: {item_id} base items × {len(VERB_PAIRS)} verb pairs × "
          f"{len(CONDITIONS)} conditions = {len(df)} rows")
    return df


def build_manual_clean(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove base items where subject and distractor singular forms are identical
    (e.g. painter/house appears twice due to dedup gap), or any obvious bad pair.
    """
    # Flag items where subject_singular == distractor_singular
    bad_items = raw_df[
        raw_df["subject_singular"] == raw_df["distractor_singular"]
    ]["item_id"].unique()

    clean_df = raw_df[~raw_df["item_id"].isin(bad_items)].copy()
    n_removed = len(raw_df["item_id"].unique()) - len(clean_df["item_id"].unique())
    print(f"Manual clean: removed {n_removed} base items with identical subject/distractor. "
          f"Remaining: {len(clean_df['item_id'].unique())} base items, {len(clean_df)} rows.")
    return clean_df


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_id: str, dtype_name: str = "float16"):
    print(f"\n{'='*80}")
    print(f"LOADING: {model_id}")
    print(f"{'='*80}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = get_dtype(dtype_name) if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── FIX: use device_map="auto" on GPU for safe multi-model loading ────────
    load_kwargs = dict(
        local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=dtype,
    )
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="eager", **load_kwargs
        )
    except (TypeError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    model.eval()
    # Only call .to(device) when not using device_map (device_map handles placement)
    if not torch.cuda.is_available():
        model.to(device)

    n_layers = len(model.model.layers) if hasattr(model, "model") else "?"
    print(f"  Device: {device} | Dtype: {dtype} | Layers: {n_layers}")
    return model, tokenizer, device


def unload_model(model, tokenizer):
    """Explicitly free model and tokenizer memory between runs."""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  Model unloaded and memory cleared.")


# ============================================================
# TOKENIZATION CHECK
# ============================================================

def check_verb_tokens(tokenizer, model_tag: str) -> dict:
    """
    Returns dict: verb_string -> token_id (or None if multi-token).
    Prints a clear report.
    """
    print(f"\n--- Verb token check: {model_tag} ---")
    verb_map = {}
    all_verbs = set()
    for vp in VERB_PAIRS:
        all_verbs.add(vp["singular"])
        all_verbs.add(vp["plural"])

    for verb in sorted(all_verbs):
        # Try with leading space (standard for mid-sentence tokens)
        ids_space = tokenizer.encode(" " + verb, add_special_tokens=False)
        ids_plain = tokenizer.encode(verb,        add_special_tokens=False)

        if len(ids_space) == 1:
            verb_map[verb] = ids_space[0]
            print(f"  ' {verb}' → single token {ids_space[0]}  ✓")
        elif len(ids_plain) == 1:
            verb_map[verb] = ids_plain[0]
            print(f"  '{verb}' (no space) → single token {ids_plain[0]}  ✓ (no-space fallback)")
        else:
            verb_map[verb] = None
            print(f"  ' {verb}' → MULTI-TOKEN {ids_space}  ✗  (will skip this verb pair for {model_tag})")

    return verb_map


def valid_verb_pairs_for_model(verb_map: dict) -> list:
    """Return only verb pairs where both verbs are single tokens."""
    valid = []
    for vp in VERB_PAIRS:
        if verb_map.get(vp["singular"]) is not None and verb_map.get(vp["plural"]) is not None:
            valid.append(vp)
    skipped = [vp["pair_id"] for vp in VERB_PAIRS if vp not in valid]
    if skipped:
        print(f"  Skipping verb pairs (multi-token): {skipped}")
    return valid


# ============================================================
# SCORING
# ============================================================

@torch.no_grad()
def score_prompts_batch(model, tokenizer, device, prompts: list,
                         correct_ids: list, wrong_ids: list,
                         batch_size: int = 16) -> list:
    """
    Returns list of dicts with logit_correct, logit_wrong, margin,
    prob_correct, prob_wrong, rank_correct for each prompt.
    """
    results = []
    n = len(prompts)

    for start in range(0, n, batch_size):
        batch_prompts     = prompts[start:start+batch_size]
        batch_correct_ids = correct_ids[start:start+batch_size]
        batch_wrong_ids   = wrong_ids[start:start+batch_size]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out    = model(**enc)
        logits = out.logits.float()  # [B, seq, vocab]
        mask   = enc["attention_mask"]
        last_pos = mask.sum(dim=1) - 1

        for b in range(len(batch_prompts)):
            pos   = last_pos[b].item()
            lgt   = logits[b, pos, :]           # [vocab]
            probs = torch.softmax(lgt, dim=-1)

            cid = batch_correct_ids[b]
            wid = batch_wrong_ids[b]

            lc = lgt[cid].item()
            lw = lgt[wid].item()

            pc = probs[cid].item()
            pw = probs[wid].item()

            # rank of correct verb among all vocab (1 = highest)
            rank_c = int((lgt > lc).sum().item()) + 1

            results.append({
                "logit_correct": lc,
                "logit_wrong":   lw,
                "margin":        lc - lw,
                "prob_correct":  pc,
                "prob_wrong":    pw,
                "rank_correct":  rank_c,
            })

        del out, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ============================================================
# AUDIT ONE MODEL
# ============================================================

def locate_token_positions(tokenizer, prompt: str, subject: str, distractor: str) -> dict:
    """
    Find token positions of subject, distractor, and final token for a prompt.
    Returns dict with subject_token_start, subject_token_end,
    distractor_token_start, distractor_token_end, final_token_position.
    All positions are -1 if not found.
    These are stored in the audit CSV for use in patching/steering experiments.
    """
    enc = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    final_pos = len(ids) - 1

    def find_span(word: str):
        # Try with leading space first, then without
        for prefix in (" ", ""):
            word_ids = tokenizer.encode(prefix + word, add_special_tokens=False)
            n = len(word_ids)
            for i in range(len(ids) - n + 1):
                if ids[i:i+n] == word_ids:
                    return i, i + n - 1  # start, end (inclusive)
        return -1, -1

    subj_start, subj_end = find_span(subject)
    dist_start, dist_end = find_span(distractor)

    return {
        "subject_token_start":     subj_start,
        "subject_token_end":       subj_end,
        "distractor_token_start":  dist_start,
        "distractor_token_end":    dist_end,
        "final_token_position":    final_pos,
        "prompt_token_length":     len(ids),
    }


def audit_model(model, tokenizer, device, model_tag: str,
                clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score every prompt in clean_df for this model.
    Print per-sentence pass/fail.
    Save token positions for patching/steering experiments.
    Return audit dataframe.
    """

    verb_map   = check_verb_tokens(tokenizer, model_tag)
    valid_vps  = valid_verb_pairs_for_model(verb_map)
    valid_pair_ids = {vp["pair_id"] for vp in valid_vps}

    # Filter to valid verb pairs only
    df = clean_df[clean_df["verb_pair"].isin(valid_pair_ids)].copy()

    print(f"\n{'='*80}")
    print(f"AUDITING {model_tag}  |  {len(df['item_id'].unique())} base items  "
          f"|  {len(df)} rows  |  verb pairs: {sorted(valid_pair_ids)}")
    print(f"{'='*80}")

    # ── Scoring ───────────────────────────────────────────────────────────────
    prompts     = df["prompt"].tolist()
    correct_ids = [verb_map[v] for v in df["correct_verb"].tolist()]
    wrong_ids   = [verb_map[v] for v in df["wrong_verb"].tolist()]

    scores = score_prompts_batch(model, tokenizer, device,
                                  prompts, correct_ids, wrong_ids)
    score_df = pd.DataFrame(scores)
    df = df.reset_index(drop=True)
    audit_df = pd.concat([df, score_df], axis=1)
    audit_df["model_tag"]        = model_tag
    audit_df["pass_margin_gt_0"] = audit_df["margin"] > MARGIN_CLEAN
    audit_df["pass_margin_gt_1"] = audit_df["margin"] > MARGIN_STRONG

    # ── Token position saving (Fix 6) ────────────────────────────────────────
    # Compute once per unique prompt (all verb pairs share the same prompt text)
    print(f"\nComputing token positions for {model_tag}...")
    pos_rows = []
    for _, row in audit_df.iterrows():
        pos = locate_token_positions(
            tokenizer, row["prompt"], row["subject_singular"]
            if row["subject_number"] == "singular" else row["subject_plural"],
            row["distractor_singular"]
            if row["distractor_number"] == "singular" else row["distractor_plural"],
        )
        pos_rows.append(pos)
    pos_df = pd.DataFrame(pos_rows)
    audit_df = pd.concat([audit_df.reset_index(drop=True),
                        pos_df.reset_index(drop=True)], axis=1)

    # Check whether subject/distractor token spans were found correctly
    missing_subj = int((audit_df["subject_token_start"] == -1).sum())
    missing_dist = int((audit_df["distractor_token_start"] == -1).sum())

    print(f"[{model_tag}] Missing subject spans: {missing_subj}/{len(audit_df)}")
    print(f"[{model_tag}] Missing distractor spans: {missing_dist}/{len(audit_df)}")

    # ── Per-sentence print ────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"PER-SENTENCE RESULTS  [{model_tag}]")
    print(f"{'─'*80}")

    pass_count = fail_count = strong_count = 0
    for _, row in audit_df.iterrows():
        p0 = bool(row["pass_margin_gt_0"])
        p1 = bool(row["pass_margin_gt_1"])
        status_0 = "✓ PASS  " if p0 else "✗ FAIL  "
        status_1 = "★ STRONG" if p1 else "        "
        if p0: pass_count += 1
        else:  fail_count += 1
        if p1: strong_count += 1
        print(
            f"[{row['item_id']}] {row['verb_pair']:12s} {row['condition']}  "
            f"correct={row['correct_verb']:5s} wrong={row['wrong_verb']:5s}  "
            f"margin={row['margin']:+7.4f}  {status_0} {status_1}"
            f"\n    prompt : {row['prompt']}"
            f"\n    subj_pos={row['subject_token_start']}  "
            f"dist_pos={row['distractor_token_start']}  "
            f"final_pos={row['final_token_position']}"
        )

    total = pass_count + fail_count
    print(f"\n{'─'*80}")
    print(f"[{model_tag}] SUMMARY: {pass_count}/{total} PASS  "
          f"({100*pass_count/total:.1f}%)  |  {strong_count}/{total} STRONG")
    print(f"{'─'*80}")

    return audit_df


# ============================================================
# ITEM-LEVEL FILTERING
# ============================================================

def filter_items(audit_df: pd.DataFrame, margin_thresh: float,
                  model_tag: str) -> pd.DataFrame:
    """
    A base item is clean for a verb pair only if ALL 4 conditions pass.
    Filter at item_id × verb_pair level.
    """
    # group by (item_id, verb_pair) and check all 4 conditions pass
    group = audit_df.groupby(["item_id", "verb_pair"])["pass_margin_gt_0" if margin_thresh == MARGIN_CLEAN
                                                         else "pass_margin_gt_1"].all()
    passing = group[group].reset_index()[["item_id", "verb_pair"]]
    passing["_keep"] = True

    merged = audit_df.merge(passing, on=["item_id", "verb_pair"], how="left")
    clean  = merged[merged["_keep"] == True].drop(columns=["_keep"])

    n_base = len(clean[["item_id", "verb_pair"]].drop_duplicates())
    print(f"\n[{model_tag}] margin > {margin_thresh}: {n_base} clean (item_id, verb_pair) groups  "
          f"| {len(clean)} rows")
    return clean


# ============================================================
# SUMMARY
# ============================================================

def compute_summary_row(audit_df: pd.DataFrame, model_tag: str,
                          vp: dict) -> dict:
    df = audit_df[
        (audit_df["model_tag"]  == model_tag) &
        (audit_df["verb_pair"] == vp["pair_id"])
    ].copy()

    if df.empty:
        return None

    n_base_raw = len(df["item_id"].unique())

    clean_mask  = df.groupby(["item_id"])["pass_margin_gt_0"].transform("all")
    strong_mask = df.groupby(["item_id"])["pass_margin_gt_1"].transform("all")

    n_clean  = len(df[clean_mask]["item_id"].unique())
    n_strong = len(df[strong_mask]["item_id"].unique())

    def acc(cond):
        s = df[df["condition"] == cond]
        return s["pass_margin_gt_0"].mean() if len(s) else float("nan")

    def mean_margin(cond):
        s = df[df["condition"] == cond]
        return s["margin"].mean() if len(s) else float("nan")

    ss_m = mean_margin("SS")
    sp_m = mean_margin("SP")
    ps_m = mean_margin("PS")
    pp_m = mean_margin("PP")

    plural_attractor  = ss_m - sp_m   # how much plural dist hurts singular subj
    singular_attractor = pp_m - ps_m  # how much singular dist hurts plural subj

    return {
        "model":                        model_tag,
        "verb_pair":                    vp["pair_id"],
        "raw_base_items":               n_base_raw,
        "clean_base_items_margin_gt_0": n_clean,
        "clean_base_items_margin_gt_1": n_strong,
        "SS_accuracy":                  acc("SS"),
        "SP_accuracy":                  acc("SP"),
        "PS_accuracy":                  acc("PS"),
        "PP_accuracy":                  acc("PP"),
        "mean_SS_margin":               ss_m,
        "mean_SP_margin":               sp_m,
        "mean_PS_margin":               ps_m,
        "mean_PP_margin":               pp_m,
        "plural_attractor_effect":      plural_attractor,
        "singular_attractor_effect":    singular_attractor,
        "asymmetry":                    plural_attractor - singular_attractor,
    }


# ============================================================
# INTERSECTION ACROSS MODELS
# ============================================================

def build_intersection(all_audit_dfs: dict, margin_thresh: float = MARGIN_CLEAN) -> pd.DataFrame:
    """
    Keep only (item_id, verb_pair) that pass in ALL models.
    """
    sets = []
    for tag, df in all_audit_dfs.items():
        col = "pass_margin_gt_0" if margin_thresh == MARGIN_CLEAN else "pass_margin_gt_1"
        grp = df.groupby(["item_id", "verb_pair"])[col].all()
        passing = set(grp[grp].index.tolist())
        sets.append(passing)

    if not sets:
        return pd.DataFrame()

    common = sets[0]
    for s in sets[1:]:
        common = common & s

    print(f"\nIntersection across {len(all_audit_dfs)} models: "
          f"{len(common)} (item_id, verb_pair) groups pass in all models.")

    # Pull rows from first model's df (any would do, all share same prompts)
    first_df = list(all_audit_dfs.values())[0]
    first_df["_key"] = list(zip(first_df["item_id"], first_df["verb_pair"]))
    inter_df = first_df[first_df["_key"].isin(common)].drop(columns=["_key"]).copy()

    # Drop model-specific score columns, keep structure only
    keep_cols = [c for c in inter_df.columns if c not in
                 ["model_tag", "logit_correct", "logit_wrong", "margin",
                  "prob_correct", "prob_wrong", "rank_correct",
                  "pass_margin_gt_0", "pass_margin_gt_1"]]
    return inter_df[keep_cols].drop_duplicates()


# ============================================================
# MAIN
# ============================================================

def main():
    torch.set_grad_enabled(False)

    # ── 1. Build raw dataset ─────────────────────────────────────────────────
    print("\n" + "="*80)
    print("STEP 1: BUILD RAW DATASET")
    print("="*80)
    raw_df = build_raw_dataset()
    raw_path = os.path.join(OUTPUT_DIR, "agreement_items_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved: {raw_path}")

    # ── 2. Manual clean ──────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("STEP 2: MANUAL CLEAN")
    print("="*80)
    clean_df = build_manual_clean(raw_df)
    manual_clean_path = os.path.join(OUTPUT_DIR, "agreement_items_manual_clean.csv")
    clean_df.to_csv(manual_clean_path, index=False)
    print(f"Saved: {manual_clean_path}")

    # ── 3. Per-model audit ───────────────────────────────────────────────────
    all_audit_dfs   = {}
    summary_rows    = []

    for model_cfg in MODELS:
        model_id  = model_cfg["model_id"]
        model_tag = model_cfg["tag"]

        model, tokenizer, device = load_model(model_id, model_cfg.get("dtype", "float16"))

        audit_df = audit_model(model, tokenizer, device, model_tag, clean_df)

        # Save full audit
        audit_path = os.path.join(OUTPUT_DIR, f"agreement_audit_{model_tag}.csv")
        audit_df.to_csv(audit_path, index=False)
        print(f"\nSaved audit: {audit_path}")

        all_audit_dfs[model_tag] = audit_df

        # Save filtered sets
        clean_set  = filter_items(audit_df, MARGIN_CLEAN,  model_tag)
        strong_set = filter_items(audit_df, MARGIN_STRONG, model_tag)

        clean_path  = os.path.join(OUTPUT_DIR, f"clean_items_{model_tag}_margin_gt_0.csv")
        strong_path = os.path.join(OUTPUT_DIR, f"clean_items_{model_tag}_margin_gt_1.csv")
        clean_set.to_csv(clean_path,  index=False)
        strong_set.to_csv(strong_path, index=False)
        print(f"Saved: {clean_path}")
        print(f"Saved: {strong_path}")

        # Summary rows — reuse verb_map computed inside audit_model
        verb_map  = check_verb_tokens(tokenizer, model_tag)
        valid_vps = valid_verb_pairs_for_model(verb_map)
        for vp in valid_vps:
            row = compute_summary_row(audit_df, model_tag, vp)
            if row:
                summary_rows.append(row)

        # ── FIX: unload model AND tokenizer together ──────────────────────────
        unload_model(model, tokenizer)

    # ── 4. Cross-model intersection ──────────────────────────────────────────
    print("\n" + "="*80)
    print("STEP 4: CROSS-MODEL INTERSECTION")
    print("="*80)
    inter_df = build_intersection(all_audit_dfs, MARGIN_CLEAN)
    inter_path = os.path.join(OUTPUT_DIR, "clean_items_all_models_intersection.csv")
    inter_df.to_csv(inter_path, index=False)
    print(f"Saved: {inter_path}")

    # ── 5. Dataset summary ───────────────────────────────────────────────────
    print("\n" + "="*80)
    print("STEP 5: DATASET SUMMARY")
    print("="*80)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "dataset_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # Print summary table to stdout
    print("\n" + "="*80)
    print("DATASET SUMMARY TABLE")
    print("="*80)
    with pd.option_context("display.max_columns", None,
                            "display.width", 160,
                            "display.float_format", "{:.4f}".format):
        print(summary_df.to_string(index=False))

    # ── 6. Final counts ──────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("FINAL COUNTS")
    print("="*80)
    print(f"Raw base items:                {len(raw_df['item_id'].unique())}")
    print(f"Manual-clean base items:       {len(clean_df['item_id'].unique())}")
    for tag, df in all_audit_dfs.items():
        n0 = len(filter_items(df, MARGIN_CLEAN,  tag)["item_id"].unique())
        n1 = len(filter_items(df, MARGIN_STRONG, tag)["item_id"].unique())
        print(f"  {tag:20s} margin>0: {n0:4d}   margin>1: {n1:4d}")
    if not inter_df.empty:
        print(f"Intersection (all models):     {len(inter_df['item_id'].unique())}")

    print("\n" + "="*80)
    print("DONE — outputs in:", OUTPUT_DIR)
    print("="*80)


if __name__ == "__main__":
    main()