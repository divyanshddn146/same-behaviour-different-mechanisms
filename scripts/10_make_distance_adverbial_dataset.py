"""
make_distance_adverbial_dataset.py

Creates a cleaner long-distance subject–verb agreement dataset.

Goal:
  Increase subject-to-verb distance WITHOUT adding extra noun attractors
  like "station", "hotel", "district", etc.

Template:
  The {subject} of the {distractor}{adverbial_filler}

Target next token:
  " is" / " are"
  " was" / " were"

Examples:
  short:
    The manager of the restaurants

  medium:
    The manager of the restaurants, quite clearly,

  long:
    The manager of the restaurants, quite clearly, rather obviously, as expected,

  extra_long:
    The manager of the restaurants, quite clearly, rather obviously, as expected, at least for now, in most cases,

Important:
  No trailing space in prompt.
  Score target as prompt + " is", prompt + " are", etc.
"""

import csv
import random
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SEED = 42
random.seed(SEED)

OUT_DIR = Path("data/distance_adverbial")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_CSV = OUT_DIR / "agreement_distance_adverbial_full.csv"
EVAL_CSV = OUT_DIR / "agreement_distance_adverbial_eval_balanced.csv"
MANIFEST = OUT_DIR / "dataset_manifest.txt"

# 2 aux pairs × 4 distances × 4 conditions × 400 = 12,800 examples
N_EVAL_PER_CELL = 400

NOUN_PAIRS = [
    ("manager", "managers"),
    ("teacher", "teachers"),
    ("worker", "workers"),
    ("doctor", "doctors"),
    ("student", "students"),
    ("driver", "drivers"),
    ("farmer", "farmers"),
    ("artist", "artists"),
    ("lawyer", "lawyers"),
    ("officer", "officers"),
    ("engineer", "engineers"),
    ("nurse", "nurses"),
    ("chef", "chefs"),
    ("pilot", "pilots"),
    ("writer", "writers"),
    ("actor", "actors"),
    ("owner", "owners"),
    ("leader", "leaders"),
    ("member", "members"),
    ("player", "players"),
    ("visitor", "visitors"),
    ("customer", "customers"),
    ("director", "directors"),
    ("assistant", "assistants"),
    ("researcher", "researchers"),
    ("analyst", "analysts"),
    ("consultant", "consultants"),
    ("designer", "designers"),
    ("developer", "developers"),
    ("operator", "operators"),
    ("inspector", "inspectors"),
    ("supervisor", "supervisors"),
    ("advisor", "advisors"),
    ("employee", "employees"),
    ("specialist", "specialists"),
    ("technician", "technicians"),
    ("coordinator", "coordinators"),
    ("administrator", "administrators"),
    ("volunteer", "volunteers"),
    ("representative", "representatives"),
    ("contractor", "contractors"),
]

# These fillers add distance without adding new noun attractors.
# Comma-final prompts naturally invite a finite verb next.
DISTANCE_FILLERS = [
    {
        "distance_bucket": "short",
        "distance_level": 0,
        "filler_id": "none",
        "filler": "",
        "description": "base template",
    },
    {
        "distance_bucket": "medium",
        "distance_level": 1,
        "filler_id": "quite_clearly",
        "filler": ", quite clearly,",
        "description": "short parenthetical adverbial",
    },
    {
        "distance_bucket": "long",
        "distance_level": 2,
        "filler_id": "clearly_obviously_expected",
        "filler": ", quite clearly, rather obviously, as expected,",
        "description": "longer parenthetical adverbial chain",
    },
    {
        "distance_bucket": "extra_long",
        "distance_level": 3,
        "filler_id": "clearly_obviously_expected_now_cases",
        "filler": ", quite clearly, rather obviously, as expected, at least for now, in most cases,",
        "description": "extra-long parenthetical adverbial chain",
    },
]

AUX_PAIRS = [
    {
        "aux_pair": "be_present",
        "singular_verb": "is",
        "plural_verb": "are",
    },
    {
        "aux_pair": "be_past",
        "singular_verb": "was",
        "plural_verb": "were",
    },
]

CONDITIONS = ["SS", "SP", "PS", "PP"]


# ============================================================
# HELPERS
# ============================================================

def make_condition(subject_sg, subject_pl, distractor_sg, distractor_pl, condition):
    """
    SS = singular subject, singular distractor
    SP = singular subject, plural distractor
    PS = plural subject, singular distractor
    PP = plural subject, plural distractor
    """
    if condition == "SS":
        return subject_sg, "sg", distractor_sg, "sg"
    if condition == "SP":
        return subject_sg, "sg", distractor_pl, "pl"
    if condition == "PS":
        return subject_pl, "pl", distractor_sg, "sg"
    if condition == "PP":
        return subject_pl, "pl", distractor_pl, "pl"
    raise ValueError(f"Unknown condition: {condition}")


def correct_wrong_verb(subject_num, singular_verb, plural_verb):
    if subject_num == "sg":
        return singular_verb, plural_verb
    return plural_verb, singular_verb


def make_prompt(subject, distractor, filler):
    """
    No trailing space.
    Score target as prompt + " is" / " are" / " was" / " were".
    """
    return f"The {subject} of the {distractor}{filler}"


def approx_word_distance(prompt, subject, distractor):
    words = prompt.replace(",", " ,").split()

    try:
        subj_idx = words.index(subject)
    except ValueError:
        subj_idx = None

    try:
        dist_idx = words.index(distractor)
    except ValueError:
        dist_idx = None

    pred_site_idx = len(words)

    subj_to_pred = None if subj_idx is None else pred_site_idx - subj_idx
    dist_to_pred = None if dist_idx is None else pred_site_idx - dist_idx

    return len(words), subj_to_pred, dist_to_pred


# ============================================================
# BUILD DATASET
# ============================================================

def build_full_dataset():
    rows = []
    pair_id = 0

    for subj_idx, (subj_sg, subj_pl) in enumerate(NOUN_PAIRS):
        for dist_idx, (dist_sg, dist_pl) in enumerate(NOUN_PAIRS):
            if subj_idx == dist_idx:
                continue

            pair_id += 1

            for aux_cfg in AUX_PAIRS:
                aux_pair = aux_cfg["aux_pair"]
                singular_verb = aux_cfg["singular_verb"]
                plural_verb = aux_cfg["plural_verb"]

                for dist_cfg in DISTANCE_FILLERS:
                    distance_bucket = dist_cfg["distance_bucket"]
                    distance_level = dist_cfg["distance_level"]
                    filler_id = dist_cfg["filler_id"]
                    filler = dist_cfg["filler"]

                    for condition in CONDITIONS:
                        subject, subject_num, distractor, distractor_num = make_condition(
                            subject_sg=subj_sg,
                            subject_pl=subj_pl,
                            distractor_sg=dist_sg,
                            distractor_pl=dist_pl,
                            condition=condition,
                        )

                        correct, wrong = correct_wrong_verb(
                            subject_num=subject_num,
                            singular_verb=singular_verb,
                            plural_verb=plural_verb,
                        )

                        prompt = make_prompt(subject, distractor, filler)

                        n_words, subj_to_pred, dist_to_pred = approx_word_distance(
                            prompt=prompt,
                            subject=subject,
                            distractor=distractor,
                        )

                        rows.append({
                            "item_id": f"{aux_pair}_{distance_bucket}_{pair_id}_{condition}",
                            "pair_id": pair_id,

                            "subject_lemma_sg": subj_sg,
                            "subject_lemma_pl": subj_pl,
                            "distractor_lemma_sg": dist_sg,
                            "distractor_lemma_pl": dist_pl,

                            "aux_pair": aux_pair,
                            "singular_verb": singular_verb,
                            "plural_verb": plural_verb,

                            "distance_bucket": distance_bucket,
                            "distance_level": distance_level,
                            "filler_id": filler_id,
                            "filler": filler,

                            "condition": condition,
                            "subject": subject,
                            "subject_num": subject_num,
                            "distractor": distractor,
                            "distractor_num": distractor_num,

                            "prompt": prompt,
                            "correct_verb": correct,
                            "wrong_verb": wrong,
                            "correct_token_text": " " + correct,
                            "wrong_token_text": " " + wrong,

                            "n_words_prompt": n_words,
                            "approx_subject_to_pred_words": subj_to_pred,
                            "approx_distractor_to_pred_words": dist_to_pred,

                            "template": "The {subject} of the {distractor}{adverbial_filler}",
                            "target_depends_on": "subject_number",
                            "dataset_name": "agreement_distance_adverbial",
                        })

    return pd.DataFrame(rows)


def make_balanced_eval_subset(df):
    sampled = []
    group_cols = ["aux_pair", "distance_bucket", "condition"]

    for key, sub in df.groupby(group_cols):
        if len(sub) < N_EVAL_PER_CELL:
            raise ValueError(
                f"Cell {key} has only {len(sub)} examples; need {N_EVAL_PER_CELL}"
            )

        sampled.append(sub.sample(n=N_EVAL_PER_CELL, random_state=SEED))

    out = pd.concat(sampled, ignore_index=True)
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return out


def write_manifest(full_df, eval_df):
    lines = []
    lines.append("=" * 80)
    lines.append("ADVERBIAL DISTANCE AGREEMENT DATASET")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Purpose:")
    lines.append("  Increase subject-to-verb distance without adding extra noun attractors.")
    lines.append("")
    lines.append("Template:")
    lines.append("  The {subject} of the {distractor}{adverbial_filler}")
    lines.append("")
    lines.append("Target next-token auxiliaries:")
    lines.append("  be_present: is / are")
    lines.append("  be_past:    was / were")
    lines.append("")
    lines.append("Distance fillers:")
    for d in DISTANCE_FILLERS:
        lines.append(
            f"  {d['distance_bucket']:10s} | level={d['distance_level']} | "
            f"filler='{d['filler']}' | {d['description']}"
        )
    lines.append("")
    lines.append("Important design choices:")
    lines.append("  - No extra noun attractors in the filler.")
    lines.append("  - No trailing space in prompt.")
    lines.append("  - Score target as prompt + ' is', prompt + ' are', etc.")
    lines.append("  - Behavioural audit required before causal analysis.")
    lines.append("")
    lines.append("Full counts:")
    lines.append(str(full_df.groupby(["aux_pair", "distance_bucket", "condition"]).size()))
    lines.append("")
    lines.append("Eval counts:")
    lines.append(str(eval_df.groupby(["aux_pair", "distance_bucket", "condition"]).size()))
    lines.append("")
    lines.append("Mean approximate distances:")
    lines.append(
        str(
            full_df.groupby("distance_bucket")[
                ["n_words_prompt", "approx_subject_to_pred_words", "approx_distractor_to_pred_words"]
            ].mean()
        )
    )
    lines.append("")

    MANIFEST.write_text("\n".join(lines), encoding="utf-8")


def main():
    full_df = build_full_dataset()
    eval_df = make_balanced_eval_subset(full_df)

    full_df.to_csv(FULL_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    eval_df.to_csv(EVAL_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    write_manifest(full_df, eval_df)

    print("=" * 80)
    print("ADVERBIAL DISTANCE DATASET CREATED")
    print("=" * 80)
    print(f"Full CSV: {FULL_CSV.resolve()}")
    print(f"Eval CSV: {EVAL_CSV.resolve()}")
    print(f"Manifest: {MANIFEST.resolve()}")
    print("")
    print(f"Full rows: {len(full_df)}")
    print(f"Eval rows: {len(eval_df)}")
    print("")
    print("Eval counts:")
    print(eval_df.groupby(["aux_pair", "distance_bucket", "condition"]).size())
    print("")
    print("Mean approximate word distances:")
    print(
        full_df.groupby("distance_bucket")[
            ["n_words_prompt", "approx_subject_to_pred_words", "approx_distractor_to_pred_words"]
        ].mean()
    )
    print("")
    print("Example rows:")
    print(
        eval_df[
            [
                "aux_pair",
                "distance_bucket",
                "condition",
                "prompt",
                "correct_token_text",
                "wrong_token_text",
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()