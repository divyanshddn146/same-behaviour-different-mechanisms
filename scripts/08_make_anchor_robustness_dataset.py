"""
make_anchor_robustness_dataset.py

Creates a serious anchor-robustness dataset for controlled subject–verb agreement.

Goal:
  Test whether the main agreement result is specific to the fixed phrase
  "in question", by varying the final anchor phrase while keeping the
  subject/distractor number manipulation controlled.

Template:
  The {subject} of the {distractor} {anchor}

Target next token:
  " is" / " are"
  " was" / " were"

Conditions:
  SS = singular subject, singular distractor
  SP = singular subject, plural distractor
  PS = plural subject, singular distractor
  PP = plural subject, plural distractor

Outputs:
  anchor_robustness_dataset/agreement_anchor_robustness_full.csv
  anchor_robustness_dataset/agreement_anchor_robustness_eval_balanced.csv
  anchor_robustness_dataset/dataset_manifest.txt
"""

import csv
import random
from pathlib import Path

import pandas as pd


SEED = 42
random.seed(SEED)

OUT_DIR = Path("data/anchor_robustness")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_CSV = OUT_DIR / "agreement_anchor_robustness_full.csv"
EVAL_CSV = OUT_DIR / "agreement_anchor_robustness_eval_balanced.csv"
MANIFEST = OUT_DIR / "dataset_manifest.txt"

# 2 aux pairs × 8 anchors × 4 conditions × 300 = 19,200 examples
N_EVAL_PER_CELL = 300

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

# These anchors are short, neutral, and compatible with both singular/plural subjects.
# They are designed to make an auxiliary continuation natural.
ANCHORS = [
    {"anchor_id": "in_question", "anchor": "in question"},
    {"anchor_id": "under_review", "anchor": "under review"},
    {"anchor_id": "on_duty", "anchor": "on duty"},
    {"anchor_id": "on_site", "anchor": "on site"},
    {"anchor_id": "in_charge", "anchor": "in charge"},
    {"anchor_id": "at_work", "anchor": "at work"},
    {"anchor_id": "on_call", "anchor": "on call"},
    {"anchor_id": "under_observation", "anchor": "under observation"},
]

AUX_PAIRS = [
    {"aux_pair": "be_present", "singular_verb": "is", "plural_verb": "are"},
    {"aux_pair": "be_past", "singular_verb": "was", "plural_verb": "were"},
]

CONDITIONS = ["SS", "SP", "PS", "PP"]


def make_condition(subject_sg, subject_pl, distractor_sg, distractor_pl, condition):
    if condition == "SS":
        return subject_sg, "sg", distractor_sg, "sg"
    if condition == "SP":
        return subject_sg, "sg", distractor_pl, "pl"
    if condition == "PS":
        return subject_pl, "pl", distractor_sg, "sg"
    if condition == "PP":
        return subject_pl, "pl", distractor_pl, "pl"
    raise ValueError(condition)


def correct_wrong_verb(subject_num, singular_verb, plural_verb):
    if subject_num == "sg":
        return singular_verb, plural_verb
    return plural_verb, singular_verb


def make_prompt(subject, distractor, anchor):
    # No trailing space. Score next token as " is", " are", etc.
    return f"The {subject} of the {distractor} {anchor}"


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

                for anchor_cfg in ANCHORS:
                    anchor_id = anchor_cfg["anchor_id"]
                    anchor = anchor_cfg["anchor"]

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

                        prompt = make_prompt(subject, distractor, anchor)

                        rows.append({
                            "item_id": f"{aux_pair}_{anchor_id}_{pair_id}_{condition}",
                            "pair_id": pair_id,

                            "subject_lemma_sg": subj_sg,
                            "subject_lemma_pl": subj_pl,
                            "distractor_lemma_sg": dist_sg,
                            "distractor_lemma_pl": dist_pl,

                            "aux_pair": aux_pair,
                            "singular_verb": singular_verb,
                            "plural_verb": plural_verb,

                            "anchor_id": anchor_id,
                            "anchor": anchor,

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

                            "template": "The {subject} of the {distractor} {anchor}",
                            "target_depends_on": "subject_number",
                            "dataset_name": "agreement_anchor_robustness",
                        })

    return pd.DataFrame(rows)


def make_balanced_eval_subset(df):
    sampled = []

    group_cols = ["aux_pair", "anchor_id", "condition"]

    for key, sub in df.groupby(group_cols):
        if len(sub) < N_EVAL_PER_CELL:
            raise ValueError(
                f"Cell {key} has only {len(sub)} examples; "
                f"need {N_EVAL_PER_CELL}"
            )

        sampled.append(sub.sample(n=N_EVAL_PER_CELL, random_state=SEED))

    out = pd.concat(sampled, ignore_index=True)
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return out


def write_manifest(full_df, eval_df):
    lines = []
    lines.append("=" * 80)
    lines.append("ANCHOR ROBUSTNESS AGREEMENT DATASET")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Purpose:")
    lines.append("  Test whether controlled agreement results are specific to 'in question'.")
    lines.append("")
    lines.append("Template:")
    lines.append("  The {subject} of the {distractor} {anchor}")
    lines.append("")
    lines.append("Target next-token auxiliaries:")
    lines.append("  be_present: is / are")
    lines.append("  be_past:    was / were")
    lines.append("")
    lines.append("Anchors:")
    for a in ANCHORS:
        lines.append(f"  {a['anchor_id']:20s} -> {a['anchor']}")
    lines.append("")
    lines.append("Full dataset counts:")
    lines.append(str(full_df.groupby(["aux_pair", "anchor_id", "condition"]).size()))
    lines.append("")
    lines.append("Eval dataset counts:")
    lines.append(str(eval_df.groupby(["aux_pair", "anchor_id", "condition"]).size()))
    lines.append("")
    lines.append("Recommended use:")
    lines.append("  1. Run behavioural audit first.")
    lines.append("  2. Use margin_accuracy and mean_margin as the main forced-choice metrics.")
    lines.append("  3. Use clean margin>0 or margin>1 examples for steering if needed.")
    lines.append("  4. This is a template-anchor robustness check, not a new main dataset.")
    lines.append("")

    MANIFEST.write_text("\n".join(lines), encoding="utf-8")


def main():
    full_df = build_full_dataset()
    eval_df = make_balanced_eval_subset(full_df)

    full_df.to_csv(FULL_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    eval_df.to_csv(EVAL_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    write_manifest(full_df, eval_df)

    print("=" * 80)
    print("ANCHOR ROBUSTNESS DATASET CREATED")
    print("=" * 80)
    print(f"Full CSV: {FULL_CSV.resolve()}")
    print(f"Eval CSV: {EVAL_CSV.resolve()}")
    print(f"Manifest: {MANIFEST.resolve()}")
    print("")
    print(f"Full rows: {len(full_df)}")
    print(f"Eval rows: {len(eval_df)}")
    print("")
    print("Eval counts:")
    print(eval_df.groupby(["aux_pair", "anchor_id", "condition"]).size())
    print("")
    print("Example rows:")
    print(
        eval_df[
            ["aux_pair", "anchor_id", "condition", "prompt", "correct_token_text", "wrong_token_text"]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()