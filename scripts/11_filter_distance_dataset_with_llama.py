from pathlib import Path
import pandas as pd

# ============================================================
# INPUTS
# ============================================================

BASE_CSV = Path("data/distance_adverbial/agreement_distance_adverbial_eval_balanced.csv")

# Raw behavioural audit output for Llama used to create the clean distance set.
LLAMA_RAW = Path("results/distance_stress/behavioural/raw/behavior_raw_llama32_3b.csv")

OUT_DIR = Path("data/distance_adverbial_filtered")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MARGIN_GT0 = OUT_DIR / "agreement_distance_adverbial_llama_clean_margin_gt0.csv"
OUT_MARGIN_GT1 = OUT_DIR / "agreement_distance_adverbial_llama_clean_margin_gt1.csv"
OUT_BALANCED_GT0 = OUT_DIR / "agreement_distance_adverbial_llama_clean_margin_gt0_balanced.csv"

SEED = 42


# ============================================================
# MAIN
# ============================================================

def main():
    base = pd.read_csv(BASE_CSV)
    llama = pd.read_csv(LLAMA_RAW)

    print("=" * 80)
    print("FILTERING ADVERBIAL DATASET USING LLAMA BEHAVIOURAL AUDIT")
    print("=" * 80)

    print(f"Base rows:  {len(base)}")
    print(f"Llama rows: {len(llama)}")

    # Make sure item IDs match.
    base_ids = set(base["item_id"])
    llama_ids = set(llama["item_id"])

    missing_in_llama = base_ids - llama_ids
    extra_in_llama = llama_ids - base_ids

    print(f"Missing in Llama audit: {len(missing_in_llama)}")
    print(f"Extra in Llama audit:   {len(extra_in_llama)}")

    if missing_in_llama:
        print("WARNING: Some base item_ids were not found in Llama audit.")

    # ----------------------------
    # Filter 1: margin > 0
    # ----------------------------
    clean0_ids = set(llama[llama["margin_gt_0"] == True]["item_id"])
    clean0 = base[base["item_id"].isin(clean0_ids)].copy()

    clean0.to_csv(OUT_MARGIN_GT0, index=False)

    print("\nLlama clean margin > 0:")
    print(f"Rows: {len(clean0)} / {len(base)}")
    print(clean0.groupby(["aux_pair", "distance_bucket", "condition"]).size())

    # ----------------------------
    # Filter 2: margin > 1
    # ----------------------------
    clean1_ids = set(llama[llama["margin_gt_1"] == True]["item_id"])
    clean1 = base[base["item_id"].isin(clean1_ids)].copy()

    clean1.to_csv(OUT_MARGIN_GT1, index=False)

    print("\nLlama clean margin > 1:")
    print(f"Rows: {len(clean1)} / {len(base)}")
    print(clean1.groupby(["aux_pair", "distance_bucket", "condition"]).size())

    # ----------------------------
    # Balanced version of margin > 0
    # ----------------------------
    group_cols = ["aux_pair", "distance_bucket", "condition"]
    counts = clean0.groupby(group_cols).size()
    min_n = int(counts.min())

    print("\nBalanced margin > 0:")
    print(f"Minimum cell count = {min_n}")

    balanced_parts = []
    for key, sub in clean0.groupby(group_cols):
        balanced_parts.append(sub.sample(n=min_n, random_state=SEED))

    balanced = pd.concat(balanced_parts, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    balanced.to_csv(OUT_BALANCED_GT0, index=False)

    print(f"Balanced rows: {len(balanced)}")
    print(balanced.groupby(group_cols).size())

    print("\nSaved:")
    print(f"  {OUT_MARGIN_GT0}")
    print(f"  {OUT_MARGIN_GT1}")
    print(f"  {OUT_BALANCED_GT0}")

    print("\nUse this for causal experiments if you want the clean shared set:")
    print(f"  {OUT_BALANCED_GT0}")


if __name__ == "__main__":
    main()