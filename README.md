# Same Behaviour, Different Mechanisms

Code, cleaned data subsets, processed result CSVs, and figure-generation scripts
for the paper:

**Same Behaviour, Different Mechanisms: Causal Layouts of Agreement Control
Across Three Decoder-Only Language Models**

## Overview

The paper studies subject–verb agreement in three decoder-only language models:

- **Phi-2** (32 layers, partial RoPE)
- **Llama-3.2-3B** (28 layers, RoPE)
- **Qwen2.5-3B** (36 layers, RoPE with long-context scaling)

Behavioural evaluations show that all three models are accurate on subject–verb
agreement and all three show distractor-attraction. But causal interventions
reveal that the models solve the task through different internal layouts:

- **Activation patching** shows that agreement control is subject-driven, not
  distractor-driven, in every model.
- **Linear steering** with a subject-number direction splits the models. Phi-2
  and Llama-3.2-3B are *final-token integrated* (subject/final ratio ≈ 0.9× and
  1.0×). Qwen2.5-3B is *source-localized* (subject/final ratio ≈ 3.7×).
- The dissociation is robust across held-out items, a was/were replication,
  eight anchor templates, all four subject–distractor conditions, and
  adverbial-filler distance stress.

## Repository structure

```text
data/
  agreement/                       core agreement dataset + clean intersections
  anchor_robustness/               anchor-template robustness dataset
  distance_adverbial/              adverbial-filler distance dataset
  distance_adverbial_filtered/     Llama-audited clean subset for distance

scripts/
  01_make_dataset.py               build agreement dataset, per-model audits
  02_behavioral_analysis.py        behavioural metrics + Figure 1
  03_activation_patching_agreement.py     Table 1, 7, 8
  04_activation_steering_agreement.py     Table 2 (main steering) + controls
  05_all_position_steering.py             all-position profile, Figure 3a
  06_temporal_dynamics_analysis.py        Table 3 (Src/Final AUC, onset layer)
  07_heldout_steering.py                  50% train / 50% test steering
  08_make_anchor_robustness_dataset.py    8-anchor template dataset
  09_anchor_robustness_steering.py        Tables 13, 14 (anchor all-four + SP-only)
  10_make_distance_adverbial_dataset.py   4-bucket distance dataset
  11_filter_distance_dataset_with_llama.py  Llama-audited clean distance items
  12_distance_steering.py                 Table 4 (distance S/F ratios)
  13_distance_allpos_steering.py          Table 15 (all-position distance)
  14_distance_allpos_attenuation_analysis.py   attenuation stats + ratios
  plotting/
    generate_behavioural_figures.py       Figure 1, Figure A1
    plot_figure2_steering.py              Figure 2
    plot_figure3_localization_robustness.py   Figure 3

results/
  behavioural/                     per-model audits, margins, summaries
  patching/                        {is_are, was_were} × {summary, layerwise}
  steering/                        {is_are, was_were} × {summary, layerwise}
  all_position_steering/           {is_are, was_were} × {summary, layerwise}
  temporal/                        Src/Final AUC and onset per model
  heldout_steering/                50/50 split steering summaries
  anchor_robustness/               ratios, summaries, train/test samples
  distance_stress/
    behavioural/                   distance-prompt behavioural audit
    steering/                      is_are, was_were subject/final steering
    all_position/                  is_are all-position × distance
    attenuation/                   bootstrap attenuation, key ratios

figures/
  paper_main/                      Figures 1, 2, 3 (paper-final)
  behavioural/                     margins, asymmetry, distributions
  patching/                        per-model layerwise + summary plots
  steering/                        per-model layerwise + summary plots
  all_position_steering/           position × layer heatmaps
  appendix/                        Figure A1 and other appendix figures
```

## Requirements

Python 3.10+. Install with:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```text
torch
transformers
numpy
pandas
matplotlib
tqdm
scipy
```

The model-running scripts (`01`, `03`, `04`, `05`, `07`, `09`, `12`, `13`)
require access to the corresponding HuggingFace model weights. GPU is
recommended; Phi-2 is run in float32 for numerical stability, Llama-3.2-3B and
Qwen2.5-3B in float16.

## Reproducing paper figures

Run from the repository root. The processed result CSVs are already in
`results/`, so figure generation does not require rerunning any model.

**Figure 1** (behavioural agreement):

```bash
python scripts/plotting/generate_behavioural_figures.py
```

**Figure 2** (steering profile at subject/distractor/final):

```bash
python scripts/plotting/plot_figure2_steering.py
```

**Figure 3** (localization and robustness):

```bash
python scripts/plotting/plot_figure3_localization_robustness.py
```

Generated figures are written to `figures/paper_main/`.

## Reproducing analyses from result CSVs

These analyses read from existing `results/` CSVs and do not require GPU.

**Temporal dynamics (Table 3):**

```bash
python scripts/06_temporal_dynamics_analysis.py
```

Default reads `results/all_position_steering/is_are/layerwise/` and writes to
`results/temporal/is_are/`. For the was/were replication, edit the CONFIG
paths at the top of the script.

**Distance attenuation analysis (Appendix H stats):**

```bash
python scripts/14_distance_allpos_attenuation_analysis.py \
  --input results/distance_stress/all_position/is_are/summary \
  --pattern "allpos_distance_role_summary_*.csv" \
  --output_dir results/distance_stress/attenuation/is_are \
  --preview
```

## Rerunning the model experiments

The model-running scripts are included for reproducibility of the original
experiments. Some scripts write to their original scratch output folders by
default; the processed CSVs used by the paper have already been organized under
`results/` and `data/`. The plotting scripts read from the organized repository
paths.

Full end-to-end pipeline, in order:

**1. Build the agreement dataset and per-model audits** (Section 2):

```bash
python scripts/01_make_dataset.py
```

Produces the cleaned agreement datasets and all-model intersections used by the
behavioural and causal experiments.

**2. Behavioural analysis** (Section 3, Table 6):

```bash
python scripts/02_behavioral_analysis.py
```

**3. Activation patching** (Section 4, Tables 1, 7, 8):

```bash
python scripts/03_activation_patching_agreement.py
```

Default runs is/are. For was/were, change the CONFIG block to use
`VERB_PAIR="be_past"`, `SINGULAR_V="was"`, `PLURAL_V="were"`, and the
corresponding was/were output directory.

**4. Main steering** (Section 5, Table 2, Figure 2):

```bash
python scripts/04_activation_steering_agreement.py
```

Same CONFIG-block pattern for was/were.

**5. All-position steering** (Section 5.3, Figure 3a):

```bash
python scripts/05_all_position_steering.py
```

**6. Held-out steering** (Section 6.1, Table 12):

```bash
python scripts/07_heldout_steering.py
```

**7. Anchor-template robustness** (Section 6.3, Tables 13, 14). The auxiliary
pair is intentionally held fixed to is/are so that the check isolates
template variation; auxiliary-pair robustness is covered by the was/were runs
in steps 3–6.

```bash
python scripts/08_make_anchor_robustness_dataset.py
python scripts/09_anchor_robustness_steering.py
```

**8. Distance stress** (Section 6.4, Tables 4, 15):

```bash
python scripts/10_make_distance_adverbial_dataset.py
python scripts/11_filter_distance_dataset_with_llama.py
python scripts/12_distance_steering.py
python scripts/13_distance_allpos_steering.py
python scripts/14_distance_allpos_attenuation_analysis.py \
  --input results/distance_stress/all_position/is_are/summary \
  --pattern "allpos_distance_role_summary_*.csv" \
  --output_dir results/distance_stress/attenuation/is_are \
  --preview
```

The distance dataset is filtered against Llama's behavioural audit
(`results/distance_stress/behavioural/raw/behavior_raw_llama32_3b.csv`) to
retain items where the model is behaviourally accurate before running causal
interventions.

## Result files by paper table/figure

| Paper element | File |
|---|---|
| Figure 1 (behavioural) | `figures/paper_main/figure1_behavioural.pdf` |
| Figure 2 (steering) | `figures/paper_main/figure2_steering_main.pdf` |
| Figure 3 (localization + robustness) | `figures/paper_main/figure3_localization_robustness.pdf` |
| Table 1 (patching subj vs dist) | `results/patching/{is_are,was_were}/summary/patching_bootstrap_summary_*.csv` |
| Table 2 (steering S/D/F) | `results/steering/{is_are,was_were}/summary/steering_bootstrap_*.csv` |
| Table 3 (temporal Src/Final + onset) | `results/temporal/{is_are,was_were}/temporal_dynamics_summary.csv` |
| Table 4 (distance ratios) | `results/distance_stress/steering/is_are/ratios/` |
| Table 6 (behavioural, all verb pairs) | `results/behavioural/summary/behavioral_summary_all_verbs.csv` |
| Tables 7, 8 (patching bidirectional + controls) | `results/patching/{is_are,was_were}/summary/` |
| Tables 10, 11 (steering controls + α sweep) | `results/steering/{is_are,was_were}/summary/steering_alpha_sweep_*.csv` |
| Table 12 (held-out steering) | `results/heldout_steering/{is_are,was_were}/robust_split_*_summary.csv` |
| Tables 13, 14 (anchor robustness) | `results/anchor_robustness/is_are/{ratios,summary}/` |
| Table 15 (distance all-position) | `results/distance_stress/all_position/is_are/summary/` |
| Appendix H attenuation stats | `results/distance_stress/attenuation/is_are/allpos_distance_bootstrap.csv` |

## Notes on omitted files

To keep the repository light, the following are not included:

- Large raw per-item × per-layer intervention outputs
  (`*raw*.csv`, `*_raw_*.csv`)
- Cached hidden-state direction pickles (`*directions*.pkl`)
- HuggingFace model cache files

The included CSVs are sufficient to reproduce every paper table and figure.
The `.gitignore` documents which patterns are excluded.

One raw file is intentionally kept:
`results/distance_stress/behavioural/raw/behavior_raw_llama32_3b.csv`. This is
used by `11_filter_distance_dataset_with_llama.py` to build the clean distance
dataset, and shipping it here makes that step reproducible without rerunning
the behavioural audit.
