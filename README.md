# Causal Layouts of Agreement Control Across Decoder-Only Language Models

**Independent mechanistic interpretability research, 2026**  
**Ongoing work, being extended for ACL Rolling Review / NAACL 2027**

📄 **Current manuscript:** [PDF](paper/agreement_control_causal_layouts.pdf)

This repository contains the code, controlled datasets, processed result CSVs,
and figure-generation scripts for a causal study of subject-verb agreement
control across decoder-only language models.

## Overview

This project asks whether subject-verb agreement is causally controllable from
the same internal locations across different decoder-only language models.

We study three models:

* **Phi-2** (32 layers, partial RoPE)
* **Llama-3.2-3B** (28 layers, RoPE)
* **Qwen2.5-3B** (36 layers, RoPE with long-context scaling)

The models are evaluated on controlled agreement prompts in which grammatical
subject number must determine the auxiliary despite an intervening distractor
noun.

The main result is a cross-model difference in **causal controllability**:

* **Activation patching** shows that agreement control is primarily
  subject-driven rather than distractor-driven in all three models.
* **Layer-specific linear steering** reveals different positional layouts.
  Phi-2 and Llama-3.2-3B show comparable subject- and final-position
  controllability, with subject/final ratios of **0.88×** and **0.99×**.
  Qwen2.5-3B is substantially more source-localized, with a ratio of
  **3.76×**.
* The model ordering persists across held-out items, a `was/were` replication,
  eight anchor templates, all four subject-distractor number conditions,
  and increased dependency distance.
* Layer-resolved analyses show substantially later emergence of strong
  final-position controllability in Qwen2.5-3B than in Phi-2 or
  Llama-3.2-3B.

These results characterize **where subject-number representations are causally
usable for the agreement decision**. They are not intended as evidence of
literal token-to-token information flow or as a complete recovered circuit.

## Intervention design

Subject-number directions are estimated **separately at every transformer
layer**.

Each steering intervention modifies the contextual hidden state at:

1. **one transformer layer**, and
2. **one token position**

at a time.

The layer-averaged values shown in some summary tables are computed only after
the individual layer-specific intervention effects have been measured.

Interventions therefore target contextual hidden states aligned with the
subject, distractor, or prediction position, rather than isolated lexical
representations.

## Headline results

### Activation steering

Main `is/are`, SP condition:

| Model | Subject | Distractor | Final | Subject / Final |
|---|---:|---:|---:|---:|
| Phi-2 | 2.85 | 0.36 | 3.22 | **0.88×** |
| Llama-3.2-3B | 3.11 | 0.28 | 3.15 | **0.99×** |
| Qwen2.5-3B | 4.83 | 0.11 | 1.29 | **3.76×** |

Values are signed effects on the agreement logit margin, averaged over
independently evaluated layers 1+.

Phi-2 and Llama-3.2-3B show strong causal usability of the subject-number
direction at the final prediction position. Qwen2.5-3B instead retains much
stronger controllability at the subject position.

### Activation patching

For the main plural-to-singular patching direction:

| Model | Subject patch | Distractor patch | Subject / Distractor |
|---|---:|---:|---:|
| Phi-2 | 4.10 | 0.453 | **9.07×** |
| Llama-3.2-3B | 3.73 | 0.563 | **6.63×** |
| Qwen2.5-3B | 6.62 | 0.601 | **11.02×** |

Subject-position patching dominates distractor-position patching in every
model.

### Temporal dynamics

For `is/are`:

| Model | Source / Final AUC | Final onset layer |
|---|---:|---:|
| Phi-2 | **1.80×** | 15 |
| Llama-3.2-3B | **1.75×** | 13 |
| Qwen2.5-3B | **5.96×** | 28 |

Qwen2.5-3B maintains much stronger source-side controllability and develops
strong final-position controllability substantially later in the network.

### Robustness

The subject/final ordering persists across:

* held-out items
* `was/were` prompts
* eight anchor templates
* all four SS/SP/PS/PP agreement conditions
* noun-free distance stress
* shuffled-label steering controls
* matched-norm random directions
* opposite-sign steering checks

The magnitude of the effects sometimes changes under stress, but the
cross-model ordering does not reverse in the tested conditions.

## Main figures

* [Figure 1: Behavioural agreement](figures/paper_main/figure1_behavioural.pdf)
* [Figure 2: Subject / distractor / final steering](figures/paper_main/figure2_steering_main.pdf)
* [Figure 3: Localization and robustness](figures/paper_main/figure3_localization_robustness.pdf)

## Experimental setup

The core prompt template is:

```text
The {subject} of the {distractor} in question ___
````

Subject and distractor number are independently varied to produce four
conditions:

| Condition | Subject  | Distractor |
| --------- | -------- | ---------- |
| SS        | singular | singular   |
| SP        | singular | plural     |
| PS        | plural   | singular   |
| PP        | plural   | plural     |

The grammatical subject determines the correct auxiliary.

Behavioural analyses cover four auxiliary paradigms:

```text
is / are
was / were
has / have
does / do
```

The main causal experiments use `is/are`, with `was/were` as an independent
be-auxiliary replication. These pairs are single-token across the evaluated
models, allowing intervention effects to be measured at a common prediction
site without introducing multi-token autoregressive confounds.

## Repository structure

```text
paper/
  agreement_control_causal_layouts.pdf     public manuscript

data/
  agreement/                       core agreement dataset + clean intersections
  anchor_robustness/               anchor-template robustness dataset
  distance_adverbial/              adverbial-filler distance dataset
  distance_adverbial_filtered/     Llama-audited clean subset for distance

scripts/
  01_make_dataset.py                       build agreement dataset, per-model audits
  02_behavioral_analysis.py                behavioural metrics + Figure 1
  03_activation_patching_agreement.py      Tables 1, 7, 8
  04_activation_steering_agreement.py      Table 2 + steering controls
  05_all_position_steering.py              all-position profile, Figure 3a
  06_temporal_dynamics_analysis.py         Table 3, source/final AUC + onset
  07_heldout_steering.py                   50% train / 50% test steering
  08_make_anchor_robustness_dataset.py     8-anchor template dataset
  09_anchor_robustness_steering.py         Tables 13, 14
  10_make_distance_adverbial_dataset.py    4-bucket distance dataset
  11_filter_distance_dataset_with_llama.py Llama-audited clean distance items
  12_distance_steering.py                  Table 4
  13_distance_allpos_steering.py           Table 15
  14_distance_allpos_attenuation_analysis.py
                                             attenuation statistics + ratios

  plotting/
    generate_behavioural_figures.py
    plot_figure2_steering.py
    plot_figure3_localization_robustness.py

results/
  behavioural/
  patching/
    is_are/
    was_were/
  steering/
    is_are/
    was_were/
  all_position_steering/
    is_are/
    was_were/
  temporal/
  heldout_steering/
  anchor_robustness/
  distance_stress/
    behavioural/
    steering/
    all_position/
    attenuation/

figures/
  paper_main/
  behavioural/
  patching/
  steering/
  all_position_steering/
  appendix/
```

## Requirements

Python 3.10+.

Install dependencies with:

```bash
pip install -r requirements.txt
```

`requirements.txt` contains:

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
require access to the corresponding HuggingFace model weights.

GPU execution is recommended.

* Phi-2 is run in `float32` for numerical stability.
* Llama-3.2-3B and Qwen2.5-3B are run in `float16`.

## Reproducing paper figures

The processed result CSVs are already included in `results/`, so reproducing
the main figures does not require rerunning the language models.

Run commands from the repository root.

### Figure 1: behavioural agreement

```bash
python scripts/plotting/generate_behavioural_figures.py
```

### Figure 2: main steering result

```bash
python scripts/plotting/plot_figure2_steering.py
```

### Figure 3: localization and robustness

```bash
python scripts/plotting/plot_figure3_localization_robustness.py
```

Generated figures are written to:

```text
figures/paper_main/
```

## Reproducing analyses from existing results

These analyses operate directly on processed CSVs and do not require GPU
execution.

### Temporal dynamics

```bash
python scripts/06_temporal_dynamics_analysis.py
```

The default configuration reads:

```text
results/all_position_steering/is_are/layerwise/
```

and writes to:

```text
results/temporal/is_are/
```

For the `was/were` replication, use the corresponding configuration paths.

### Distance attenuation

```bash
python scripts/14_distance_allpos_attenuation_analysis.py \
  --input results/distance_stress/all_position/is_are/summary \
  --pattern "allpos_distance_role_summary_*.csv" \
  --output_dir results/distance_stress/attenuation/is_are \
  --preview
```

## Rerunning the model experiments

The model-running scripts are included for end-to-end reproducibility.

Some scripts retain their original scratch-output defaults. The processed CSVs
used by the manuscript have been organized under `results/` and `data/`.

### 1. Build and audit the agreement dataset

```bash
python scripts/01_make_dataset.py
```

This builds the controlled agreement data, runs model-specific behavioural
audits, and creates the clean all-model intersections used by subsequent
experiments.

### 2. Behavioural analysis

```bash
python scripts/02_behavioral_analysis.py
```

Computes SS/SP/PS/PP agreement margins, attraction costs, asymmetry measures,
bootstrap confidence intervals, and behavioural figures.

### 3. Activation patching

```bash
python scripts/03_activation_patching_agreement.py
```

The default configuration runs `is/are`.

For `was/were`, change the configuration to:

```text
VERB_PAIR="be_past"
SINGULAR_V="was"
PLURAL_V="were"
```

and use the corresponding output directory.

### 4. Main activation steering

```bash
python scripts/04_activation_steering_agreement.py
```

This script:

* constructs a separate subject-number direction at every layer
* intervenes independently at subject, distractor, and final positions
* evaluates each intervention layer separately
* runs shuffled-label controls
* runs matched-norm random-direction controls
* evaluates the opposite steering sign

### 5. All-position steering

```bash
python scripts/05_all_position_steering.py
```

This extends the same layer-specific subject-number intervention across every
prompt position and produces the position-by-layer causal profile.

### 6. Held-out steering

```bash
python scripts/07_heldout_steering.py
```

Subject-number directions are estimated on one half of the items and evaluated
on the held-out half.

### 7. Anchor-template robustness

The auxiliary pair is intentionally held fixed to `is/are` so this experiment
isolates template variation.

```bash
python scripts/08_make_anchor_robustness_dataset.py
python scripts/09_anchor_robustness_steering.py
```

The eight anchor phrases are:

```text
in question
under review
on duty
on site
at work
in charge
under observation
on call
```

Auxiliary-pair robustness is evaluated separately using the `was/were`
experiments.

### 8. Distance stress

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

The distance manipulation inserts noun-free adverbial material between the
distractor phrase and the prediction position in order to increase dependency
length without introducing additional noun attractors.

The distance dataset is filtered against Llama's behavioural audit:

```text
results/distance_stress/behavioural/raw/behavior_raw_llama32_3b.csv
```

to retain items on which the model is behaviourally correct before causal
interventions are evaluated.

## Result files by manuscript table and figure

| Manuscript element                              | File                                                                          |
| ----------------------------------------------- | ----------------------------------------------------------------------------- |
| Figure 1, behavioural results                   | `figures/paper_main/figure1_behavioural.pdf`                                  |
| Figure 2, steering profile                      | `figures/paper_main/figure2_steering_main.pdf`                                |
| Figure 3, localization + robustness             | `figures/paper_main/figure3_localization_robustness.pdf`                      |
| Table 1, subject vs distractor patching         | `results/patching/{is_are,was_were}/summary/patching_bootstrap_summary_*.csv` |
| Table 2, subject/distractor/final steering      | `results/steering/{is_are,was_were}/summary/steering_bootstrap_*.csv`         |
| Table 3, temporal dynamics                      | `results/temporal/{is_are,was_were}/temporal_dynamics_summary.csv`            |
| Table 4, distance ratios                        | `results/distance_stress/steering/is_are/ratios/`                             |
| Table 6, behavioural results across auxiliaries | `results/behavioural/summary/behavioral_summary_all_verbs.csv`                |
| Tables 7, 8, patching directions + controls     | `results/patching/{is_are,was_were}/summary/`                                 |
| Table 10, steering controls                     | `results/steering/{is_are,was_were}/summary/`                                 |
| Table 11, alpha sign check                      | `results/steering/{is_are,was_were}/summary/steering_alpha_sweep_*.csv`       |
| Table 12, held-out steering                     | `results/heldout_steering/{is_are,was_were}/robust_split_*_summary.csv`       |
| Tables 13, 14, anchor robustness                | `results/anchor_robustness/is_are/{ratios,summary}/`                          |
| Table 15, distance all-position analysis        | `results/distance_stress/all_position/is_are/summary/`                        |
| Appendix H attenuation statistics               | `results/distance_stress/attenuation/is_are/allpos_distance_bootstrap.csv`    |

Note that the files currently named `steering_alpha_sweep_*` contain the
`alpha = -1` versus `alpha = +1` **sign-orientation check** used in the current
manuscript. They should not be interpreted as a full intervention-strength
magnitude sweep.

## Notes on omitted files

To keep the repository lightweight, the following large intermediate artifacts
are not included:

* raw per-item × per-layer intervention outputs matching `*raw*.csv`
  or `*_raw_*.csv`
* cached hidden-state direction pickles
* HuggingFace model cache files

The included processed CSVs are sufficient to reproduce every manuscript table
and figure.

One raw file is intentionally retained:

```text
results/distance_stress/behavioural/raw/behavior_raw_llama32_3b.csv
```

This file is required by:

```text
11_filter_distance_dataset_with_llama.py
```

to reproduce the clean distance-stress subset without rerunning the behavioural
audit.

## Current extension

The current three-model study is being extended to test whether causal layouts
of agreement control remain stable **within model families across scale** while
differing across families.

The planned extension includes:

* larger Qwen2.5 checkpoints
* a larger Llama checkpoint
* full intervention-strength robustness
* probability-space evaluation of causal effects
* explicit peak-layer and normalized-depth comparisons
* stricter lexical held-out evaluation
* controlled variation in the number of intervening distractors

The goal is to determine whether the current checkpoint-level dissociation
reflects a reproducible family-level pattern.

Results from these ongoing experiments are not included in the current
repository unless explicitly added in future updates.

## Scope and limitations

This study uses controlled fill-in-the-blank agreement prompts because they
allow the subject, distractor, prediction site, donor state, and intervention
location to be defined precisely.

The current experiments cover three decoder-only models in the approximately
2B to 3B parameter range. The observed Phi/Llama versus Qwen difference should
therefore be interpreted as an empirical cross-model dissociation, not yet as a
universal taxonomy of model families.

Architecture, tokenizer, positional encoding, training data, and optimization
are partially confounded across publicly released model families.

The causal interventions characterize **where a learned subject-number
direction is functionally usable**. They do not by themselves identify the
attention heads, MLP components, or path-level circuits responsible for the
observed layouts.

The distance-stress experiment tests positional robustness under additional
noun-free intervening material. It is not intended as evidence for a particular
positional-encoding mechanism.

## Citation

If you use this repository or its results in academic work, please cite the
manuscript linked at the top of this page.

## License

See [LICENSE](LICENSE).
