# All-position distance attenuation report

## Column mapping used
- `model_col`: `model_tag`
- `aux_col`: `aux_pair`
- `distance_col`: `distance_bucket`
- `role_col`: `role`
- `effect_col`: `signed_real`
- `item_col`: `group_id`
- `layer_col`: `layer`
- `random_col`: `signed_rand`
- `aux_default_if_missing`: `be_present`
- `keep_layer0`: `False`

## Main analysis
Uses medium, long, and extra_long distances. The short bucket is ignored by default because short prompts can make distractor/final roles overlap.

Groups tested: 18
Groups where medium > extra_long: 18/18
Groups monotonic medium >= long >= extra_long: 14/18

## Bootstrap evidence
Groups with bootstrap-supported positive medium-to-extra_long drop: 18/18

## Key model patterns
- **Llama-3.2-3B / be_present**: subject/final = medium: 0.589x, long: 0.620x, extra_long: 0.687x; filler/final = medium: 0.171x, long: 0.145x, extra_long: 0.136x.
- **Phi-2 / be_present**: subject/final = medium: 0.449x, long: 0.554x, extra_long: 0.675x; filler/final = medium: 0.143x, long: 0.147x, extra_long: 0.114x.
- **Qwen2.5-3B / be_present**: subject/final = medium: 1.482x, long: 1.780x, extra_long: 1.899x; filler/final = medium: 0.290x, long: 0.201x, extra_long: 0.141x.

## Paper-safe wording

> We next analyze all-position distance steering by bucketing token positions into subject-adjacent, distractor, filler, and final roles. Across medium, long, and extra-long prompts, steering effects generally attenuate with distance. The attenuation is not explained by broad spreading into intervening filler tokens: filler-role effects remain small compared with subject and final roles, and distractor effects remain near zero. Instead, controllability remains concentrated at the subject, immediately adjacent between-subject-distractor tokens, and the final prediction token, with reduced magnitude at longer distances.

## Caveats
- Interpret these as causal controllability effects, not literal movement of a signal between tokens.
- Do not infer attention-routing mechanisms from these steering results alone.
- If this analysis is only run for `is/are`, describe it as an all-position diagnostic on the main present-tense paradigm.
