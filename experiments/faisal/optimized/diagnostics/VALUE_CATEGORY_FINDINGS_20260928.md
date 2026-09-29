# Value baseline category check

The aggregate held-out improvement did not hold across tasks:

| Held-out category | Episodes | Critic RMSE | Constant RMSE |
|---|---:|---:|---:|
| Early return | 2 | 8.23 | 5.21 |
| Re-approach | 2 | 5.21 | 8.48 |
| Natural full task | 2 | 21.15 | 3.68 |
| Basic pass | 2 | 7.52 | 61.65 |

The large basic-pass improvement dominates the pooled result. The calibrated
critic is not approved for guiding actor updates on this evidence. These
categories contain very few independent flights; do not interpret their error
rates as population estimates. The critic and actor checkpoints are preserved.

## Broader natural-flight dataset

Collected 80 predetermined source-policy flights under
`value-natural-coverage-20260928`, without optimization or outcome filtering:
32 natural nominal, 32 natural delayed, eight centered, eight varied. Each
category holds out every fourth case, giving 60 fit and 20 holdout flights.
Reset seeds 2970000–2970079 and Torch seeds 988000–988079 were declared before
collection. All actual actions came from the preserved source policy. Its
checkpoint hash was unchanged. This is value-model development data, not
independent acceptance or a trained-policy performance improvement.

Every NPZ contains genuine observations, actions, rewards and gamma 0.999
return targets, plus explicit natural-start metadata. Next use only the fit
portion to broaden critic calibration alongside the existing authentic return
fit data. Keep the fresh heldout portion out of optimization, and inspect
errors separately for acquisition, early return, re-approach and basic skills.
Keep the actor frozen until the relevant baseline predictions are supported.
Do not substitute the simulator's hidden state for public-history inputs.
