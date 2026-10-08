# On-policy result and larger paired development comparison

The five-iteration on-policy trial completed 80 actual learned-policy episodes and 363 accepted actor updates, with no rejected updates or retention stop. The original source remains unchanged.

Final counts per 100 existing development layouts:

| Suite | Model | First full pass | Crashes | Return | Clean repeat |
|---|---|---:|---:|---:|---:|
| Aligned | Original | 84 | 16 | 84 | 84 |
| Aligned | On-policy | 84 | 16 | 84 | 84 |
| Nominal | Original | 80 | 26 | 80 | 74 |
| Nominal | On-policy | 80 | 25 | 80 | 75 |
| Randomised | Original | 73 | 52 | 70 | 48 |
| Randomised | On-policy | 74 | 48 | 72 | 52 |

The randomized gain is four episodes on reused development layouts. It does not establish a general improvement, mission acceptance or flight readiness. The candidate is not promoted and training is not extended merely because the point estimate is better.

Before deciding on further training, compare the frozen original model and frozen `policy_05.zip` on 200 paired layouts per suite: aligned 9610000–9610199, nominal 9620000–9620199, randomised 9630000–9630199. Total 1,200 learned-policy episodes. These are fresh development seeds, explicitly separate from reserved 7M acceptance. Primary metric remains clean repeated episodes; first full passage, return and crashes remain recorded. Preserve paired discordant outcomes and every trajectory. No optimization or automatic promotion occurs during this comparison.

```text
python -m tools.compare_boundary_candidate --out D:/DroneSoccerData/runs/onpolicy-fresh-20261007 --candidate D:/DroneSoccerData/runs/onpolicy-20261007/policy_05.zip --candidate-label onpolicy --episodes 200 --cohort 961
```

The runner freezes protocol, models and source hashes before evaluation. Completion requires `DONE`; intermediate suite results alone are incomplete. Compilation passed. Simulation geometry, randomization, actor inputs and success criteria remain unchanged. No trainer or physical drone is active during this CPU evaluation; no master changes or publication.
