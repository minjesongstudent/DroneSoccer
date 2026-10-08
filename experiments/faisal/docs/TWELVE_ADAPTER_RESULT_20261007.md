# Restricted twelve-command result and fresh comparison

The restricted update trial completed 5,000 online steps, with 4,997 accepted optimizer steps and three rolled back. All original actor parameters remained bitwise identical; this was checked during training and independently against the saved final model. Initial actions, positions and headings matched the original across all 75 parity episodes.

Development outcomes per 100 episodes:

| Suite | Original first / crash / return / clean repeat | Restricted candidate first / crash / return / clean repeat |
| --- | --- | --- |
| Aligned | 84 / 16 / 84 / 84 | 84 / 16 / 84 / 84 |
| Nominal | 80 / 26 / 80 / 74 | 79 / 26 / 79 / 74 |
| Randomized | 73 / 52 / 70 / 48 | 75 / 50 / 73 / 50 |

The randomized two-episode improvement is small and uses the repeatedly inspected development set. It is not sufficient evidence for promotion. The original policy remains selected. The final fixed 8,192-input guard had mean KL .004017 and maximum .05356; this is a sampled guard, not a global bound.

A paired fresh development comparison is running on 600 new layouts: 200 each at aligned seeds 9710000-9710199, nominal 9720000-9720199 and randomized 9730000-9730199. Both the original 85-input policy and the explicit 121-input candidate run on every layout, for 1,200 total episodes. The model files, source snapshots, profiles and hashes are frozen before evaluation. Primary outcome is clean repeated episodes; first full passage, crash, return-to-front and paired disagreements are also recorded. These are new development seeds, not the reserved 7M independent acceptance seeds.

```text
python -m tools.compare_boundary_candidate --out D:/DroneSoccerData/runs/twelve-adapter-fresh-20261007 --candidate D:/DroneSoccerData/runs/twelve-adapter-20261007/online_005000/model.zip --candidate-label adapter --candidate-profile twelve121 --episodes 200 --cohort 971
```

Inspect `results.json`, `DONE`, adjacent logs and process state in the output directory before taking the next action. No policy training is active during this comparison. No automatic promotion, physical operation, publication or group-master change occurs. Sustained fixed qualification followed by the reserved independent test remains necessary for simulation completion.
