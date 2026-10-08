# Twelve-command result and restricted update trial

The full-network twelve-command trial completed 5,000 accepted updates with no rejections. Aligned/nominal/randomized first-pass counts per 100 were 85/80/74, crashes 15/25/53, returns 85/80/73 and clean repeat 85/75/47. Original clean repeat was 84/74/48 and randomized crashes 52. There is no overall improvement or promotion; the original policy remains selected.

An offline weight-ablation audit used 8,192 fitting and 8,192 held-out recorded inputs. It did not generate flights or modify the stored model. Full trained mean absolute action change from the source was .00593 on fitting inputs. Removing only the new command columns changed trained outputs by .00105; applying those columns to the otherwise original actor changed source outputs by .00106. Thus most output change was associated with changes elsewhere in the network. These nonlinear ablations are diagnostic, not additive causal percentages.

The larger audit also exposed limited coverage of the original 1,024-sample trust check. On the wider fitting sample the final model's maximum source KL was .348; held-out maximum was .528, although its fixed guard maximum was .0502. The .1 bound was enforced on the guard, not globally. Evidence is preserved in `D:/DroneSoccerData/runs/twelve-usage-audit-20261007`, including source/candidate/replay hashes, sample indices and the audit script. No parameter-ablated model was flown.

The next bounded experiment restarts the original actor and trains only its added 36 input columns. All original actor weights and biases are frozen; checks require them to remain bitwise identical at every online checkpoint. A fresh Adam optimizer has no prior momentum or weight decay, and a gradient mask protects the original columns of the shared first-layer weight. The added columns use learning rate 1e-5 instead of the earlier all-actor rate 1e-6. This tests whether adapting the new inputs while preserving the existing computation is more useful than changing the whole actor.

The guard increases from 1,024 to 8,192 fitting samples. KL thresholds .01/.1 and the boundary penalty .1 are unchanged. This increases measured coverage without claiming a global guarantee. Uniform replay, frozen-actor critic calibration, 5,000-step online cap, one-hour training cap, exact initial 75-episode trajectory parity, retention guards and final 100-per-suite development evaluation remain unchanged. No failed checkpoint is continued.

```text
python -m tools.run_constrained_clearance --out D:/DroneSoccerData/runs/twelve-adapter-20261007 --twelve-actions --new-command-columns-only --guard-size 8192 --guard-boundary-weight 0.1
```

Thirteen focused tests passed, including preservation of every original actor parameter through optimizer steps and saved-model reload. The first smoke revealed a standard-loader optimizer-group mismatch before any training; that failed output is preserved. Keeping the standard parameter groups while freezing/masking gradients fixes reload without altering the intended update restriction. A fresh smoke checks the repaired full execution path. Reloaded models are for evaluation; automatic training resume is unsupported because the gradient restriction must be explicitly reapplied.

No physical flight, publishing, master modification, acceptance-seed use or success claim is part of this trial.
