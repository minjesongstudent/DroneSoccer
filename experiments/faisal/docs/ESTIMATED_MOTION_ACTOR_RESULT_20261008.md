# Estimated-motion actor: rejected at the retention check

The bounded trial completed its five training rounds, 240 complete episodes and 68,542 transitions on the RTX 4070 Ti. There were 271 accepted optimizer updates and zero fully rejected updates. The fifth checkpoint failed the unchanged retention check, so the run stopped after 378.3 seconds. The original `seenflag-s1-level1` policy remains selected. This variant is closed; do not extend it, choose an earlier checkpoint, or run additional development seeds to seek a favorable result.

The trial implemented the [prespecified residual-actor protocol](ESTIMATED_MOTION_ACTOR_TRIAL_20261008.md): one frozen public-input recurrent estimator, frozen original actor and action variance, and a learned motion-dependent correction. All simulated movement came from the learned policy. No simulator truth entered its actor observations. The estimator's earlier prediction improvement did not translate into a better controller in this experiment.

## Measured outcome

These are **25-case retention checks**, paired on identical seeds. They are not the deferred final 100-case development test or independent acceptance.

| Suite | Original first / return / clean repeat / crash | Final candidate first / return / clean repeat / crash |
|---|---|---|
| Aligned | 19 / 19 / 19 / 6 | 19 / 19 / 19 / 6 |
| Nominal | 18 / 18 / 17 / 8 | 18 / 18 / 15 / 10 |
| Randomized | 18 / 17 / 13 / 12 | 18 / 17 / 9 / 16 |

The randomized loss of four clean repeats and increase of four crashes exceeded the allowed three-case retention change. Consequently, the final 100-case comparison and final offline feature-use probes were **not run**. The independent fresh-seed test was not reached. There is no flight-performance improvement or acceptance claim.

Saved trajectories show no paired repeat gains. Both additional nominal crashes occurred at the cage net after two passages. The four additional randomized crashes occurred after two or three passages: three ring contacts and one cage-net contact. First-pass counts were unchanged. This locates the measured regression in continued laps; it does not prove one unique cause.

The final correction's latent-mean magnitude was similar on the fixed FIT guard and randomized evaluation trajectories: median 0.00423 versus 0.00429, maximum 0.01990 versus 0.02229. These are Gaussian-mean units, not meters or physical commands. Final source-guard KL was mean 0.0006103 and maximum 0.01934, below the unchanged 0.01/0.1 bounds. No rejected optimizer steps occurred. The evidence does not support blaming hard rollback or a large out-of-range correction for this failure.

A separate review of already saved predictions found motion RMSE of 4.008–4.224 mm overall and 3.903–4.019 mm during invisible intervals across the five fresh training rounds. All 1,032 correction weights changed. These observations provide no evidence of estimator collapse or an optimizer that never updated. Clean episodes generally had higher initial discounted returns than failures, but that association does not establish correct credit for individual actions. The underlying cause of unsuccessful policy improvement remains unresolved.

## Software and evidence checks

Before training, all 75 saved initialization episodes reproduced the original actions, positions, headings and outcomes exactly; startup validity, recurrence reset and checkpoint restoration passed. The two-episode CUDA smoke completed 625 transitions and three actual optimizer steps. Component, initialization and runner checks passed; their overlapping counts should not be added together.

Independent main-run closure verified all 240 unique FIT seeds, stored public-input prefixes and issued actions, rewards and Monte Carlo returns, optimizer counters, all 375 guard episodes, source snapshots and input/evidence hashes. It confirmed unchanged original model and estimator files. The final head equals the preserved stopped-run head. No extra inference, fitting or simulator rollouts were performed by that closure.

The original stopped-run report contains misleading generic reasons: `frozen_unchanged` remained at its initial false value on the early-stop path, and the final screen attempted to open the intentionally skipped evaluation. These are reporting defects, not evidence of changed weights or a reason to ignore the observed retention failure. The original result and decision remain untouched. The independent closure documents the actual state. A future-run reporting repair now records successful frozen checks after a stop, distinguishes completed training from incomplete final validation, skips intentionally absent evaluations, and binds available artifacts. All 52 runner/on-policy checks passed. `reporting_repair.json` and `reporting_repair_source/` preserve the repair separately; no new training or rollout was needed.

## Preserved artifacts and reproduction

Main directory: `D:/DroneSoccerData/runs/estimated-motion-residual-20261008`.

- `protocol.json`, `source/`, model/head and optimizer/value/RNG checkpoints preserve the as-run implementation and experiment.
- `trajectories/`, `estimated_trajectories/`, `rollout_01..05.npz` and `advantages_01..05.npy` preserve actual training samples.
- `guard_01..05/`, `screen_reference/`, `result.json` and `decision.json` preserve the evaluations and original stop decision.
- `closure_review.py/json` independently recount and bind the evidence. Its current-workspace source checks record the audit before the later reporting repair; use the immutable `source/` snapshot when comparing the as-run implementation.
- `trajectory_diagnosis.py/json` compare the saved paired guard trajectories and calculate the linear residual directly from recorded public features. They do not run a policy or environment.
- `learning_signal_review.py/json` reproduce the saved motion-error, weight-change, return and advantage summaries without inference or new rollouts. The report binds its input hashes and states the diagnostic limitations.

Final head SHA256: `b4c4ad3656ca23c1ecd89a0990d960c9bdf5f6ad661f9b14e3512d9c2d5a19e0`.

Independent closure report SHA256: `3c569a6ede3751544efbd22bf8b9b81088ddcc0a5ab41004b8e5228e357a3ca8`.

The historical launch command was:

```text
python -m tools.run_estimated_motion_trial --out D:/DroneSoccerData/runs/estimated-motion-residual-20261008 --initialization-report D:/DroneSoccerData/runs/estimated-motion-initialization-20261008/decision.json
```

The existing directory must never be overwritten. Reproducing the historical computation requires its frozen source and input hashes, not a later edited runner. Do not launch an unchanged repeat as continued development.

No hardware, publication, group-master edits or outside messages occurred. Physical flight and simulation acceptance remain unvalidated. The accurate estimator and tested integration remain preserved, but neither is evidence that this rejected actor is suitable for flight. Further policy work needs a distinct, reviewable hypothesis; another parameter sweep is not justified by this result.

An independent review also checked earlier work before suggesting a replacement. Fresh RecurrentPPO, learned-policy distillation and recurrent RL appear in the [September 28 history](../optimized/LATEST_PROGRESS_20260928.md); authenticated continuation states and independent-peer return baselines appear in the [September 29 history](../optimized/LATEST_PROGRESS.md). The later [decision-block experiment](DECISION_BLOCK_RESULT_20261007.md) did not establish reliable mission discrimination from longer learned prefixes. Renaming these directions would not make them new experiments. No further policy run was warranted or launched at this review. This is an evidence-based stopping decision for this search branch, not a claim that simulation is complete or impossible.
