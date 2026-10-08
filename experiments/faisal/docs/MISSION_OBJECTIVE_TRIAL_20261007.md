# Baseline audit and direct mission-outcome trial

The larger-step on-policy trial completed without a guard stop. Final aligned/nominal/randomised clean repeat was 84/75/50 per 100 versus original 84/74/48; randomized first pass was 72 versus 73 and crashes 50 versus 52. Mixed development differences do not establish improvement. Original remains retained.

A value-only CUDA audit split 80 recorded FIT episodes by whole episode (every fifth held out). Raw return regression, scaled return regression, and scaled regression with elapsed-step clock were compared at 0/100/500/1,000 updates. Heldout explained variance remained approximately 0.04–0.07; more fitting mostly helped FIT rather than heldout data. Do not adopt clock or scaling as a demonstrated solution. Evidence and frozen audit source: `D:/DroneSoccerData/runs/value-baseline-audit-20261007`. No actor was updated by this audit.

Actual episode discounted-return audit: 34 surviving-repeat episodes averaged 40.30 (range 7.35–51.94); 11 repeated-then-crashed episodes averaged 28.70 (8.90–48.59); 34 crashes without repeat averaged -15.57. This cross-layout overlap does not prove a causal reward defect, but confirms shaped return does not rank all episodes in the same order as the required binary mission outcome. Evidence: `onpolicy-step-20261007/reward_outcome_audit.json`.

Next controlled trial changes only the on-policy return target: one for an episode with at least two full-body passages, all 600 steps survived and zero wall contact; zero otherwise. Every action in that episode receives the same undiscounted outcome target for the clipped likelihood-ratio update. A fitted state-value baseline estimates this outcome from public actor observations. This tests direct finite-mission learning, with sparse feedback and potential high variance acknowledged. Normalized advantages and clipping remain practical surrogate updates, not an assertion of an unbiased estimator.

The environment's shaped rewards are still logged, but do not enter this actor target. Success scoring and acceptance thresholds are unchanged. Simulator truth supports the outcome reward only, never the actor input. First-pass, return, crashes and repeat retention guards remain active. No scripted control.

Restart original 85D model, actor rate 1e-5, same FIT reset/algorithm seeds, at most 5×16 learned episodes, same source KL hard limits/rollback, one-hour training cap, guard and final evaluations. Nine focused checks passed, including rejecting shortened, crashing, single-pass and wall-contact episodes; CUDA smoke completed actor updates and reload/evaluation.

```text
python -m tools.run_onpolicy_trial --out D:/DroneSoccerData/runs/mission-objective-20261007 --actor-lr 0.00001 --objective mission
```

Historical models and full evidence retained. No physical operation, publishing, master change or reserved acceptance-seed use.
