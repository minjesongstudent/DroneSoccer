# Fresh on-policy comparison and update-size test

The full 1,200-episode fresh development comparison completed. Clean-repeat counts original/candidate per 200 were aligned 174/174, nominal 145/144, randomized 92/91. Randomized paired changes were five candidate-only successes and six original-only successes. The earlier four-point improvement did not generalize. Candidate is not promoted; original remains retained.

Read-only drift audit on the same 1,024 FIT guard observations (chosen with seed 11520000, no development data) found mean source-to-candidate Gaussian KL 0.00021005 and maximum 0.0114817, versus hard limits 0.01/0.1. Mean absolute change in deterministic normalized actions was 0.0007345. The 363 accepted updates barely changed behavior and were far from the constraints. This motivates testing update size, not extending the failed candidate unchanged or relaxing criteria.

Next bounded run restarts from original source and the same FIT seed schedule, changing actor learning rate from 1e-6 to 1e-5. All rollout budgets, fitted value network, clipped objective, source KL constraints, boundary penalty, rollback, regression guards, finite-horizon return calculation and final evaluation remain unchanged. At most five iterations of sixteen episodes, one-hour training cap plus bounded evaluations. Actual later trajectories naturally differ as the policy changes.

Added diagnostics record mean/max guard KL, likelihood-ratio clipping fraction, fitted value explained variance and return distribution each iteration. These diagnose update size and baseline quality; they are not success criteria. Eight focused checks and a CUDA smoke with saved reload/evaluation passed. Drift evidence: `D:/DroneSoccerData/runs/onpolicy-20261007/fit_drift_audit.json`; paired comparison: `D:/DroneSoccerData/runs/onpolicy-fresh-20261007`.

```text
python -m tools.run_onpolicy_trial --out D:/DroneSoccerData/runs/onpolicy-step-20261007 --actor-lr 0.00001
```

All previous results, model checkpoints and sources are preserved. No physical hardware, publication, group-master changes, simulator-truth actor inputs or reserved acceptance seeds.
