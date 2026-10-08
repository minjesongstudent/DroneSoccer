# Matched-layout outcome trial and longer-history diagnostic

The grouped mission trial completed five rounds, 240 learned-policy episodes and 365 accepted optimizer updates, with no rejected updates. It did not improve overall development performance. The original 85-input policy remains selected; the candidate is preserved but not promoted.

Counts below are per 100 deterministic development episodes. Clean repeat requires at least two full-body passages and survival for the full episode without wall contact.

| Suite | Original first / crash / return / clean repeat | Candidate first / crash / return / clean repeat |
| --- | --- | --- |
| Aligned | 84 / 16 / 84 / 84 | 84 / 16 / 84 / 84 |
| Nominal | 80 / 26 / 80 / 74 | 79 / 26 / 79 / 74 |
| Randomized | 73 / 52 / 70 / 48 | 75 / 53 / 73 / 47 |

Matching four stochastic attempts on each training layout produced mixed outcomes in 3-6 groups per round, so there was a genuine outcome contrast to learn from. Nevertheless, this did not improve clean repeat or crashes. Final source-to-candidate Gaussian KL was mean 0.0005603 and maximum 0.0312932 on the fixed training guard. These are development results, not independent acceptance results.

Evidence: `D:/DroneSoccerData/runs/grouped-mission-20261007`, including the frozen protocol, source snapshots, actual trajectories, per-round models, and `result.json`. Reproduction command is in `GROUPED_MISSION_TRIAL_20261007.md`.

## Longer camera history

A separate CUDA diagnostic compared value prediction from the original 85 features with 117 features retaining camera samples at lags 60, 50, 40 and 32 in addition to all original lags. It reconstructed history from actual saved camera observations and actually issued actions, resetting at each episode boundary. Every reconstructed sample was required to preserve the original 85-feature tail exactly; all checks passed.

The diagnostic used the same saved training trajectories and whole-episode holdout as the earlier value audit. It trained only value networks, not the flight actor. At 1,000 updates:

| Inputs | Fit explained variance | Held-out explained variance | Held-out RMSE |
| --- | --- | --- | --- |
| 85 | 0.1699 | 0.0465 | 25.3984 |
| 117 | 0.2048 | 0.0519 | 25.3038 |

The small held-out difference does not establish a useful generalization improvement or justify another actor architecture trial by itself. It also does not prove that memory is irrelevant: this was one value-network diagnostic of discounted returns, not a recurrent actor comparison or direct state-estimation test.

```text
python -m tools.audit_value_baseline --out D:/DroneSoccerData/runs/long-history-value-audit-20261007 --long-history
```

The output contains input hashes, frozen audit source, whole-episode split, fitted value models and results. No actor observations were supplemented with simulator truth.

## Decision

Do not extend either failed training configuration unchanged. No trainer remains active after these bounded jobs. Before selecting another training intervention, examine whether action effects and failure locations can be distinguished from the available observation/action history; use training trajectories for fitting and separate complete episodes for checking the diagnosis. Any privileged-state comparison must remain diagnostic or critic-only, never an actor input.

Simulation acceptance remains false. Sustained fixed validation and the reserved independent acceptance test are still required. No reserved acceptance seeds, physical drone operations, publishing or group-master changes occurred. The remote master was checked read-only and remains `71c577ec5b4dd7d424c2addbbc3e068eb5987bf6`.
