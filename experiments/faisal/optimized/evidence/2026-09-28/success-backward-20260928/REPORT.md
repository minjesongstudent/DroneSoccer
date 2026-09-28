# Authentic backward-reset trial: stopped at 15,000 steps

The predeclared crash gate rejected this candidate. Do not resume or promote it.

| Check | Source | 10k | 15k |
|---|---:|---:|---:|
| Centered single passes |24/25|24/25|24/25|
| Varied single passes |23/25|23/25|23/25|
| Front-start full missions |0/25|0/25|0/25|
| Front-start crashes |4/25|5/25|25/25|
| General full-task crashes |22/25|21/25|24/25|
| Delayed full-task crashes |25/25|25/25|25/25|
| Return-development new-pass successes |0/50|0/50|0/50|
| Return-development safe returns |0/50|0/50|0/50|
| Return-development crashes |1/50|0/50|25/50|
| FIT deterministic late-return successes |3/4|3/4|0/4|
| FIT stochastic late-return successes |11/32|15/32|6/32|

The curriculum never advanced beyond near_pass. Stochastic repetitions share four selected training layouts; they are not 32 independent layouts. Reserved internal checks and the independent final 7M test remain unused. No simulation acceptance or physical flight validation.

## Actual learned-policy trajectory audit

Eight deterministic continuations (four FIT layouts, source versus15k) are saved in diagnostic_trajectories.npz. Both models stayed frozen during this audit. No scripted movement or edited observations were used.

Source credited a new pass on all four layouts and survived200 steps on three; the fourth hit the ring at161. Candidate credited a pass on three, then exited the cage at local84/85/89; the fourth exited at12 before a new pass. This is primarily a post-pass safety regression on these selected starts, not evidence of improved returns. The actor accepted3602 proposals and rejected1399 after5000 critic-calibration updates.

Actual discounted reward (gamma.995, finite continuation, no entropy term) source -> candidate:
-2810249:6.495 -> -18.680
-2810252:10.551 -> -38.776
-2810296:6.546 -> -19.765
-2810312:-10.147 -> -19.909

These returns decrease on all four starts. This does not support a new reward loophole as the explanation for these trajectories; it does not establish the causal optimizer/critic defect. Longer surviving rollouts are capped at200, and these are conditional FIT states, not generalization results.

## Reproduction and next work

run.py and source_snapshot preserve the preregistered20k-ceiling training protocol, stopped at15k; checkpoints, final replay, all evaluations and source hashes remain intact. diagnose.py reproduces the frozen trajectory audit. Full source tests514passed with two existing Gymnasium bounds warnings.

Do not extend the failed run or modify success criteria. Next read-only work should inspect critic action ranking/targets on these concrete failed return states and actual replay coverage of positive passages plus subsequent crashes. Distinguish inaccurate value estimates from harmful actor extrapolation before choosing a new update method. Keep the preserved initial source as the retention baseline; no new training is running.
