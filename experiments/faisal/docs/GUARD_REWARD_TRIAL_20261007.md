# Bounded guard-alignment reward trial

The measured center/guard alignment mismatch motivates one reward-only trial from the preserved original 85-input actor. It is not a continuation of a failed candidate and does not alter physics, observations, commands, collision checks or passage acceptance.

## Reward

At the pre-action pose, translate the yawed guard box along the ring normal to its plane. Let `d` be minimum noodle-centerline-to-box distance and `r` the noodle radius. Add:

```text
-0.2 * clip((along + 0.4) / 0.3, 0, 1)
     * clip(-along / 0.1, 0, 1)
     * clip(1 - d / r, 0, 1)
```

The penalty is continuous, bounded by 0.2 per step, and zero outside the front 40 cm band. It penalizes projected guard contact, not blindness or yaw itself. It is additional shaping, not a policy-invariant potential; possible avoidance/hovering or changed return behavior must be evaluated. Hanger and net collision checks remain native and unchanged.

Original 200,000-transition replay is preserved. A separately saved penalty array was matched against every saved action, actor observation, next observation and original reward before use. All reset geometry matches. 2,880 transitions receive nonzero penalty, minimum -0.1992, mean -0.0005251 over the full replay. The FIT/held-out family split remains unchanged.

Twenty-four focused tests passed, including learned-action online/offline reward parity and bitwise observation/position parity. The short CUDA smoke completed successfully; its two-episode results are software checks, not performance evidence.

## Bound and decision

```text
python -m tools.run_constrained_clearance --out D:/DroneSoccerData/runs/guard-reward-20261007 --guard-alignment --guard-boundary-weight .1 --guard-size 8192
```

One RTX 4070 Ti CUDA run, original actor, 5,000 critic-only updates then at most 5,000 actor updates, one-hour training cap. The 8,192-observation FIT guard addresses the previously measured coverage weakness of the smaller guard. Hard KL limits remain mean 0.01 and maximum 0.1 on those samples. This larger guard is a second difference from the old 1,024-guard boundary trial; it prevents claiming a perfectly isolated reward ablation.

Evaluate fixed development checks at 100/500/1,000/2,000/5,000 steps, with unchanged retention thresholds. Stop for regression, numerical error, repeated rejected updates or cap. Final evaluation covers 100 episodes each in aligned, nominal and randomized suites, scoring first full passage, return, clean repeats and crashes. Do not promote based on reward or critic error. A promising result needs paired fresh development evaluation before sustained validation and reserved acceptance testing. An unpromising result must be closed, not extended unchanged.

Model/source snapshots, replay, penalty reconstruction, trajectories and reports are saved under the run. Automation was resumed at the user's request with an explicit instruction against repetitive unproductive diagnostics. Physical flight remains unvalidated; no hardware or publication is authorized by this run.
