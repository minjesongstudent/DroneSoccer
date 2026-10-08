# Constrained policy continuation - 7 October 2026

Faisal renewed authorization to keep diagnosing, fixing and training through
simulation completion without routine approval requests. The existing 20-minute
project heartbeat is active again. The prior September pause is superseded for
this work. Physical hardware and publication remain outside this run.

## Why change the update method

The previous full-clearance reward experiment regressed within 1,000 actor
updates; its actor drift was far larger than the control. Simply continuing
that run is not justified. This trial starts again from the unchanged 85-input
source and separates critic adaptation from actor updates.

1. Reuse only the verified 200,000 source-policy warmup transitions, with
   full-clearance rewards. No post-regression data is used.
2. Hold out every fifth episode family (index mod 5 equals 4). No transitions
   from those families enter fitting or the optimization guard.
3. Perform 5,000 critic-only CUDA updates while requiring the actor and entropy
   parameter to remain bitwise unchanged. Preserve checkpoints at 0/1,000/3,000/5,000.
4. Permit up to 5,000 new learned-policy transitions and constrained actor updates.
   Use the original actor as a fixed reference, an explicit KL penalty, and a
   fixed 1,024-observation guard sampled only from FIT data. Reject a proposed
   actor update if mean KL exceeds 0.01 or maximum KL exceeds 0.1 on that guard.
   Restore both actor weights and optimizer moments on rejection. Stop after
   100 consecutive rejections rather than extending stalled optimization.
5. Evaluate at 100/500/1,000/2,000/5,000 new steps using the existing development
   suites and unchanged retention stops. A final 100 episodes per suite records
   the last candidate, including failures. No automatic promotion occurs.

Actor learning rate is 0.000001; critic rate is 0.0003. Entropy coefficient stays
at the source value. Actor/critic gradient clips are 0.5/10, KL penalty weight is
1.0. Critic targets use the usual timeout-aware SAC bootstrap and target updates.
Heldout moving-target Bellman residuals are diagnostics, not independent proof
of value accuracy or better flight. These settings are a bounded hypothesis,
not an established recipe or weakened acceptance criteria.

All new online movements come from the learned policy, with the exact 85-input
camera/action history. The group simulator, full-clearance metric, collision
geometry and randomized conditions remain unchanged. Online reset seeds begin
at 11,100,000, separate from previous warmup families. No development or reserved
acceptance observations enter optimization.

## Verification and artifacts

124 relevant checks passed, including Gaussian KL parity, episode-family
isolation, exact actor/entropy preservation and optimizer-state rollback.
A CUDA smoke run completed eight critic updates and sixteen accepted actor
updates, with saved model reloads and evaluations. Smoke scores are not evidence
of improved reliability.

Full run: `D:/DroneSoccerData/runs/constrained-clearance-20261007`, launched
2026-10-06 16:38:23 UTC. Sources, replay hash, partition indices, FIT guard indices,
checkpoints, optimizers, replay/RNG/environment state and actual online/evaluation
trajectories are preserved there. Adjacent stdout/stderr logs and `status.json`
show progress; `result.json` contains the outcome. One CUDA trainer runs at a
time under the existing OS lock. The run has a one-hour wall cap plus bounded
final evaluation. A `STOP` file in the output directory requests a stop.

Reproduce from the frozen source with the preserved parent replay and a new output:

```text
python -m tools.run_constrained_clearance --out <new-directory>
```

The recurring follow-up must inspect the result and diagnose the next step, not
blindly restart or extend a failed run. Simulation acceptance still requires the
sustained fixed tests and independent fresh-seed acceptance. Physical flight
remains unvalidated. Original models and group master remain unchanged.
