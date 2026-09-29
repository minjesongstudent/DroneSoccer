# Authentic snapshot rollout integration

Implemented a standalone single-trajectory segment collector and sequence
evaluator in `src/recurrent_snapshot_rollout.py`. It deliberately avoids SB3
autoreset handling: each segment belongs to exactly one flight. Resetting to
another saved flight requires a separate collection call with its own prefix.

The current-weight actor and critic memory is reconstructed from actual public
inputs before collection and before each optimizer evaluation. Segment inputs
are appended to the authentic prefix before continuing after an update. Prefix
replay is detached truncated backpropagation, not a differentiable full-episode
training method. No generated replay actions are sent to the simulator.

Raw Gaussian actions and their likelihoods are stored for PPO; only bounded
actions are issued to the environment, matching SB3's action convention.
Collector length cuts bootstrap from the next observation and critic memory.
The finite 600-step mission endpoint is treated as terminal, with zero bootstrap.
This endpoint convention must remain explicit in future experiment protocols;
it differs from SB3's generic time-limit bootstrapping.

## CUDA smoke result

Command: project Python running
`DroneSoccer-Optimized/diagnostics/test_snapshot_rollout.py`.

Seed 982001, first hash-verified training snapshot, eight collected steps, two
in-memory Adam updates at 1e-5, then 92 continuation steps. Pre-update value and
log-probability agreement passed. Gradients were finite. Actor and critic memory
both changed after updates; actual history remained intact. Numerical GAE tests
passed for both mission termination and collector truncation. Preserved source
checkpoint SHA256 was unchanged. No candidate model was saved.

The first test caught cuDNN's requirement for training-mode recurrent forward
passes during backward. The sequence evaluator now enables training mode before
the forward, while prefix replay restores that mode after inference.

The continuation ended outside the cage. This is an integration test, not a
performance result or successful mission. No full training run was launched.

## Next experiment requirements

Use this collector with an explicitly bounded protocol, reward/value scaling
and single-pass retention checks. Rebuild prefixes at every gradient step;
never cache burn-in states across updates. Preserve source snapshots and actual
rollout observations/actions. Compare natural-start fixed validation with the
preserved model before any promotion. The reserved independent acceptance set
remains untouched. Do not interpret saved historical passes as new successes.
