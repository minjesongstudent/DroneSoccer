# Bounded analytic mean-head trial, 8 October 2026

The original learned actor achieves 48/100 clean repeats with 52 crashes in the
fixed randomized development suite. Several added crashes in failed variants
occurred after multiple passages. Shadow motion and collision-margin derivatives
now pass their separate recorded-flight checks. Test whether those derivatives
can improve actual flight safety while retaining passage skill. This is a new
analytic gradient mechanism, not a restart of failed memory/value/noise trials.

Freeze this protocol before any experimental gradient. Start from original
`seenflag-s1-level1.zip` (SHA256 `8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef`).
Only SAC `actor.mu.weight` and `actor.mu.bias` learn. Backbone, variance and all
critic weights remain frozen. Public85 camera/issued-command history is unchanged.
All actual movement is deterministic SAC prediction; no scripted movement.

## Fixed budget and objective

- One process with the project OS training lock, CUDA RTX4070Ti, allocator cap25%,
  one CPU thread. FP32 actor without TF32; FP64 shadow loss. Total cap3600seconds,
  including preparation, fitting and evaluation; STOP checked during each frame
  and shadow tick. No second trainer or cohort.
- Five rounds of32 fresh FIT episodes, seeds11244240..11244399. Each batch collects
  four entire actual policy episodes at current weights, then exactly one update;
  160 flights and40 update attempts. No reuse of a trajectory for multiple updates.
  Resets use existing approach curriculum; maximum600frames and current geometry.
- Smoke uses four separate seeds11244790..11244793, one update and2/suite screens;
  software-only, never qualifies. Algorithm RNG11607000; no seed tuning.
- Capture actual public85, issued commands, poses and per-frame dt. A copied reset
  RNG provides dt without consuming the environment RNG, and its final state must
  equal the actual RNG. Separate truth geometry/initial dynamics are loss inputs only.
- Local fixed-history loss: equal mean of four episode losses, each
  `mean(relu(.02-nearest_margin)^2)/.02^2`, plus `.1*mean(source_KL)` and
  `.1*boundary_penalty(source_KL,.01,.1)`. The2cm buffer is a design loss scale;
  no collider, passage definition or success threshold changes.
- Adam learning rate1e-4, gradient norm cap.5. Hard source Gaussian KL mean<=.01,
  maximum<=.1 over8192 original FIT public85 rows. Existing rollback restores both
  weights and Adam moments; at most four half-rate backtracks per update.
  Eight rejected updates, any nonfinite quantity, replay mismatch or retention
  failure stops without qualification. No automatic extension or alternate rates.
- FIT guard is the previously verified original public85 prefix of
  `trainable-memory-actor-20261008/guard.npz`, SHA256
  `87dcd21364d0037386556e09926fb95c92727401a3371b2ba099e2c0915ceefb`.
  Additional memory features are discarded before actor access. No development or
  acceptance observations enter fitting or KL loss.

## Integration and outcomes

Initialization must match source state and per-row SB3 commands exactly (64 FIT
rows), including SB3 float32 action unscale arithmetic. Every collected batch's
actor commands must match actual per-frame commands within2e-6; shadow
poses and wrapped headings must match actual trajectories within1e-4 m/rad.
These numerical tolerances do not modify
collision or passage checks. Whole geometry/dynamics prerequisites remain frozen.
Every saved updated checkpoint must reload identical actor tensors and commands.
Non-mean actor parameters must stay bitwise equal to source throughout.

After each round, original25case/suite retention guards: first-pass drop<=2,
clean-repeat drop<=3, crash rise<=3. CPU deterministic evaluation matches the
preserved baseline arithmetic. Check all frames for STOP/time. Intermediate
checkpoints are never eligible. After all five rounds and all guards complete,
screen only the final checkpoint on unchanged100case/suite development sets:
randomized clean repeats>=53 and crashes<=47, all-suite first/repeat drop<=2 and
crash rise<=2. Report returns as well. Failure closes this variant; do not select
an earlier checkpoint or the best failure. Passing permits only one later fresh
paired development screen. Sustained validation and fresh independent acceptance
still follow before simulation completion. Physical flight remains unvalidated.

Preserve sources, protocol, separate raw trajectories/loss inputs, initial and
round checkpoints, optimizer/RNG state, update metrics, outcomes and hash evidence.
Never alter or publish the group master. No hardware operation.

Limit: the gradient holds observed camera/history fixed, differentiates neither
feedback nor episode termination, and can vanish at deep contact or wall clamps.
It supplies no established passage/progress or wrong-way gradient. Lower training
loss alone is not success: retained and improved actual outcomes are mandatory.

Software integration correction before main training: the first smoke on
11244784..11244787 stopped before any gradient at3.33786e-6 batched action error.
Its files remain preserved. Loss inference now calls the exact single-frame
network arithmetic for each genuine observation, with padding added afterward.
No tolerance was widened; the replacement smoke uses the separate seeds above.
