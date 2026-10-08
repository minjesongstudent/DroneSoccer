# Differentiable clearance-loss prerequisite passed - 8 October 2026

The loss geometry agrees with the unchanged simulator on all192 saved FIT flights,
64,010 poses. Its gradient through the delayed motion model passed the frozen
numerical check. This supports a bounded learned-policy transfer trial; no policy
has been trained or promoted by these checks.

## Geometry and limits

New `src/shadow_clearance.py` computes ring centreline-to-drone-box distance minus
noodle half-thickness, the hanger's separating-axis margin, and the cage's oriented
box limits. The loss is mean squared deficit below a labelled DESIGN2cm buffer
at the nearest obstacle. This buffer is not a measurement, changed collider,
passage definition or success threshold. Truth geometry is only a loss/diagnostic
input. The actor still receives public85 camera and issued-action history.

Ten CPU/CUDA tests passed: source geometry comparison, smooth-region derivatives,
hanger boundary/disable behavior, finite deep-contact gradients, loss units and
invalid contracts. The initial smoke stopped on the prior Windows source manifest's
path spelling, before gradient evaluation. It remains preserved. Canonicalizing
prior keys with duplicate rejection fixed the reader; six-flight v2 smoke completed
2.656seconds and all46 evidence files verified. No tolerance was changed.

Main run: `D:/DroneSoccerData/runs/shadow-clearance-parity-20261008`.
Completed5.281seconds on CUDA float64, one exclusive project-lock process,
allocator.15 and one CPU thread, within the600-second bound.

- Ring margin maximum error4.44089e-16m against the source _box_distance, below1e-10m.
- Zero ring, hanger or cage classification disagreements on all64,010 poses.
- Ten cage poses were inside the predeclared1e-10m numerical boundary band; none
  disagreed. Ring and hanger had no such cases. This audit band never changes
  simulator contact decisions. Historical wall events are distinct from geometry
  queried at already-clamped saved poses; motion parity verified the former.
- On the first eight deterministic saved episodes, mean clearance loss was
  7.01055727e-6m^2 and action-gradient norm0.0001109947284, finite and nonzero.
- Analytic directional derivative -0.0000190261819710072 versus finite difference
  -0.0000190261820604829. Absolute error8.94756e-14, relative4.70276e-9, passing
  fixed absolute<=1e-8 OR relative<=1e-3 limits.

No env.step calls, actor inference, optimizer or new flight occurred. Saved poses
were assigned only to query source geometric functions. Finite-difference perturbed
command tensors were never flown or counted as learned-policy results.

Nearest-point/axis/contact switches are nonsmooth. Gradients can vanish after deep
ring penetration or an already-clamped wall hit. This surrogate does not capture
wrong-way passage, image visibility changes, policy feedback or altered termination
times. A lower loss cannot establish flight improvement or replace outcome tests.

## Evidence and concrete next step

The [frozen protocol](SHADOW_CLEARANCE_PROTOCOL_20261008.md), input hashes, sources,
every pose comparison, episode losses, direction and action gradients are preserved.
verification.json confirms46 output evidence files,390 input files and40 current/
snapshot sources. Do not repeat this completed check. Result SHA256:

`ca0a80b61401447e44a5b6b47baab0e8fc764caf226c8611c9e04c3a4d930f3c`

Next implement one bounded analytic-gradient actor trial from the retained original
SAC controller. Keep the public85 contract and original feature backbone/action
variance fixed; learn only its mean-output layer through the verified motion and
clearance surrogate, with source-distribution constraints. First establish exact
source initialization/action conversion, then collect fresh learned-policy FIT
rollouts with actual tick and command history. Fixed-history gradients are local
surrogates; refreshing real on-policy histories and evaluating the resulting
neural controller in the unchanged simulator are required.

Freeze the fitting budget, fresh seed allocation, source-KL/retention limits and
final-only go/no-go rule before the first gradient update. Use the existing final
single-pass/return/clean-repeat/crash gate and no intermediate promotion. Save every
model, optimizer, RNG state, source and trajectory. A smoke run must show a real
CUDA neural update and clean checkpoint reload without claiming performance.
This is a different analytic learning mechanism, not an extension of failed
memory/noise variants or proof that removing sampling solves learning.

No trainer is currently active. Original85 remains selected; simulation acceptance
and physical flight remain unvalidated. No hardware, group master, publication or
outside messaging was touched. The heartbeat remains active for this next action.
