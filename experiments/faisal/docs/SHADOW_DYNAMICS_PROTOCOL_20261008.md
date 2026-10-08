# Differentiable motion parity prerequisite - 8 October 2026

The prior policy updates and continuation-value studies showed noisy learning
signals, weak value generalization and additional late crashes despite completed
updates. Their failed actor/estimator/credit/noise variants remain closed. This
prerequisite investigates a different route: differentiating the known simulator
motion equations directly for a future training loss. No learning intervention is
authorized by this numerical check alone.

Implement the existing delayed first-order body motion, yaw response, tick duration
and oriented-box wall clamp in a separate float64 PyTorch module. This module never
chooses commands, provides actor observations, scores passages or replaces the
original environment. It accepts recorded learned commands and simulator constants
for diagnostics/loss calculations only. No truth enters a deployed actor.

The finite prerequisite uses all192 previously saved original85 sampling flights
(64,010 frames) without new env.step calls or actor inference. Reconstruct constants
and tick RNG by an exact recorded reset, require matching initial public85/pose/ring/
dynamics, then integrate the already-issued commands. Preserve actual episode ends.
The original simulator's camera visibility, ring/hanger collision, full-body passage
and terminal events remain authoritative and are not differentiated in this test.

Fixed gate: maximum position error <=1e-8 metres, wrapped heading error <=1e-8
radians, exact per-episode wall counts across every recorded frame. One fixed
directional finite-difference check on shadow position/heading arithmetic must
have absolute error <=1e-7 OR relative error <=1e-3, epsilon1e-5, direction RNG11606000.
Directions are zero on padded frames and command components within.1 of saturation.
This checks piecewise arithmetic; it is not proof of valid gradients across camera,
collision, clipping or policy-feedback switches. No surrogate objective is silently
substituted for the task's actual pass/crash criteria.

CPU/CUDA unit checks cover delay1..4, future/padding isolation, smooth-region
finite differences, wall stopping and invalid contracts. Smoke covers the first
six saved flights, identical numerical gate but cannot qualify this prerequisite.
Main covers every192 record; no subset/checkpoint selection, tolerance relaxation
or extra-flight sampling on failure. Failure closes this implementation pending
an identified code discrepancy, not more fitting. Passing permits designing and
testing a separate loss/actor experiment only, with unchanged final outcome gates.

One project-lock CUDA process, allocator.15, one CPU thread, double-precision
physics, noTF32, total600seconds including loading and derivative calculations;
STOP at each episode and integration tick. No optimizer or training state; preserve
input/source hashes, source snapshot, truth-labeled diagnostic tensors, errors,
gradient direction and outputs. No development/acceptance seeds or hardware.
