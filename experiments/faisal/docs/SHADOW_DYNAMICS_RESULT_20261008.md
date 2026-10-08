# Differentiable motion prerequisite passed - 8 October 2026

A separate PyTorch implementation reproduced the existing simulator's recorded
motion on all192 source-policy flights (64,010 frames), including exact per-episode
wall counts. Its arithmetic derivative also passed a fixed finite-difference check.
This is a prerequisite for a distinct training method, not an improved flight policy.
Original85 remains selected; simulation acceptance and physical flight are incomplete.

## What was implemented and checked

`src/shadow_dynamics.py` reproduces command delays1..4, body-to-world command rotation,
first-order velocity/yaw response, actual variable tick durations, heading wrapping
and oriented-box wall clamping. All commands were previously issued by the learned
source policy. No new simulator step, actor inference, optimizer or hardware action
was used by the main parity check. Resets recover recorded parameters and tick RNG;
initial public observations, position, heading, ring and dynamics must match exactly.

The module takes truth constants for diagnostics or future training losses only.
It does not choose commands, create policy observations, score passages or replace
the simulator. The deployed observation contract stays public85. Camera visibility,
collision termination and full-body passage scoring remain authoritative in the
original simulator.

13 focused CPU/CUDA checks passed: command delay causality, full-episode reset,
future/padding isolation, smooth-region numerical gradients, wall stopping and
invalid contracts. The first software smoke stopped on Windows evidence-path
spelling before tensor integration; it is preserved. Normalizing path separators
with duplicate-name rejection fixed the loader. Corrected six-flight smoke passed
in3.000seconds with44 evidence hashes verified; smoke cannot qualify the prerequisite.

## Full saved-flight result

Run: `D:/DroneSoccerData/runs/shadow-dynamics-parity-20261008`.
One exclusive project-lock CUDA process, float64 physics, allocator.15, one CPU
thread, maximum600seconds. Completed in3.844seconds.

- All192 episodes,64,010 genuine frames included; no favorable subset selection.
- Maximum position error7.32747e-15m, below frozen1e-8m limit.
- Maximum wrapped heading error7.10543e-15rad, below frozen1e-8rad limit.
- Wall-contact counts exactly match every saved episode.
- Fixed directional derivative0.0147649506626433 versus finite difference
  0.0147649506510028. Absolute error1.16405e-11, relative error7.88384e-10;
  unchanged limit absolute<=1e-7 OR relative<=1e-3.

Numerical derivative correctness applies to this fixed-command motion arithmetic.
It does NOT establish smooth gradients through ring/hanger contacts, episode
termination, image visibility, camera ellipse fitting, action quantization or
closed-loop policy feedback. Float64 here mirrors the simulator's physics, not a
new neural-policy precision or sensor claim. Perturbed tensors in finite differences
are arithmetic tests; they are never flown or counted as learned-policy passages.

## Evidence and next action

The [frozen protocol](SHADOW_DYNAMICS_PROTOCOL_20261008.md), source snapshots,
reset-derived diagnostic inputs, learned-command traces, tensor outputs and derivative
direction remain preserved. Verification checked44 output evidence files,387 source
input files and39 current/snapshot sources. It is saved in verification.json; do not
repeat the completed review. Main result SHA256:

`516d16e2faeb8462e1fe6a5b3d07513fe04e0be00e4137271406715a3e493751`

The next concrete step is a separately tested differentiable collision-clearance
loss: derive geometric margins from the existing ring centreline, drone box, hanger
and cage, verify values against original geometry on saved FIT states, and preserve
original progress/retention constraints. Truth may enter a loss, never actor inputs.
A fixed-history derivative is only a surrogate: its proposed neural-policy direction
must be checked with bounded learned-policy flights in the unchanged simulator
before an extended optimization batch is justified. Freeze budgets and go/no-go
criteria before those flights. Do not infer flight improvement from motion parity,
select an intermediate or relax the final pass/return/repeat/crash criteria.

The known noisy critic/score-function signal motivates this separate analytic
training route. It is not a retry of failed memory, credit, reset, parameter-noise
or sampling variants. No learned model has changed yet. No trainer remains active,
no physical flight was performed, and no master branch or remote was changed.
