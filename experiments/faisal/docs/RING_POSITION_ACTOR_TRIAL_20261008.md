# Frozen ring-position residual actor trial — 8 October 2026

## Measured hypothesis and identities

The ring-position estimator reduced invisible-ring position error by 49.9% on historical heldout recordings and 46.6% on the separate GAE recordings. The previous displacement estimate does not represent current ring position. In the descriptive original 64-flight FIT bank, 27/36 crashes and 6/7 crashes after two passages occurred with the ring out of view. This is motivation, not proof of causality.

Test whether an explicitly estimated ring-relative position supports safer repeated passage. This changes the estimated quantity, not the dynamics, rewards, success definitions or controller's public observations. It is not an extension of a failed motion, checkpoint-reset or parameter-signal trial. Historical comparisons are not a matched causal experiment because seeds differ.

- Frozen source: `seenflag-s1-level1.zip`, SHA256 `8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef`.
- Frozen ring-position GRU: `D:/DroneSoccerData/runs/ring-position-estimator-20261008/gru64.pt`, SHA256 `b387f73399052d73e4aa99cc9b6fe2bb8f13f85f8314654c7cc1e3402c8a31ac`.
- Required completed 75-case initialization decision: `D:/DroneSoccerData/runs/ring-position-initialization-20261008/decision.json`, SHA256 `c7329979ea84f3fea4f571896160baa31c9dff3a70f1990b56ba1b1416434d3e`. Runner rechecks evidence and bound sources before training.

## One fixed run

Use `tools/run_ring_position_actor_trial.py`. One standalone CUDA worker owns the existing project lock. No additional trainer/cohort may run. Use strict float32, TF32 disabled, one intra/inter-op CPU thread and 0.15 GPU allocator fraction. Memory admission is recorded before launch. Total budget includes preparation, fitting and evaluation: 3,600 seconds main, 600 seconds software smoke. STOP is checked every frame/batch. No automatic retry, extension or sweep.

Main FIT seeds are exactly 11243200..11243439, five rounds of 48 complete episodes. Smoke uses 11243450..11243451, one round of two episodes. Allocation must first check both executed logs and unexecuted protocol/cohort reservations. Do not reuse development or acceptance seeds for fitting. Algorithm RNG seed is 11599000 after source-model loading; guard/probe selection retains fixed seeds 11520000, 11591000 and 11591001.

Use the existing alternating approach/full-task curriculum, unchanged level1 hanging ring, cage, dense rewards and 600-step horizon. Every movement comes from the learned policy. At most 144,000 main transitions; complete episodes only. No resets from successful tails or artificial actor observations.

Public85 remains unchanged. Append three FIT-standardized estimates of current ring position (body forward/right/up, millimetres relative to current body) and startup validity. The profile is `estimated-ring-position-residual-v1`; never infer semantics from width89. The frozen recurrence consumes each observation exactly once, masks frames0/1 and resets only at a real episode reset. No truth, reward, current action or future observation enters these features.

Learn only zero-initialized W[4,258]: correction = validity * W @ vec(outer([1, public85], estimated_position3)). The source mean network, log standard deviation, GRU and normalizers stay frozen. Head Adam1e-5, one epoch/minibatch256, clipped ratio .9..1.1, source KL penalty and boundary weight .1. Source mean KL <=.01 and maximum <=.1 on 8,192 fixed original FIT frames; four-step optimizer backtracking restores moments on rejection. Reconstruct guard/probe estimates from full original episodes under the new runtime, never reuse old displacement features.

Full-horizon Monte Carlo reward-to-go gamma1, zero tails at crash/step600. Fit an independent 89→128→128→1 value baseline using Adam3e-4, 100 batches/round (two in smoke), complete current FIT returns. Detached, standardized actor advantages. This preserves the prior full-horizon optimizer; no critic-gradient or credit-method sweep.

Stop on nonfinite values, STOP/wall budget, 100 consecutive rejected proposals, broken source/integrity contract or existing 25-per-suite retention regression. Save all source snapshots, exact observations/features/actions/trajectories, seeds, heads, value/optimizer/RNG states, input hashes and decisions. No checkpoint selected after a stop.

## Unchanged final-only decision

Only completed checkpoint5 may receive final100 episodes per aligned/nominal/randomized suite. Recount the hash-bound original baseline: first/return/repeat/crash aligned84/84/84/16; nominal80/80/74/26; randomized73/70/48/52. Require randomized clean repeats >=53 and crashes <=47, and no suite loses more than2 first or clean-repeat episodes or adds more than2 crashes. Return-to-front remains reported. Complete physical clearance, full600 horizon and wall/crash accounting remain unchanged.

Also require the existing feature-use test on 8,192 valid FIT and 8,192 historical heldout frames: estimates shuffled jointly within visibility strata; both source and shuffle mean action differences >1e-6, shuffle difference >= half source difference; zero-estimate actions exactly recover source. These are saved-frame probes, not flights with fake observations. Heldout frames do not supply gradients, normalization or intermediate selection.

Every software/integrity/completion/retention/feature-use gate must pass. Smoke can never qualify. Passing permits one later fresh paired development screen; sustained fixed validation and independent fresh-seed acceptance still remain. Failure closes this quantity/architecture/budget combination without favorable intermediate selection or unchanged extension. Original85 stays selected until full qualification. No physical flight, master edits, remote publication or outside messages.

## Software preflight correction, before any fitting

The first smoke (`ring-position-actor-smoke-20261008`) stopped before its first flight or update: proposed seed11248300 was inside the outer FIT guard but outside ApproachCurriculumEnv's existing [11240000,11245000) interval. Preserve its snapshot and reserved smoke seeds11248300/1. The original proposed main range11248000..11248239 was never flown. The corrected allocations above have no conflict in795 episode logs/29,233 rows and121 protocol/cohort declarations. The curriculum guard, physics and all outcome gates remain unchanged. The second smoke uses a separate output directory and a new actual-reset contract test. This is a configuration correction, not an outcome-based retry.
