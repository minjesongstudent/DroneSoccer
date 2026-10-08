# Public-history dynamics identification — 8 October 2026

This is one bounded offline information test, not another actor update or flight qualification.

## Evidence and question

The preserved paired dynamics diagnostic (`D:/DroneSoccerData/runs/dynamics-probe-20261007/results.json`) compared the unchanged learned policy on identical layouts. Normal randomization gave54/100 clean repeats; fixing command delay or response time separately gave61, and fixing both gave66. These diagnostic overrides cannot replace randomized validation. They establish sensitivity to these variables, not that a policy can infer them.

The recent position-estimate actor finished normally but produced46 randomized clean repeats versus48 for the original, with seven additional crashes after two or more passages. That trial is closed. Estimating ring location and estimating how commands turn into motion are distinct questions.

Test whether causal recurrent public history predicts the episode's command delay (steps) and response time tau (seconds) better than equal-budget feedforward history and the FIT mean. No actor is loaded or changed, no simulation movement occurs, and neither true parameter is fed to an actor or an estimator input.

## Data and label verification

Use the existing558 complete original-policy warmup recordings. Whole-episode indices divisible into every fifth heldout (111) and remaining FIT (447) preserve the existing split. The separately recorded GAE240 cohort supplies diagnostics only, never fitting, normalizers or model selection. Historical cohorts have been inspected previously and are not independent acceptance.

Reconstruct each episode's reset with its original environment type: normal HoopEnv for warmup, ApproachCurriculumEnv for GAE. Verify historical hashes for sim, parameters, contract and curriculum, then require exact saved first public85 observation, ring centre/normal and drone half-box. Read the two randomized parameters from this verified reset. Never call step, replay commands, advance to future states or change randomization. Save labels, seeds, source records and input hashes. A mismatch stops the entire trial.

Inputs are exactly recorded pre-action public85 (camera and actually issued command history). Labels and seeds live outside model inputs. Both networks are causal. Evaluate after32 observed transitions, giving the model an opportunity to observe response; earlier outputs are retained but excluded from loss/metrics. Shorter episodes are reported in coverage and excluded from this narrowly defined prediction test, never from flight validation. This limitation matters for early crashes.

## Fixed CUDA experiment

One project-lock-owning worker, two networks sequentially, RTX4070Ti CUDA, strict float32/no TF32, 0.15 allocator fraction and one CPU thread per pool. No additional trainer/cohort may run. Total1,800 seconds main or180 seconds smoke includes preparation and evaluation; STOP each episode/batch. Preserve partial artifacts on failure; no automatic extension or retry after a substantive failed result.

- Feedforward85→128tanh→128tanh→2 versus GRU85→64→2.
- Exactly3,000 Adam0.0003 updates each; gradient norm cap10, no scheduler or intermediate selection. Smoke uses two updates and cannot qualify.
- RNG11601000; common predeclared whole-episode schedule of eight draws with replacement from eligible FIT episodes. Full unroll up to600 frames. Average squared standardized error within each episode after32 frames, then equally across episodes. Do not overweight long survivors solely because of duration.
- FIT-only input normalization. Target normalization and fixed prior use equally weighted FIT episode labels, including short episodes. Original actor checkpoints remain unchanged.
- Save final models, optimizers/RNG, normalizers, schedule, split, reset labels, all final predictions and source/input hashes.

## Final-only go/no-go

Report per-axis RMSE in its actual units, using equally weighted episodes; rounded-delay accuracy is descriptive only. On BOTH historical heldout and separate GAE cohorts, GRU RMSE must improve by at least15% versus feedforward AND20% versus the FIT-mean prior for EACH parameter. No mixing seconds with steps in a reported physical RMSE. Report excluded/available episode counts. Recurrent improvement on only one parameter or cohort fails.

A pass supports only a separately specified integration review. It cannot promote a controller, authorize feeding true delay/tau to an actor, weaken dynamics randomization or establish simulation/physical readiness. A failure closes this fixed architecture/data/budget test; do not extend it or fit to heldout/GAE failures. Source snapshots and reproducible results remain preserved; master, hardware and remote publication remain untouched.
