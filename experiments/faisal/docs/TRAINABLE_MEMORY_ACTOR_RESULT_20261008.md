# Trainable control-memory result - 8 October 2026

The completed final candidate failed the unchanged advancement gate. The original
seenflag-s1-level1 policy remains selected. Simulation acceptance is incomplete;
physical flight is unvalidated. No candidate or intermediate checkpoint is promoted.

## Measured final outcomes

Each suite contains the same 100 preserved starting cases for original and candidate.
Counts are first full passage / return to front / clean repeat / crashes.

| Suite | Original | Final candidate |
|---|---|---|
| Aligned | 84 / 84 / 84 / 16 | 84 / 84 / 84 / 16 |
| Nominal | 80 / 80 / 74 / 26 | 79 / 79 / 71 / 29 |
| Randomized | 73 / 70 / 48 / 52 | 74 / 72 / 49 / 51 |

Randomized clean repeats needed at least 53 and crashes at most 47. The candidate
achieved 49 and 51. Nominal repeat loss and crash increase were both three, exceeding
the unchanged allowance of two. First-pass improvement alone does not qualify.
These reused development cases are not independent acceptance evidence.

The paired saved trajectories show eight randomized repeat rescues and seven losses.
All seven added randomized crashes followed two or more full passages. Nominal had
zero rescues and three losses: two added crashes after repeated passages and one
before the first passage. State divergence means action differences on those paths
do not establish which individual command caused an outcome.

## What this experiment established

The distinct intervention allowed control-loss gradients to train GRU memory over
complete causal public-observation histories. Original actor, action variance and
input normalization remained frozen; the former physical-position decoder was
excluded. Each actual simulator movement came from the learned policy. Memory is a
neural feature, not a measured physical position or privileged simulator input.

All five rounds completed in 1077.531 seconds: 240 distinct FIT episodes, 64,476
transitions, 240 accepted optimizer proposals and zero rejected proposals. This was
not a hard-guard rejection stall. Final source KL mean was 0.0001253124 and maximum
0.0064673424; the guard limits remained 0.01 and 0.1.

Safe checkpoint comparison confirms changes to all four GRU parameter tensors
(28,991 of 28,992 scalar parameters) as well as the correction head. Feature-use
checks passed on FIT and heldout histories. Their source differences were
0.000795486 and 0.000786537; joint-memory shuffle differences were 0.000956721 and
0.000954039. Zero-memory output exactly reproduced the original actor. This supports
actual memory use, but the resulting flight outcomes still failed.

The original pretrained GRU artifact was immutable initialization only. The candidate
GRU was intentionally trainable. The recorded frozen-source checks refer to the
original controller, normalization and initialization files, not frozen candidate
memory. The final probe regenerated current memory from public histories.

Fitted value explained variance ranged from -0.00033 to 0.03886. This is descriptive,
not evidence to repeat the already-closed return-scaling, clock or value audit.
Different stochastic FIT rounds also cannot be presented as paired learning trends.

## Reproducibility and preserved evidence

The frozen [trial plan](TRAINABLE_MEMORY_ACTOR_TRIAL_20261008.md) and historical
[launch checks](TRAINABLE_MEMORY_ACTOR_LAUNCH_20261008.md) describe the software
preflight, seed allocation and fixed stop rules. The run directory is:

`D:/DroneSoccerData/runs/trainable-memory-actor-20261008`

It preserves protocol.json, all 44 bound source files, inputs and normalization
hashes, RNG/optimizer states, all five checkpoints, behavior rollouts, public
histories, complete trajectories, final probes, final evaluation and decision.
Main FIT seeds are 11243760 through 11243999; algorithm seed is 11604000. No acceptance
seeds were used. Historical source snapshots must be used when reproducing this
experiment; reproductions require a new output path and explicit seed-role handling.

One saved-data closure has completed: closure_review.py and closure_review.json.
It checked all 1,289 decision evidence hashes and 2,200 unique files, current and
snapshot sources, all 240 FIT histories/actions/rewards against saved rollouts,
paired final outcomes, feature probes and actual GRU changes. It used safe saved
arrays only, with no additional flight, inference or optimizer. Do not repeat it.
Decision SHA256:

`2d1b4bade84ebc412d9f52f5a71dba7120857a96accf7fb99c582d9386fc2fc2`

## Disposition

Close this architecture and budget. No extension, intermediate selection, relaxed
threshold, fresh development screen or acceptance test is justified for this
candidate. All checkpoints remain preserved for diagnosis. No trainer remains
active; only the read-only replay UI processes were present at closure.

No further training is queued solely because the GPU is idle. A next intervention
must identify a distinct measured mechanism and predeclare a bounded test; this
result alone does not justify another memory/head/credit sweep or looser trust
limits. No unavailable physical measurement is inferred as the cause of this
simulation failure. Existing flight-interface and calibration limitations remain
in the readiness handoff. The heartbeat remains active for justified follow-up.
No hardware operation, group-master change, publication or outside message occurred.
