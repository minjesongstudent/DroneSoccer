# Direct recurrent-memory actor result — 8 October 2026

The bounded trial completed but failed the unchanged advancement gate. The
original85 policy remains selected. No candidate, intermediate checkpoint or
physical-flight capability was promoted.

| Suite (100 episodes) | Original first / return / clean repeat / crash | Final candidate |
|---|---:|---:|
| Aligned |84 /84 /84 /16|84 /84 /84 /16|
| Nominal |80 /80 /74 /26|80 /80 /76 /24|
| Randomized |73 /70 /48 /52|74 /72 /46 /54|

The required randomized result was at least53 clean repeat episodes and at most47
crashes. The candidate missed both. A small nominal improvement and additional
randomized first passages/returns do not establish reliable repeat flight.

## What ran and what failed

All five rounds completed:240 unique FIT episodes,69,751 transitions,275 accepted
updates and zero rejected updates,403.265 seconds total. Source actor, action
variance, GRU and normalization remained frozen. Final source KL was0.000065518
mean and0.002501845 maximum. The run was not stalled by rejected updates; these
numbers do not justify relaxing trust limits or extending training.

The explicit150-input policy used real public85 plus frozen learned hidden64 and
startup validity, with256 trainable correction weights. The physical3 predictions
were not actor inputs. Both saved-frame feature-use probes passed: FIT source/shuffle
mean action differences0.000534847/0.000671490; historical heldout0.000529419/
0.000664415. Zero memory exactly restored the original policy. These probes confirm
that the correction used memory, not that the use improved flight.

Paired trajectory review found two nominal repeat rescues with no losses. The
randomized suite had five rescues and seven losses; six added crashes followed
at least two complete ring passages, one followed a single passage. Later-flight
stability remains the primary observed failure. This is an outcome comparison on
the fixed development layouts, not independent acceptance. Same-time trajectory
action differences also contain divergence in reached states.

The saved FIT value baseline explained only0.95%,1.41%,1.33%,3.92%,1.82% of return
variance across rounds. This is a useful limitation of the fitted feedback, not
proof of the cause of the crashes. The earlier value-baseline audit already tested
raw/scaled returns and a clock, with weak heldout performance (see
[mission-objective plan](MISSION_OBJECTIVE_TRIAL_20261007.md)); do not repeat those
changes on the assumption that these current low values newly justify them.
Different rounds use different layouts and sampled commands, so their survival
counts17/15/21/14/19 of48 are not a paired learning curve.

## Preserved evidence and reproduction

Run: `D:/DroneSoccerData/runs/recurrent-memory-actor-20261008`.
The frozen [plan](RECURRENT_MEMORY_ACTOR_TRIAL_20261008.md) and
[launch record](RECURRENT_MEMORY_ACTOR_LAUNCH_20261008.md) describe preflights,
seeds, budgets and prerequisites. The run contains source snapshots, all heads,
optimizer/value/RNG states, raw observations, hidden features, actions, rewards,
trajectories, fixed evaluations and decision hashes.

One completed saved-data closure (`closure_review.py` / `closure_review.json`)
verified1,286 decision evidence hashes and2,196 unique input files, including43
current/source snapshot bindings. It recounted paired outcomes and feature-use
metrics and checked all240 FIT trajectories against saved features/actions/rewards
and rollout arrays. It performed no inference, simulation or training.

Decision SHA256: `9b15dcbd62b96d4db30ae17a0b5c5226463a7e32bfee1064cf0127db6ad6940c`.
Closure SHA256: `1f175ef90a1a0596d28c4b425f8983adb46ea19e83ff5ba5856e8c89a37376fd`.

Reproduction command from DroneSoccer-Publish, using a fresh output directory and
an explicitly reserved new schedule for any separately justified experiment:
`../DroneSoccer/.venv/Scripts/python.exe -m tools.run_recurrent_memory_actor_trial --out <new-directory> --initialization-report D:/DroneSoccerData/runs/recurrent-memory-initialization-20261008/decision.json`.
The existing runner deliberately rejects reusing its now-consumed FIT allocation;
this command documents the launch, not authorization to rerun the failed variant.
The exact original outputs can be checked with the saved closure source and hashes.

## Decision and next boundary

Close this representation/architecture/budget. Do not extend it, choose an
intermediate, run a fresh development screen for it, or retest unchanged failed
credit/reset/parameter/estimator/value-scaling approaches. This result does not
justify another training run simply because the GPU is idle. Further fitting
requires a distinct measured mechanism and a predeclared go/no-go decision.

No trainer remains active. The existing heartbeat remains active for justified
follow-up; do not repeat this closure. Simulation acceptance is incomplete and
physical flight remains unvalidated. No unavailable lab requirement was invented
to explain this algorithmic result. No master change, push, hardware operation or
outside message occurred. The UI still shows earlier three-arm replays.
