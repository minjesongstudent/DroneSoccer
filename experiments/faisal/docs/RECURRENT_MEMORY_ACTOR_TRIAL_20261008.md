# Frozen direct recurrent-memory actor trial — 8 October 2026

The previous position actor completed its budget but returned 46 clean repeats and
54 crashes in 100 randomized flights, versus the original 48/52. Seven of its
eight added crashes followed at least two full passages. Its offline position
estimator improved prediction accuracy; that improvement did not improve flight.

This trial tests a distinct representation: use the pretrained GRU's hidden64
directly, rather than compressing its history into three physical estimates.
The hypothesis is that this learned temporal summary supports useful corrections
discarded by the output decoder. This is not established causality. Compared with
the previous actor, representation and residual factorization both change;
different FIT seeds also preclude a matched causal claim between those trials.

## Model and causal inputs

- Frozen original source SHA256: `8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef`.
- Frozen ring-position GRU SHA256: `b387f73399052d73e4aa99cc9b6fe2bb8f13f85f8314654c7cc1e3402c8a31ac`, at `D:/DroneSoccerData/runs/ring-position-estimator-20261008/gru64.pt`.
- New profile `recurrent-memory-residual-v1`: public85 + hidden64 + valid1 =150.
- Hidden64 is dimensionless learned memory, not measured position or simulator truth.
  The existing frozen adapter consumes exactly one public85 per frame. Its hidden
  state after that step is copied; its three predicted outputs are not actor inputs.
  First two frames are masked. Reset only at real episode boundaries.
- Only W[4,64], 256 zero-initialized parameters, is learned. Correction to original
  Gaussian mean is `valid * W @ hidden64`. No bias, public-feature products, GRU
  updates, or source actor/log-standard-deviation updates. Public85 stays unchanged.
- New files and explicit manifests preserve all previous runtimes/checkpoints.

## Preflight and fixed budget

Before fitting, require component/runner tests, CPU/CUDA hidden-state consistency,
75 full deterministic initialization cases bitwise equal to actual source SAC
and preserved baseline trajectories, and a two-episode CUDA software smoke.
The initialization report is `D:/DroneSoccerData/runs/recurrent-memory-initialization-20261008/decision.json`.
It must be complete, non-smoke, and all evidence/source hashes must verify.

One standalone CUDA worker holds the project lock; no simultaneous trainer or
cohort. RTX4070Ti, strictFP32, no TF32/autocast, one intra/inter-op CPU thread,
0.15 allocator fraction. Record memory admission. 3600-second total main budget
(including preparation/evaluation), 600-second smoke budget, STOP checked at each
frame/batch. Do not extend, retry the same failed approach, or sweep variants.

Main FIT seeds exactly11243500..11243739, five rounds of48; smoke11243750/751,
one round of2. Algorithm RNG11603000 after model loading. Prelaunch allocation
scan found no conflict across827 episode logs/30762 rows/128 protocols. Each run
rechecks executed and reserved allocations. All these seeds are within the
existing curriculum guard; no development or acceptance fitting.

Use unchanged alternating approach/full-task resets, level1 randomization,
hanging ring, cage, rewards, scoring and600-step horizon. Every movement is a
learned-policy action. Maximum240 complete FIT episodes/144000 transitions.

Adam1e-5 residual head, one epoch per round, batch256, clipped ratio.9..1.1,
source Gaussian KL penalty and boundary weight.1. Source meanKL<=.01 and max<=.1
on8192 fixed original FIT rows; max4 optimizer backtracks. Guard and feature probes
are reconstructed from full public episodes through the new runtime. Full-horizon
Monte Carlo gamma1 with zero crash/600 tails; separate150→128→128→1 value baseline,
Adam3e-4 and100 batches/round (2 smoke), detached standardized advantages. Retain
existing source action-likelihood math; numerical audit did not justify changing it.

Stop on nonfinite data, STOP/time, integrity failure,100 consecutive rejected
updates or existing25-per-suite retention regression. Preserve all trajectories,
public/features/actions, sources, hashes, heads, optimizer/value/RNG states and
final decisions. No intermediate checkpoint promotion.

## Unchanged outcome gate

Only fully completed final checkpoint5 receives100 cases each aligned, nominal,
randomized. Baseline first/return/repeat/crash:84/84/84/16,80/80/74/26,73/70/48/52.
Require randomized clean repeats>=53 and crashes<=47; no suite loses>2 first or
clean-repeat episodes, or gains>2 crashes. Report return-to-front. Preserve full
body clearance, collision checks,600-step outcomes and all failure accounting.

Require saved-frame feature-use probes on8192 valid FIT and8192 historical heldout
rows: shuffle all64 hidden components jointly within visibility strata; mean action
differences from source and shuffled both>1e-6; shuffled difference>=.5 source
difference; zero memory exactly restores source. These probes issue no flight
actions; actual training/evaluation always uses causal real history. Heldout
rows supply no fitting, normalization, or intermediate selection.

All gates must pass. Software smoke cannot qualify. Passing allows only one later
fresh paired development screen; sustained fixed validation and independent fresh
acceptance still remain. Failure closes this architecture/budget without selecting
a favorable intermediate or extending it. Original85 remains selected. Physical
flight remains unvalidated. No hardware, master edits, publication or messaging.
