# Frozen plan: memory trained for control — 8 October 2026

## Evidence and scope

Three representation trials used frozen pretrained estimators: displacement3,
position3 and direct hidden64. Their actor corrections did not improve randomized
repeat-flight reliability. The latest completed240 FIT episodes and275 accepted
updates with no rejection, but finished46 clean repeats/54 crashes versus48/52.
Six of seven added crashes followed two or more complete passages. Its memory
feature-use test passed. This rules out a frozen correction being silently unused
in that run; it does not prove all recurrent policies are inadequate.

This distinct hypothesis trains the memory representation with the control loss,
rather than freezing a representation learned for position regression. Initialize
again from the registered original85 and initial pretrained GRU, never from a
failed actor checkpoint. Learn a zero4x64 action-mean head and the GRU85→64 weights
jointly. The position decoder is discarded. Its former physical prediction
semantics do not apply to the control memory. New explicit profile:
`trainable-control-memory-residual-v1`, public85 + neural64 + startup-validity1.
Actor inputs remain camera history and actually issued prior commands only.
Source actor, source log standard deviation and FIT input normalization stay frozen.

Original model SHA256 `8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef`;
GRU initialization SHA256 `b387f73399052d73e4aa99cc9b6fe2bb8f13f85f8314654c7cc1e3402c8a31ac`.
Candidate files save both head and learned GRU weights with the distinct manifest,
not the stream's episode state. Earlier files remain unchanged.

## Required integration checks

Full600-step sequence/stream consistency, CPU/CUDA, actual nonzero head, fixed
absolute1e-6 bound; causality, padding/reset isolation, gradients through preceding
inputs and GRU, frozen-source checks, and safe checkpoint roundtrip. The initial
fused CUDA sequence kernel exceeded tolerance (7.56979e-6); preserved under
`D:/DroneSoccerData/runs/trainable-memory-integration-20261008`. Explicit shared
cell equations fixed it; tolerance unchanged. No physical estimates or truth
features were added.19 component/initialization tests passed before flight preflight.

The75-case initialization must exactly reproduce the actual source SAC actions
and complete preserved baseline trajectories. Report:
`D:/DroneSoccerData/runs/trainable-memory-initialization-20261008/decision.json`.
It completed all75 cases in30.031seconds. Before training, verify its full hashes
and current bound sources. Run a two-episode CUDA software smoke with two optimizer
steps so memory, initially behind a zero head, can receive a nonzero gradient.
Smoke must show actual GRU change; it can never qualify flight outcomes.

## One bounded experiment

One standalone project-lock CUDA worker,0.15 allocator fraction, strictFP32,
TF32/autocast off, one intra/inter-op CPU thread. No parallel trainer/cohort.
Main total3600seconds including preparation, gradients and evaluations; smoke600.
STOP/time checks every flight frame and minibatch/trust history group. Full finite
episodes only, maximum600steps. No physical operation or scripted controller.

Five rounds of48 FIT episodes =240, seeds11243760..11243999. Smoke11243752/753.
Algorithm RNG11604000; seed allocation must check both executed logs and reservations.
The accepted allocation scan covered871 logs/33041 rows/136 protocols without
conflict. An initially considered range11243800..11244039 overlapped the earlier
cohort and was rejected by the scanner before any flight/optimizer; it was never
reserved or used. Keep the existing curriculum interval and reset distribution.
Never fit development/acceptance data. Unchanged level1 dynamics, hanging ring,
cage, rewards, scoring and alternating approach/full-task starts.

Four complete episodes per actor minibatch; four actor epochs per round (two
epochs in smoke). This gives48 proposals/round,240 total, comparable in count but
not identical in batching to the prior frozen-head trial. Adam1e-5 trains head and
GRU, full backpropagation from the start of each actual episode, no hidden-state
detachment or artificial subepisode reset. Sequence fitting regenerates current
hidden features for every new likelihood evaluation. Behavior-policy log likelihoods
and detached advantages remain fixed for the current rollout. Ratio clip.9..1.1,
source KL penalty, gradient norm clip.5, up to4 optimizer backtracks preserving
all GRU/head weights and Adam moments on rejection.

Recompute source trust features using the current GRU from full original FIT
public histories for every proposal; never compare using stale cached hidden64.
Keep the original8192 guard row ids and hard source meanKL<=.01/max<=.1. Boundary
penalty weight.1 now uses the complete current minibatch so gradients reach the
GRU; full-history trust validation remains the hard acceptance guard. This change
and sequence minibatching mean the trial is not a single-factor causal estimate
against the preceding run. A changed memory checkpoint is required before any
advancement claim.

Full-horizon dense Monte Carlo gamma1, zero terminal/600 tail. Separate150→128→128→1
value network, Adam3e-4,100 minibatches per round (2 smoke). It is fitted on the
behavior-policy cached features only and supplies detached standardized advantages;
it never changes the memory or actor inputs. No new value scaling/clock, unverified
sensor, likelihood correction, or physics/randomization reduction.

Stop on nonfinite values, integrity failure, STOP/time,100 rejected proposals in
sequence, or unchanged25-per-suite retention regression. Save all sources,
protocol/input hashes, complete trajectories, behavior features/actions/rewards,
head/GRU checkpoints, value/optimizer/RNG states, outcomes and final decisions.
No unbounded continuation, favorable intermediate selection, or automatic retry.

## Unchanged final-only go/no-go

Only completed final checkpoint5 receives100 episodes each aligned/nominal/randomized.
Require randomized >=53 clean repeat episodes and <=47 crashes; every suite loses
at most2 first or clean-repeat episodes and gains at most2 crashes versus the
hash-bound original. First/return/repeat/crash baseline84/84/84/16,
80/80/74/26,73/70/48/52. Preserve full-body clearance and the complete600-step mission.

Final feature-use probes must regenerate the same saved FIT/heldout row ids using
the final GRU and full preceding public history. Store these refreshed probes.
Shuffle64 memory components jointly within visibility strata: both mean action
differences from source and shuffled >1e-6; shuffle difference >=half source
difference; zero memory exactly restores source. Probes issue no simulator actions.
Heldout histories supply no fitting/normalization or intermediate selection.

All completion/integrity/retention/feature-use/memory-learning gates required.
Passing permits only one later fresh paired development screen; sustained fixed
validation and independent acceptance remain required. Failure closes this
architecture/budget. Original85 remains selected, simulation incomplete and
physical flight unvalidated. No master modification, publication or outside message.
