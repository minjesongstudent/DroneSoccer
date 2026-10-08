# Original SAC sampling result - 8 October 2026

The diagnostic completed but did not meet its predeclared threshold for preparing
an exploration change. The original policy is unchanged and remains selected.
No model was fitted, no policy was promoted, and this is not simulation acceptance.

## Matched outcomes

The same 64 fresh FIT layouts were flown once deterministically and twice using
the original SAC actor's normal sampled actions, with independent explicit command
RNG seeds. All 192 flights used learned actions and the same public85 input
contract. Initial observations, pose, ring and randomized dynamics matched within
each layout; subsequent trajectories could diverge.

| Mode | First full passage | Return to front | Clean repeat | Crashes |
|---|---:|---:|---:|---:|
| Deterministic | 42 | 41 | 32 | 32 |
| Sampled replica 0 | 39 | 38 | 23 | 41 |
| Sampled replica 1 | 38 | 38 | 28 | 35 |

Counts are out of64. Replica1 also had one surviving episode without a clean repeat;
repeat failures and crashes must not be treated as exact complements.

Deterministic commands gained6.5 clean-repeat episodes and reduced crashes by6.0
relative to the two-replica average. Both needed to reach8/64 under the frozen
protocol. Benefits were positive in both32-layout halves and both reset strata,
but the overall threshold failed. These are descriptive paired results, not a
statistical significance or independence claim. They do not establish that sampling
has no effect; they do not justify lowering the declared threshold after observing
these outcomes.

Sampled replica0 rescued four deterministic failures and lost13 deterministic
successes. Replica1 rescued four and lost eight. Sampling therefore sometimes
found successful paths, as well as disrupting existing ones. The deterministic
policy itself still crashed in32/64 cases. Removing sampling is not a demonstrated
solution to the repeat-flight problem or a learning algorithm improvement.

## Verification and reproduction

Run: `D:/DroneSoccerData/runs/policy-sampling-comparison-20261008`.
Completed53.047seconds,192 episodes/64,010 frames, one project-lock CUDA process,
float32/noTF32, allocator fraction.15 and one CPU thread. No gradients or optimizer.
All source actor tensors and the checkpoint file were unchanged. Ten focused tests
and the six-flight smoke were completed before the main run; no tests were rerun
merely because the batch finished.

The [frozen protocol](POLICY_SAMPLING_COMPARISON_20261008.md) specifies the outcome
gate and allocation11244720..11244783, with policy RNG11605000+2*layout_index+replica.
The preserved source snapshot includes40 source/test/plan files. Full records,
learned actions, public observations, simulator trajectories, protocol, runtime
metadata, status and evidence hashes remain in the run directory. For reproduction
use the snapshot and recorded environment, a new output directory and explicit
seed-role handling; the production runner rejects overlap with these reservations.

One saved-array closure completed in closure_review.py/json. It verified428 evidence
hashes/470 unique files, all192 episode records and full histories, actual issued
command alignment, passage totals, terminal flags, matched resets, paired outcomes,
source integrity and the final failure decision. It ran no model or simulator.
Do not repeat the closure. Result SHA256:

`706d9c77eb633a2b183fa44a00f1de91e0bc4429bc4cbb240287b5a44776ad68`

## Decision

Close this diagnostic without added layouts, repetitions, reduced-noise fitting or
an exploration-scale sweep. No new training is queued from this result, and earlier
failed memory/credit/reset/value/parameter variants remain closed. A further
experiment needs a distinct measured mechanism and a bounded predeclared decision;
GPU availability alone is insufficient justification.

No trainer or diagnostic process remains active; the process inventory contained
only the read-only replay UI. Heartbeat remains active for justified follow-up.
Simulation acceptance remains incomplete and physical flight unvalidated. This
research result is not an unavailable-lab blocker. No hardware, master-branch edit,
publication or external message occurred.
