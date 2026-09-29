# Progress and experiment record — 29 September 2026

Work is paused. The latest bounded experiment completed before the pause;
no training or simulator viewer is running. No checkpoint has met simulation
acceptance, and physical flight remains unvalidated.

The target remains one learned policy that fully passes through the hanging
ring, returns around its outside and passes again in the 2.05 m cage. Every
training and evaluation movement in the experiments below came from the learned
policy. Simulator geometry was used for rewards, resets and separate diagnostics,
not silently supplied to the deployable actor observations.

## Latest measured results

The preserved recurrent source and the latest two-update latency-bank trial
both scored 25/25 centered passes and 21/25 varied passes on the fixed suite.
The latest run used 29,770 new environment steps; both accepted updates retained
those basic scores, with 0/25 full-mission successes under both nominal and
200 ms delayed commands. The latest model is an experimental checkpoint, not
a replacement for the preserved source.

| Trial | Accepted updates | Environment steps | Fixed nominal full /25 | Fixed delayed full /25 |
|---|---:|---:|---|---|
| Independent-rollout baseline |2; third rolled back|20,118|0 throughout|0 throughout|
| Return-focused objective |2; third rolled back|20,632|0 throughout|1 at update 1; 0 at update 2|
| Authentic latency-bank update |2; budget complete|29,770|0 throughout|0 throughout|

Centered/varied retention stayed 25/21 for these fixed checks. The first two
runs stopped when the unchanged maximum policy-divergence limit was exceeded;
rejected changes were rolled back. See the exported evaluations for crash,
return-to-front and passage counts rather than treating reward as success.

A separate development comparison of the return-focused source/update1/update2
used 25 new layouts per suite. Nominal full successes were 0/1/0 and delayed
successes 1/0/0. On one nominal episode, update1 made three passages and survived
600 steps. On that same layout with delayed commands, it regressed from the
source's two-pass survival to a collision after one pass. Full-task crash rates
were 96–100%. This is an isolated learned-policy result, not robust completion.
No candidate was promoted using that result.

## Process and findings

1. Reconstructed recurrent memory from authentic public observation histories
   before updates and continuation evaluations. Added reset-isolation, replay
   likelihood and rollback checks. Stored states never replace current-weight
   memory reconstruction during training.
2. Tested critic-only calibration with the actor frozen. Pooled prediction
   improvements hid weak return-category predictions. Broader data, balanced
   category losses and repeated-flight data were tested separately. All models
   were retained, but none was promoted as a reliable return predictor.
3. Measured stochastic target variability with 96 basic-flight repetitions and
   88 authentic return continuations. Identical starts often produced different
   sampled outcomes. These repetitions are correlated training evidence, not
   independent layouts or deterministic acceptance tests.
4. Replaced the unreliable critic baseline in a bounded policy experiment with
   the mean return of other independent trajectories from the same start. Each
   trajectory's own outcome was excluded. Tests checked that independence, and
   live likelihood replay matched exactly before updates. Basic skills remained,
   but reliable repeat missions did not emerge.
5. Located the largest accepted policy drift on a basic-pass history. A second
   trial removed basic-task reinforcement gradients while keeping those histories
   in the fixed policy-change guard and retention validation. It produced the
   isolated three-pass development episode but no sustained improvement.
6. Replayed six nominal/delayed cases with the learned policies. All outcomes,
   passage counts and terminal steps matched saved evaluation records. Failures
   were ring collisions during re-approach; the updated delayed policies collided
   after 1.6 seconds without seeing the ring. This is local diagnostic evidence,
   not proof of a universal failure cause.
7. Collected 128 full-task flights from 64 paired nominal/delayed layouts. Every
   fourth parent layout was held out across both dynamics and all descendants.
   All 42 snapshots passed recurrent-state and next-transition replay checks.
   The training split contained 15 nominal return starts, 14 delayed return
   starts and one delayed re-approach. Earlier return-history collection used
   guided lessons; it must not be described as the same full-task distribution.
8. Audited 120 continuations from the 30 training snapshots. Only two stochastic
   continuations made a new passage; one survived. These are continuation results,
   not natural-start acceptance. The resulting verified latency bank supported
   the final two-update experiment, whose result is reported above.

## Files and reproduction

- Current simulator and learning utilities: `src/` and `tests/`.
- Experiment runners and detailed reports: `diagnostics/`.
- Protocols, evaluation records, source snapshots and recorded arrays:
  `evidence/2026-09-29/`.
- Source/export hashes: `evidence/2026-09-29/export-manifest.json`.
- Selected new checkpoints: `models/september29-manifest.json`.
- Previous history: `LATEST_PROGRESS_20260928.md` and `DEVELOPMENT_PROCESS.md`.
- Mac recorder, simulation checker and motors-off camera/policy preview:
  `../lab-tools/`. Camera preview proposes actions; it does not fly the drone.

From this optimized folder, after installing `requirements-recurrent.txt` in
an environment with CUDA-enabled PyTorch, a portable fixed evaluation is:

```powershell
python evaluate_recurrent.py --model models/latency-bank-actor-20260929-update-2.zip --output latest-fixed-evaluation.json
```

The command refuses to overwrite an existing output file. Use a fresh filename.
This is a new evaluation, not a claim that the repository update reran training.
Checkpoint weight entries are unchanged; local logging metadata was removed
from selected exported archives and both original/export hashes are recorded.
The critic-only checkpoint predicts returns divided by 40 and must not be used
as an ordinary raw-value PPO resume.

Historical runners record their original layout and require path adaptation:
`PROJECT_ROOT` denotes the original workspace; archived drive references denote
local experiment storage. Exported evidence contains source snapshots and raw
arrays, but pickled environment snapshots, full optimizer states and all
historical model files remain preserved locally. The publication is not an
exact-resume bundle for every historical experiment. No data is fabricated to
fill those missing artifacts.

## Remaining requirements

Simulation completion requires three sustained fixed-validation checkpoints,
then the separate fresh 100-layout-per-suite acceptance test. Reserved acceptance
seeds remain unused. The observed isolated successes do not meet that requirement.
Physical flight still requires measured camera calibration, detector/marker
checks, command response and latency verification, and supervised lab validation.
No claim of autonomous physical ring flight is made.
