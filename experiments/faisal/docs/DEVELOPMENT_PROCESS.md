# Development process

This experiment develops one camera-feature-based reinforcement-learning policy for repeated ring passes in a small indoor enclosure. It has demonstrated simulated single passes. Reliable continuous laps, delay robustness and autonomous physical flight remain unproven.

The standalone experiment lives under `experiments/faisal/`. Paths below are relative to that directory. The upstream project is preserved alongside the experiment.

## 1. Establish a reproducible baseline

The initial inspection used repository revision `22c9543b144fadbbc228c329c72565279daaa7a1`. Python 3.12.13 and an isolated environment were configured, and the original test suite passed all 50 tests. Two Gymnasium warnings concerned unbounded observation-space declarations; they did not indicate infinite observations or test failures.

The current observation contract contains eight ring features, nine historical observations and three previous four-component actions: `9 × 8 + 3 × 4 = 84` policy inputs. The saved historical checkpoint expected 71 inputs and could not be evaluated directly in this environment.

A compatible historical simulator from revision `0ff7b7a` provided a limited comparison. The saved policy passed 185 of 200 zero-delay single-pass attempts, but only 72 of 200 with a 200 ms command delay. This was a historical compatibility benchmark, not a reproduction of the checkpoint's unrecorded original training configuration. It did not establish current-environment lap performance. Full results and limitations are in [RESULTS.md](RESULTS.md).

## 2. Match the available physical measurements

The enclosure measures 2.05 × 2.05 × 2.05 m. Photos and installation information informed the remaining provisional parameters in [`src/params.py`](../src/params.py).

| Parameter | Value used | Status |
|---|---|---|
| Enclosure | 2.05 × 2.05 × 2.05 m | Supplied measurement |
| Drone horizontal collision envelope | 0.20 × 0.20 m | Conservative design approximation from photographed spans |
| Drone height | 0.035 m | Approximate photo reading |
| Ring outside diameter | 0.45 m, with ±0.03 m variation per axis | Photo-based nominal value and deformation allowance |
| Foam diameter | 0.065 m | Unconfirmed estimate |
| Clear opening | 0.32 m nominal; 0.29–0.35 m per axis | Derived from outside diameter and estimated foam thickness |
| Installation | Hanging ring with an overhead obstacle | Installation confirmed; hanger dimensions approximate |

An earlier 0.105 m drone length was derived from photographed 0.17 m and 0.20 m spans using Pythagoras. The curved guards did not justify treating those spans as rectangle edges and a diagonal, so the smaller footprint was replaced. The 0.20 m square is still provisional, not a measured guarantee of the full rotating-propeller envelope at every attitude.

Both guard configurations currently share this envelope. Their aerodynamic differences have not been measured. The simulator models translation and yaw without pitch/roll body tilt.

Fresh training followed the geometry change so that stored transitions would not mix incompatible collision outcomes.

## 3. Add local GPU training and a visible simulator

The local trainer requires CUDA for policy inference and gradient updates. Development runs used an NVIDIA GeForce RTX 4070 Ti. Environment stepping and the local viewer's server run on the CPU.

The viewer exposes live trajectories, ring geometry, training progress, checkpoint evaluation, zoom, pacing, stopping and resuming. Training stops at its configured target, saves state and switches to deterministic replay. Continued movement during replay does not mean continued learning.

Checkpoint evaluation uses deterministic policy actions and separate environments. Evaluation transitions never enter the replay buffer. Training pass totals, individual evaluation passes and successful evaluation episodes are recorded separately.

## 4. Compare three training approaches

### A: staged learning from scratch

Centered passes were followed by varied approaches, laps, corner starts and increasing delays. The run used ordinary SAC with random initial exploration and no successful demonstration data. After 1,000,000 steps it remained on the first lesson with zero training passes. Improving reward had not translated into task completion.

### B: correct the single-pass reward target

A trajectory audit found that the easy lesson inherited the lap task's sideways target as soon as the drone center crossed the ring plane. Its pass bonus, however, was withheld until the whole collision box cleared. This could reward sideways movement during the final clearance phase.

The single-pass target was moved to a fixed point beyond the ring, accounting for the maximum yaw-projected horizontal box reach. A modest penalty of 5 was added at the 150-step lesson timeout. Collision geometry, crash penalty, pass requirements and truncation semantics were retained; lap-stage behavior remained unchanged. See [`src/curriculum.py`](../src/curriculum.py).

The correction made the reward consistent with completing a pass, but the separate B run still recorded zero passes after 1,000,000 steps. Correcting a verified reward conflict was not sufficient to teach the task.

### C: successful examples and shorter initial approaches

Two initial lessons were added: aligned starts 0.25–0.35 m and 0.35–0.50 m in front of the ring. The existing centered and varied approaches followed, then the unchanged lap lessons.

A privileged geometry-based controller generated 96 successful simulated single-pass trajectories: 2,798 actual environment transitions across the close, near and original centered lessons. Failed example trajectories were excluded. The student's inputs remained the normal 84 camera-history/action values; ground-truth position was not added to its observation contract.

The actor received 1,500 initial imitation updates. The successful transitions also filled initial replay. An auxiliary imitation loss faded to zero over the first 50,000 online steps, after which updates were ordinary SAC. Evaluation and live policy replay used the learned actor, never the demonstration controller. This warm-start is part of the saved policy's training history. See [`src/guided_curriculum.py`](../src/guided_curriculum.py) and [`src/bootstrap.py`](../src/bootstrap.py).

C learned genuine single passes and reached the first lap lesson, but its final checkpoint was weaker at passing than some earlier checkpoints. Its 5,283 cumulative training passes did not establish a reliable lap policy.

The [archived checkpoint history](results/experiment-history-summary.json) preserves the stage sequence, cumulative pass counters and fixed-suite success counts used for these observations.

## 5. Select one starting checkpoint

Three saved C checkpoints were compared with learning disabled: 520,000, 670,000 and 1,000,000 steps. Each received 100 centered tests, 100 varied-approach tests and 25 full-task tests on an additional seed range beginning at 4,000,000.

Selection prioritized the weaker of centered and varied success rates, with full-task success secondary. The 520,000-step checkpoint scored 94/100 centered and 80/100 varied passes, outperforming the other candidates on this criterion. None completed a successful full-task episode in these 25 new full-task tests.

The selected source is published as [`models/c520k.zip`](../models/c520k.zip). The original archive SHA-256 was `b828515bbe67e4b9164cd30198832e1131043046b2c2c876a2aa326f9df3ab5c`. The published archive removes its local training-log path; weights are unchanged. Original and published hashes are recorded in [the model manifest](../models/manifest.json). [Selection records](results/checkpoint-selection.json) contain the per-episode evidence. Policy weights were not averaged across A, B and C.

## 6. Continue with one unified trainer

The unified run inherits C520k's actor, critics, entropy state and optimizers. It starts fresh replay and fresh online step, update, episode and pass counters. Prior experiment counts are retained as provenance rather than presented as new results.

The first 1,000 new transitions refill replay using the inherited stochastic policy without gradient updates. Subsequent training uses ordinary SAC. No new demonstrations or imitation updates are performed. See [`src/unified_policy.py`](../src/unified_policy.py).

Training starts at the first lap task. Initial layouts are mixed: 60% lap-front, 15% varied, 15% centered, 5% near and 5% close starts. Every episode uses the current task's rewards, dynamics and horizon, including episodes with easier starting layouts. This avoids mixing single-pass terminal rewards with continuing-lap targets for otherwise similar observations.

Only current-layout episodes count toward promotion. Lap promotion requires at least 70 successful episodes in a 100-episode window, where success means surviving the full minute with at least two scored passes. Later stages introduce corner starts and increasing delay. See [`src/unified_curriculum.py`](../src/unified_curriculum.py).

Validation runs at zero new steps and every 10,000 steps. Its five fixed suites cover the current lesson, centered passes, varied passes, full laps and full laps with 200 ms delay. These use seeds beginning at 5,000,000 and fixed reference environments rather than the mixed training wrapper. See [`src/unified_evaluation.py`](../src/unified_evaluation.py).

The dated interim results in [RESULTS.md](RESULTS.md) do not show improved lap reliability. Mixed-start rehearsal is an experiment, not an established improvement.

## Verification and remaining work

Recorded verification for the unified implementation: **144 tests passed**, with the two existing Gymnasium observation-bound warnings. Coverage included curriculum transitions, full-body clearance, unchanged reference behavior, mixture semantics, focus-only promotion, policy-preserving initialization, replay refill without learning and CUDA disk resume. This is a recorded development result, not a claim that every later environment or dependency version has been tested.

The main remaining work is learning the return route without losing passing skill, preserving useful checkpoints and comparing changes on additional validation layouts. Physical integration also needs camera calibration, a real ring detector, measured command delay and response dynamics, and verification of the simulated one-sided front marker. The visible marker assumption is not confirmed by the ring photographs.

Resume files preserve model, optimizer, replay, curriculum and counters. A disk resume starts a new episode and does not reproduce the exact interrupted trajectory. Geometry and camera guesses, the absence of roll/pitch collision modeling and the simplified hanger remain material simulation limits.

## Revision context

The experiment began from `22c9543b144fadbbc228c329c72565279daaa7a1`. Publication was prepared on upstream revision `64a641c`, with the standalone implementation isolated under `experiments/faisal/` to retain newer upstream changes. The historical 71-input benchmark used compatible simulator revision `0ff7b7a`; it is separate from the current implementation.

## Latest optimization work

The [optimized experiment](../optimized/README.md) preserves the subsequent reward corrections, return-practice trials, memory/critic diagnostics and the stopped metric-ring15k trial. See its [development process](../optimized/DEVELOPMENT_PROCESS.md) and [results](../optimized/RESULTS.md) for the current status. Repeated laps remain unsolved.
