# Results and evaluation limits

All results below are simulated. Successful physical flight has not been demonstrated. Results are grouped by protocol because the historical single-pass simulator, current clean-pass lessons and continuous-lap task have different success definitions.

The latest documented unified checkpoint in this report is **110,000 new online steps**, saved on **27 September 2026, approximately 00:24 Australia/Sydney**. It is an interim snapshot, not a final result or a claim about a later running checkpoint.

## Success definitions

- **Current clean pass:** the collision box fully clears the ring without a collision, within 150 control steps.
- **Full-task success:** the episode survives 600 fixed 0.1-second steps and records at least two forward passes.
- **Individual lap-task pass:** the drone center crosses the ring plane forward inside the opening without a collision at that step. The episode continues; a subsequent crash still makes that episode unsuccessful.
- **Training passes:** cumulative events from stochastic training episodes. These are neither a success percentage nor held-out evaluation results.

Crash rates include ring/hanger hits, wrong-way crossings and geofence exits. Evaluation actions are deterministic. Episode-level records are provided under `docs/results/`, relative to the standalone experiment directory.

## Historical included model

The included checkpoint used 71 inputs, while the current simulator supplies 84. Direct current-environment evaluation was incompatible. A historical simulator at revision `0ff7b7a` provided a compatible single-pass benchmark using 200 identical layout seeds per condition.

| Controller | Command delay | Passes / 200 | Success |
|---|---:|---:|---:|
| Historical hand-coded controller | 0 ms | 168 | 84.0% |
| Historical saved SAC policy | 0 ms | 185 | 92.5% |
| Historical saved SAC policy | 100 ms | 127 | 63.5% |
| Historical saved SAC policy | 200 ms | 72 | 36.0% |
| Historical saved SAC policy | 300 ms | 64 | 32.0% |

This historical setup disabled yaw, had no hanger model and used its own single-pass termination rule. Inference for this early benchmark used the CPU. The checkpoint's exact original training revision and configuration were not recorded, so these figures are not a verified reproduction of its original evaluation. Source: [historical benchmark records](results/included-model-legacy.json).

Separately, the current hand-coded lap controller produced 40 individual passes across 200 episodes but **0/200 full-task successes**, with 189/200 crashes. Historical single-pass percentages must not be compared directly with these lap results. Source: [current-controller baseline](results/included-model-current-baseline.json).

## Completed A, B and C experiments

All three runs reached 1,000,000 online steps. The final fixed evaluations used 25 episodes per suite, with seeds starting at 3,000,000.

| Run | Method | Training passes | Final centered clean passes | Final full-task success | Final delayed full-task success |
|---|---|---:|---:|---:|---:|
| A | Staged SAC from scratch | 0 | 0/25 | 0/25 | 0/25 |
| B | Corrected single-pass shaping and timeout cost | 0 | 0/25 | 0/25 | 0/25 |
| C | Successful examples, imitation warm-start and shorter initial lessons | 5,283 | 13/25 | 0/25 | 0/25 |

A and B remained on their first centered lesson. C reached the first lap lesson between its 140,000- and 150,000-step evaluations, but remained there through the final checkpoint. C's final current-lesson evaluation recorded 25 individual passes and 1/25 successful episodes; its full-task evaluation recorded three individual passes and no successful episodes. Both full-task suites ended in crashes on every final C attempt.

Sources: [A final evaluation](results/experiment-a-final.json), [B final evaluation](results/experiment-b-final.json), [C final evaluation](results/experiment-c-final.json).

The [archived history summary](results/experiment-history-summary.json) records cumulative training passes and every saved validation checkpoint, including stage transitions and delayed-test outcomes.

C occasionally achieved a full-task success on the original fixed layouts: its 520,000- and 670,000-step checkpoints each scored 1/25. Those isolated successes were not sustained at later checkpoints. No C checkpoint recorded a successful full-task episode in the 200 ms delayed suite. Repeated evaluations use the same layouts and must not be treated as independent batches of new test cases.

Reward curves are not directly comparable across A and B because the shaping target and timeout reward changed. Improved reward alone was insufficient evidence of useful flying.

## Selecting the unified starting checkpoint

Three C candidates were independently evaluated using deterministic CUDA inference, with no learning or teacher actions. Each single-pass suite used seeds 4,000,000–4,000,099; full-task tests used 4,000,000–4,000,024.

| C checkpoint | Centered clean passes | Varied clean passes | Weaker single-pass rate | Full-task success | Full-task passes | Full-task crashes |
|---|---:|---:|---:|---:|---:|---:|
| **520,000** | **94/100** | **80/100** | **80%** | 0/25 | 4 | 20/25 |
| 670,000 | 60/100 | 31/100 | 31% | 0/25 | 3 | 24/25 |
| 1,000,000 | 53/100 | 36/100 | 36% | 0/25 | 3 | 25/25 |

The selection rule prioritized the weaker centered/varied score, with full-task success secondary. **C520k was selected for its retained passing skill**, not because reliable laps had been established. All 675 episodes ran on an RTX 4070 Ti in approximately 29.3 seconds. Model training counters and checkpoint hashes were unchanged. Source: [selection details and episode records](results/checkpoint-selection.json).

These seeds are separate from the known C demonstration range beginning at 1,000,000 and fixed evaluation range beginning at 3,000,000. Individual historical training RNG states were not logged, so absolute absence of historical sampling overlap is not formally proven. After selection, these cases are validation data rather than an untouched final test.

The selected artifact is [`models/c520k.zip`](../models/c520k.zip). The original archive SHA-256 was `b828515bbe67e4b9164cd30198832e1131043046b2c2c876a2aa326f9df3ab5c`. Local training-log metadata was removed from the published archive without changing its weights; [the manifest](../models/manifest.json) records both hashes. The other two C candidates and the immutable unified 110k snapshot are also bundled. Replay buffers are excluded, so these archives support inference or a new continuation rather than exact restoration of the original training process.

## Unified run: dated interim snapshot

The unified run begins from C520k with fresh replay and new online counters. It uses ordinary SAC with mixed starting layouts; it generates no new demonstrations or imitation updates. The source policy still inherits C's historical warm-start.

Both snapshots below use the same 25 validation seeds, 5,000,000–5,000,024. The current task remains the first lap lesson. A zero-step snapshot measures the inherited policy before new training.

| Suite | Successful episodes at 0 new steps | Successful episodes at 110,000 new steps | Individual passes at 110,000 | Crashes at 110,000 |
|---|---:|---:|---:|---:|
| Current lap lesson | 2/25 | 0/25 | 7 | 22/25 |
| Centered clean pass | 24/25 | 20/25 | 20 | 4/25 |
| Varied clean pass | 23/25 | 4/25 | 4 | 17/25 |
| Full task | 1/25 | 0/25 | 4 | 21/25 |
| Full task, 200 ms delay | 0/25 | 0/25 | 1 | 24/25 |

Sources: [initial unified evaluation](results/unified-initial.json) and [110,000-step snapshot](results/unified-110000.json).

This snapshot shows deterioration in varied passing and no demonstrated improvement in lap reliability. It does not justify presenting the mixed-layout continuation as a successful improvement. The selected C520k source remains a separate preserved checkpoint.

## Interpretation and next checks

The work established a runnable local training and evaluation pipeline and a policy capable of many simulated clean passes. It also identified a reward conflict, documented that its correction alone was insufficient, and showed that successful examples can establish initial passing skill. Continued training can still lose that skill or fail at the return route.

The next useful comparisons should measure retained centered/varied passing, return-route completion, full-minute survival and delay sensitivity. They should use saved checkpoints, explicit selection rules and additional layouts. Larger pass totals or better reward must not substitute for those outcomes.

Recorded development verification for the unified implementation was **144 passing tests**, with two non-failing warnings about infinite observation-space bounds. Tests verify software behavior, not real-world accuracy or successful physical deployment.

Physical validation remains limited by estimated camera intrinsics, response time, foam thickness, hanger geometry and collision envelope. The simulated front marker needs a matching physical feature or a revised observation design. No claim is made that a policy trained with these approximations is ready for autonomous flight.

## Latest optimization work

The [optimized experiment](../optimized/README.md) preserves the subsequent reward corrections, return-practice trials, memory/critic diagnostics and the stopped metric-ring15k trial. See its [development process](../optimized/DEVELOPMENT_PROCESS.md) and [results](../optimized/RESULTS.md) for the current status. Repeated laps remain unsolved.
