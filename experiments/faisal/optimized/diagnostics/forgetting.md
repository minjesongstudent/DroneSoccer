# Actor forgetting diagnostic

Read-only evaluation of `runs/completion-20260927-135016/checkpoints/`:

- Reference: `model_10000_steps.zip` — 5,000 critic calibration updates and 1 actor update.
- Candidate: `model_20000_steps.zip` — 5,000 critic calibration updates and 10,001 actor updates.

Both models were loaded on **CPU** for this diagnostic. No learning, replay updates, checkpoint edits, or manual control occurred. These are deterministic policy rollouts; CPU/GPU floating-point differences can change outcomes near collision or visibility boundaries.

## Closed-loop input ablation

Each row uses the same 25 seeds, **5,000,000–5,000,024**, separately for centered (`LessonEnv(stage=0)`) and varied (`LessonEnv(stage=1)`) single-pass tests. The `MissionObservation` wrapper supplies 100 inputs: the original 84, 15 camera/command memory features, and one requested-task bit. All ablations alter only the policy input; environment rules and episode termination remain unchanged. Zeroing features is an out-of-distribution diagnostic, not a proposed deployment mode.

| Checkpoint and input | Centered passes | Varied passes |
|---|---:|---:|
| 10k, normal inputs | 24/25 | 23/25 |
| 20k, normal inputs | 1/25 | 0/25 |
| 20k, memory features 84:99 zeroed | 0/25 | 1/25 |
| 20k, requested-task bit 99 zeroed | 0/25 | 3/25 |
| 20k, all added features 84:100 zeroed | 1/25 | 4/25 |

The normal 20k centered outcomes were 23 ring hits, one geofence exit, and one pass. Varied outcomes were 23 ring hits and two geofence exits. Two inspected centered failures retreated and drifted sideways while the ring remained visible, then hit its rim at approximately 32–34 cm lateral displacement from its center. They failed before completing their first passage.

## Action drift on identical observations

The 10k policy generated **1,889 pre-action observations** across the same 50 centered/varied episodes. Those observations were then passed unchanged to both models, with the stated feature ablations applied only to 20k. The reference is always the deterministic 10k action on the original observation. MAE is in normalized action units, where each command ranges from −1 to +1.

| 20k input | Right MAE | Forward MAE | Up MAE | Yaw MAE | Forward MAE while visible | Forward MAE while blind |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 0.266 | 0.565 | 0.354 | 0.322 | 0.888 | 0.126 |
| Memory zeroed | 0.252 | 0.536 | 0.351 | 0.343 | 0.836 | 0.128 |
| Task bit zeroed | 0.263 | 0.528 | 0.348 | 0.306 | 0.830 | 0.117 |
| All added features zeroed | 0.249 | 0.488 | 0.344 | 0.324 | 0.764 | 0.113 |

The actor's first-layer original 84 weight columns changed by a Frobenius norm of **2.805**. At 20k, the new memory columns had norm **1.953** and the task column **0.422**. These norms describe weight change, not causal importance.

Removing added inputs does not recover the passing skill. Together with the large action drift on identical visible-ring observations, this supports broad forgetting in the shared actor rather than a failure dominated solely by the memory features or task bit. It does not establish why the critic induced that drift.

## Remaining camera-only limitations

The original observation history spans 2.4 seconds. With the current nominal camera and centered geometry, the ring disappears approximately 44.8 cm before its plane, while full-body clearance requires approximately 14.25 cm beyond it. Persistent camera/action memory retains the last sighting and nominal movement estimates; these estimates are uncertain dead reckoning and can drift during long blind periods.

Memory does not add unseen wall measurements. Ring-relative observations can be identical in layouts with different safe-side clearances, and the simulator's selected return gate is not an actor input. Reliable generalization to arbitrary ring placements therefore remains unproven; include mirrored and near-wall tests before making that claim.

## Next controlled experiment

A reference-policy constraint on explicitly requested single-pass tasks can limit forgetting while leaving lap-task learning available. Such a constraint should use training-only observations, never these validation seeds, and be described as policy regularization. Constraining every lap state to the source actor would also retain its unsuccessful return behavior. Continue to select checkpoints by clean passages, completed laps, crashes, and timeouts rather than training reward alone.
