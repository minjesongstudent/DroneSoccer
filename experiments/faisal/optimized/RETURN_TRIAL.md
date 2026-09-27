# Return-skill trial

Status: **started** as `return-route-40k-20260927` through an attached execution session after the background launcher was blocked. The live API confirmed the RTX 4070 Ti, target 40,000, and source actor at 100,000 steps. Live status is at http://127.0.0.1:8777/; evaluations and checkpoints are saved in `runs/return-route-40k-20260927`. The user explicitly authorized this bounded trial. Startup is not evidence of improved policy performance.

## Evidence

The protected run stopped at 100,000 new steps for stagnation. Final validation: centered 24/25, varied 23/25, front-lap completion 0/25, full task 0/25, delayed full task 0/25. It reached a return side waypoint in 11/25 front-start episodes but never reached the front waypoint. Rotation or survival after one pass is not completion.

The previous reward penalized reaching each return waypoint by about five points and could insist on the opposite feasible side from the drone's actual route. The corrected potential includes all remaining route legs, accepts either feasible side until one is reached, and preserves the original passage/collision rules. Tests cover both return routes, all milestones, route resets, and repeated zero-pass loops.

## Controlled next run

Use the protected run's final actor, including its partial return skill. Preserve the separately stored original reference actor. Start with fresh critics, optimizers, replay and counters because the reward changed. Keep the observation contract and learning settings unchanged. The actor continues to receive camera features and command-derived memory only; every training action comes from it.

Run for **40,000 additional steps**, with fixed validation every 10,000. This bounded trial tests the corrected reward before committing to a longer run. A regression stop can end it earlier. Inspect first passages, side arrivals, front arrivals, two-passage episodes and complete 60-second missions separately. Do not treat scripted environment tests or rising training reward as policy success.

```powershell
..\DroneSoccer\.venv\Scripts\python.exe -m src.live --steps 40000 --name return-route-trial --port 8777 --source-checkpoint runs/completion-protected-20260927-140307/model.zip --source-config runs/completion-protected-20260927-140307/config.json
```

Use an unused run name and port. CUDA is required; there is no CPU training fallback. Existing experiments remain saved. Source code and tests are automatically snapshotted with each new run.

## Decision after 20k and 40k

- Passing must remain near the starting 24/25 centered and 23/25 varied scores; existing regression gates remain active.
- Look for actual progress from the side back to the front and then a second clean passage, alongside collision rates.
- If these skills still stall, inspect the new trajectories before extending. Candidate changes include shorter return practice starting from complete states captured during actual policy flights, or a better observation memory. A camera-blind reset with empty history is not an authentic continuation.
- Full mission success still requires at least two fully cleared passes and surviving all 600 simulation steps. The existing fresh 100-layout acceptance test is required after sustained validation success. Physical flight is not certified by either test.

## Verification

250 tests passed, with two pre-existing non-failing Gymnasium bounds warnings. CUDA transfer checks found zero parameter difference for both the transferred current actor and original frozen reference, and empty new replay/optimizer state. Independent route checks found no repeatable zero-pass reward farming. These are implementation checks, not proof the learned policy can finish.

The real-drone interface, camera calibration/front marker, telemetry availability, flight response, and physical clearances remain lab checks in `LAB_CHECK.md`.
