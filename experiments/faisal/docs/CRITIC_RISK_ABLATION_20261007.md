# Collision replay result and actor-sampling ablation

The collision-focused replay trial stopped on its unchanged retention guards after 500 actor updates. Randomised guard crashes rose from 12 to 16 of 25 and clean repeat fell from 13 to 9. All 500 proposed updates satisfied the reference KL constraint; therefore a small constrained policy change is not by itself sufficient to preserve mission performance.

Final 100-episode development results:

| Suite | First full passage | Any crash | Return to front | Clean repeat |
|---|---:|---:|---:|---:|
| Aligned | 86 | 14 | 86 | 86 |
| Nominal | 79 | 27 | 79 | 73 |
| Randomised | 69 | 55 | 66 | 45 |

Original randomised results were 73 first passages, 52 crashes, 70 returns and 48 clean repeats. The candidate is rejected, all checkpoints preserved, and original source retained. The heldout moving-target TD residual ended at 4.365 versus 3.873 for uniform replay; target means also changed substantially, so these residuals are not independent value-accuracy evidence.

The previous experiment changed state sampling for both critic and actor. The next ablation isolates the actor component: critic batches remain half uniform and half actual FIT collision approaches, but actor batches are independently uniform over FIT plus new online replay. It starts from the same original source and performs the same frozen-actor critic adaptation. No failed candidate is resumed. Different subsequent random-number consumption is expected from the extra actor batch; initial critic adaptation remains matched.

Hard KL limits, boundary penalty, actual camera-only actor inputs, rewards, terminal handling, simulation, budgets and retention/acceptance criteria remain unchanged. This is a bounded diagnostic ablation, not a claimed improvement. If it also regresses, do not keep extending collision-focused replay without new evidence.

28 focused tests passed, including exact frozen actor/entropy and optimizer rollback with each actor-sampling mode. CUDA smoke completed eight critic and sixteen actor updates, followed by saved-checkpoint evaluation. No physical operations.

```text
python -m tools.run_constrained_clearance --out D:/DroneSoccerData/runs/critic-risk-20261007 --guard-boundary-weight 0.1 --risk-replay --uniform-actor-replay
```

Evidence: `D:/DroneSoccerData/runs/risk-replay-20261007`; smoke: `D:/DroneSoccerData/runs/critic-risk-smoke-20261007`. New run preserves all source hashes, source model, replay indices, checkpoints, optimizer/RNG and actual learned trajectories. Reserved acceptance seeds are unused. Nothing was published or changed on group master.
