# Training-only simulator-state value diagnostic

The previous calibration improved value prediction but not reliable action ranking. This diagnostic asks whether additional simulator-state information helps predict the same returns on held-out episodes. It does not assume hidden-state ambiguity is the cause or modify the actor.

The 80 recorded Monte Carlo calibration episodes are reproduced with the same learned actor and seeds. Every pre-action public observation, sampled action, reward and episode boundary must match the saved recording. Only then are pre-action simulator features recorded separately: position, velocity, orientation, ring geometry, cage and camera parameters, randomized dynamics, delayed-command queue, elapsed time and passage-audit state. These values are explicitly simulator truth, not hardware observations. They never enter an actor call.

Fit two identically sized 256-by-256 diagnostic neural networks on CUDA: public 121-input history plus the current action, zero-padded to the same width; and those public inputs plus simulator truth. Use the same fixed soft-return targets, complete-episode fitting/holdout split, normalization fitted only on fitting episodes, two random initializations, batch size 256, learning rate .0003 and 3,000 updates. No checkpoint search or flight-policy training is performed.

```text
python -m pytest tests/test_privileged_value.py -q
python -m tools.audit_privileged_value --out D:/DroneSoccerData/runs/privileged-value-audit-20261007
```

The focused test verifies that reading truth features does not mutate environment state or alter the 121-input public observation contract. Runtime assertions validate the complete reproduced learned trajectories. Input hashes, explicit truth-feature names, episode split, extracted diagnostic features, fitted weights and results are saved. These diagnostic weights are not deployable policies.

Any benefit is limited evidence about prediction on this dataset. It does not prove that a privileged critic will rank actions correctly or produce a better actor; any subsequent action-ranking check needs separate states. Failure to improve also does not prove the public history is fully observable. Timeout targets retain their previously frozen critic tail and trajectory samples are correlated within episodes.

The selected original flight policy remains unchanged. No physical operation, publishing, master modification or acceptance-seed use occurs.

## Completed result

All 80 episodes passed exact public observation, learned action, reward and episode-boundary reproduction checks. The actor remained bitwise unchanged. At the prespecified 3,000 updates:

| Variant | Fitting RMSE, seeds 1 / 2 | Held-out RMSE, seeds 1 / 2 |
| --- | --- | --- |
| Public history, equal-width zero padding | 13.54 / 13.41 | 19.73 / 20.08 |
| Public history plus simulator truth | 0.89 / 0.83 | 21.24 / 21.07 |

The truth-augmented network nearly memorized fitting outcomes and generalized worse on the held-out episodes. This does not justify a privileged-critic actor experiment. It also does not establish that simulator state is useless: each fitting state has only one stochastic trajectory return, and detailed state features can identify a trajectory rather than estimate its expected return. A useful future diagnostic would distinguish prediction of individual noisy outcomes from repeated-continuation expected values, with sufficient independent reset families and no reuse of the existing ranking audit as fitting data. That work has not been run.

No further optimizer steps are warranted on this unchanged dataset. Both runs, fitted diagnostic weights, truth schema, splits and source/input hashes are preserved under `D:/DroneSoccerData/runs/privileged-value-audit-20261007`. No trainer is active; no candidate is promoted and simulation acceptance remains false.
