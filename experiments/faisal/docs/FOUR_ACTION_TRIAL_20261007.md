# Dynamics results and explicit four-command history experiment

The paired sensitivity study completed 400 learned-policy episodes. On the same 100 randomized development layouts, original first/crash/return/repeat counts were 76/46/74/54. Fixing delay at two steps yielded 80/39/80/61; fixing tau at 0.55 yielded 78/39/78/61; fixing both yielded 84/34/84/66. Baseline trajectories reproduced every saved array exactly. These conditions are diagnostics, not improved models or acceptance. Randomization bounds remain unchanged.

The deployable original profile contains only three previous issued commands while randomized delay reaches four steps. This does not prove causality or imply full observability can be achieved with one extra command, but provides a concrete input limitation worth testing without simulator truth.

Experimental profile: 89 inputs, same nine camera lags, four previous actual issued commands in chronological order, then the seen-hoop flag. Original 85D profile and packaged preview remain unchanged. The newly added oldest command is initialized to zero only at episode startup when no command existed. Offline reconstruction uses the preceding real observation's oldest command, resets at actual episode boundaries, and matches online four-command history exactly.

Transfer original weights by inserting four zero input columns after the 72 camera features, shifting the existing action/seen columns and critic action columns. All other actor, critic and target parameters are copied. Tests verify original actor distributions and critic outputs within numerical tolerance even with nonzero added inputs. Full training must first reproduce the original 75-episode guard metrics with the untrained expanded policy; otherwise stop and inspect.

The bounded experiment repeats the previous 117,000-transition successful-FIT imitation objective with this explicit profile. No successful validation data enters training. Maximum 2,000 CUDA updates, same learning rate, source-anchored KL constraints, rollback, retention checks and 100-per-suite final evaluation. Critic unused. This is a hypothesis test; offline selection bias remains and no improvement is promised.

30 focused tests passed including exact offline/online history reconstruction across resets and weight-transfer parity. CUDA smoke exercised 89D training, saved-model reload and learned evaluations. Explicit 89D evaluator records profile and checkpoint checksum; it cannot load an 85D model by dimensional guesswork. No deployment package is replaced.

```text
python -m tools.run_success_imitation --out D:/DroneSoccerData/runs/four-action-imitation-20261007 --four-actions
```

Evidence and original trials preserved on D:. No master change, publication, physical operation or reserved acceptance seed use.

Initial expansion stopped before training: one randomized guard episode changed because enlarged matrix multiplication changed floating-point rounding despite close output parity. Preserved failed run four-action-imitation-20261007. Corrected profile uses an explicit split input layer: original dimensions and weight columns are evaluated together, and the new command contribution is added separately. Exact actor distribution and critic parity tests now pass. Custom policy architecture persists through save/load; rebuilt optimizers reference the replacement layers. Revised CUDA smoke passed. Current run: D:/DroneSoccerData/runs/four-action-split-20261007 (same command flags, different output directory). Initial 75-episode guard equality is still mandatory.
