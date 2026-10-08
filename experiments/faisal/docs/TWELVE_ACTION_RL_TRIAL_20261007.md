# Twelve-command policy trial

The motion-prediction audit reduced held-out error by about 23% when nine older issued commands were added to the existing three. A same-width zero-padding control did not reproduce that improvement. This motivates a bounded twelve-command policy experiment; it does not establish flight improvement.

The experimental actor has 121 inputs: the unchanged nine camera frames, twelve actual issued commands and the original seen-ring flag. No simulator truth, inferred ground-truth velocity or diagnostic-network output enters the actor. Reset history contains zeros for commands that were never issued. Saved original 85-input observations plus every actual action reconstruct the extended history exactly; replay reconstruction occurs before episode-based fitting/holdout splitting.

The retained original model is transferred, not a failed candidate. Added first-layer weights start at zero. Separate original-width matrix multiplication preserves the original floating-point arithmetic for actor and twin critics/targets. Exact CPU/CUDA distribution and critic parity, save/reload, online history and replay/reset tests passed. Four-command behavior remains compatible. The runner requires initial evaluation metrics and every saved action, position and heading to match the original trajectories before any gradient update.

Training uses the existing constrained SAC protocol: original-policy 200,000-transition warmup data, complete-episode holdout, uniform replay, 5,000 frozen-actor critic updates, then at most 5,000 online learned-policy steps. Actor learning rate 1e-6, critic learning rate 3e-4, source entropy coefficient, mean/max source Gaussian KL limits .01/.1, boundary penalty .1 and optimizer rollback remain unchanged. Checkpoints at 100/500/1000/2000/5000 online steps use unchanged first-pass/crash/return/repeat guards; final development uses 100 episodes per suite. Training has a one-hour cap plus bounded evaluation. There is no automatic promotion or resumption.

```text
python -m pytest tests/test_twelve_action_history.py tests/test_action_history_extension.py tests/test_constrained_sac.py -q
python -m tools.run_constrained_clearance --out D:/DroneSoccerData/runs/twelve-action-rl-20261007 --twelve-actions --guard-boundary-weight 0.1
```

Twelve focused tests passed. The separate CUDA smoke run at `D:/DroneSoccerData/runs/twelve-action-rl-smoke-20261007` completed eight critic updates and sixteen accepted actor updates, with zero rejections. Its six initial trajectories exactly matched the original source, and saved models reloaded for evaluation. Smoke outcomes only validate execution and are not performance or acceptance evidence.

The full run preserves frozen source/model hashes, original replay hash, split, actual trajectories, model/replay/RNG checkpoints, evaluation episodes and results. Inspect its result/status files and process state before starting another trainer. The original policy remains selected unless subsequent evidence warrants promotion. No master changes, publishing, physical operations or reserved acceptance seed use are part of this trial.
