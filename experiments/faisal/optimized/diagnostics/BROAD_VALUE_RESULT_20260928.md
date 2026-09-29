# Broader critic calibration: completed, not promoted

The bounded 16-epoch CUDA experiment finished successfully. It changed only
critic parameters; every actor parameter remained exactly equal after every
epoch. No new flight capability was demonstrated.

Training used 71 trajectories: 11 authentic return continuations and 60
predeclared natural/basic flights. Evaluation used 24 excluded trajectories:
four old grouped return continuations and 20 new predeclared flights. Parent
flights and their derived reapproach snapshots stayed in the same split.
The old return holdout has already been used for development. None of these
results is the independent acceptance test.

## Final checkpoint results

Errors are in raw reward-return units. Each constant uses only FIT targets.
Metrics are frame weighted within each category, not independent frame trials.

| Held-out category | Episodes | Critic RMSE | Category FIT-mean RMSE | Explained variance |
|---|---:|---:|---:|---:|
| Centered pass | 2 | 37.29 | 39.27 | 0.002 |
| Early return | 2 | 5.78 | 5.48 | -0.001 |
| Natural delayed | 8 | 3.39 | 12.23 | 0.577 |
| Natural nominal | 8 | 7.17 | 12.68 | 0.070 |
| Reapproach | 2 | 8.32 | 7.78 | -0.247 |
| Varied pass | 2 | 32.54 | 30.92 | -0.060 |

Pooled held-out RMSE at epochs 4/8/16 was 10.69/10.35/10.54, against
14.36 for the global FIT mean. All checkpoints are retained; epoch 8 was not
selected using held-out performance. Final training RMSE was 19.14. Early
return and reapproach have weak predictive evidence, while basic-pass errors
remain large. These small samples do not establish population performance.
Do not promote this critic merely because pooled error improved.

## Reproduction and artifacts

Runner: diagnostics/calibrate_recurrent_value_broad.py. From the workspace:

    .\DroneSoccer\.venv\Scripts\python.exe DroneSoccer-Optimized/diagnostics/calibrate_recurrent_value_broad.py

The runner refuses to overwrite its output directory; use a new OUT path for
an intentional repeat. Original output: diagnostics/recurrent-value-broad-20260928.
It contains protocol.json with per-input hashes and source-model SHA256,
source_snapshot, runner.py, model_0/4/8/16.zip, metrics.json, result.json and
critic_optimizer.pt. Torch/ordering seed 989000, Adam 0.001, gradient clipping
1, one mean-squared episode loss per optimizer update. Targets are recorded
gamma 0.999 Monte Carlo returns. No actor optimization, hidden-state actor
input, scripted flight or new physical flight was used.

Critic outputs are normalized by 40. Multiply predictions by 40 before using
raw-return units. These saved models are not drop-in raw-value PPO resumes.
Epoch zero reports the source weights under the new scale, so it is not a
valid comparison to the original unscaled critic.

## Next justified investigation

Before another training run, inspect FIT residuals by outcome and episode,
especially the centered/varied errors and nominal long flights. Check whether
whole-episode mean losses and conflicting terminal outcomes leave a largely
constant prediction, and test sequence-vs-step prediction parity for this
critic path. These are hypotheses, not established causes. Keep held-out
trajectories out of optimization; make any model/loss decision using FIT data.
Do not extend this run unchanged or start actor updates on current evidence.
The preserved flight actor/viewer remains unchanged. Simulation acceptance
and physical flight remain unvalidated.
