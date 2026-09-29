# Critic-only calibration result

Completed 16 bounded CUDA epochs on the first acquisition-mixture batch.
Nineteen trajectories were used for fitting; eight were held out. Source
flight families 0 and 3, including their derived re-approach snapshots, were
kept together in the holdout. No fixed validation or acceptance data was used.

| Epoch | Fit RMSE | Held-out RMSE | Held-out explained variance |
|---|---:|---:|---:|
| 4 | 10.72 | 14.52 | 0.394 |
| 8 | 9.73 | 14.04 | 0.388 |
| 16 | 7.96 | 10.00 | 0.665 |

The constant predictor based on the fit-set mean had held-out RMSE 17.04.
Final error is about 41% lower. This small, grouped holdout supports further
testing of the value baseline; it is not a general flight-performance result.
Epoch-zero metrics used the newly specified output scale and must not be
presented as an audit of the original critic's raw-unit predictions.

Only the independent critic LSTM, value MLP and value head were optimized.
All actor parameters, actor LSTM and action standard deviations remained
exactly equal after every epoch. The final result confirms zero actor updates.
No simulator movements, physical commands or model promotion occurred.

Artifacts: `recurrent-value-calibration-20260928`, including checkpoints,
group assignments, source snapshot, runner, metrics and critic optimizer state.
The critic predicts returns divided by 40: **multiply its output by 40 before
using it as a raw-return baseline**. These checkpoints are not drop-in raw-value
PPO models without accounting for that scale.

Next inspect held-out errors by task category and basic versus return outcomes,
then consider a bounded actor update using this calibrated baseline, authentic
current-weight prefix reconstruction and unchanged retention/mission checks.
Do not train the actor simply because aggregate value RMSE improved.
