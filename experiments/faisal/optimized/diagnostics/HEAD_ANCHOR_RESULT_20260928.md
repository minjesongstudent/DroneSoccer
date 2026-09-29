# Frozen recurrent features with basic-output anchoring

The previous lap-only trial completed two passes on fixed lesson seed5000004
at steps34/295, but collided at435. Baseline500 passed at32 then left the cage
at100. The return trajectory is preserved in capacity_two_pass_audit.json; two
passes followed by a collision do not meet the 600-step mission criterion.

A separate CUDA trial started from baseline500. Only action_net parameters were
trainable; LSTM, hidden policy layers and other model parameters stayed fixed
(parameter hash assertions passed at each evaluation and completion). Loss was
lap imitation on all9 full600-step FIT teacher histories plus100 times the mean
squared difference from baseline outputs on8 sampled basic histories. Zero state
at genuine episode beginnings; no synthetic state or scripted actions.

| Updates | Centred | Varied | Front lesson full mission | General full | Delayed full |
|---|---:|---:|---:|---:|---:|
| 25 |25/25|21/25|1/25|0/25|0/25|
| 50 |25/25|21/25|0/25|0/25|0/25|
| 100 |25/25|21/25|0/25|0/25|0/25|

The sole lesson success at25 was seed5000003, two passes and timeout at600.
It did not persist at later checkpoints and is not acceptance evidence. Basic
retention held in these fixed tests, unlike the previous unconstrained trial.
Final pre-update lap MSE .101602 vs baseline .101967; basic anchor MSE ~8.3e-7.
The constrained head has made little progress on imitation; this does not prove
the entire architecture is incapable, but extending this trial unchanged is not
justified. It stopped at its100-update bound; no active trainer remains.

All model checkpoints, source snapshot, runner, protocol, loss history and fixed
evaluations are in lap-head-anchor-20260928. No reserved or7M acceptance layouts,
physical hardware or publication. Viewer/model exports remain baseline500.

Next: investigate the difference in the lone success trajectory and whether a
less restrictive change can retain basic skill. Do not declare the policy ready
from one selected validation success. Any new optimizer/architecture variant
must preserve retention and produce sustained full-suite gains before fresh-seed
acceptance testing.
