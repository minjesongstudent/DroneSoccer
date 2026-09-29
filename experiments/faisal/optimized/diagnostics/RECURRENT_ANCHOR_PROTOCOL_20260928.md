# Recurrent adaptation with basic-output protection

The preceding frozen-LSTM trials retained basic skill but produced no sustained
full-mission result. The optimal linear-head audit left .0782 imitation MSE,
and the matched-history reacquisition audit found .3485 MSE at the key events.
This supports testing recurrent representation adaptation rather than another
output-head coefficient sweep. It does not establish that representation is the
sole blocker; only nine successful FIT histories remain a coverage limitation.

Comparison: identical to lap-mlp-anchor10-retry-20260928 except actor LSTM weights
can now train alongside policy MLP/action head. Starting policy500, same data,
seeds44/45, Adam1e-4, basic-output MSE coefficient10, whole600-step lap histories,
maximum100updates and checks25/50/100. Critic and other nonactor parameters remain
frozen and hash-checked. Stop if centred<22/25 or varied<20/25. Full mission success
still requires the existing600-step survival/two-pass test; no reward change.

No new demonstrations, scripted controls, invented recurrent history or truth
observations. This is supervised transfer, not new RL experience. No independent
acceptance or reserved data used. Results/source/hashes are preserved under
lap-recurrent-anchor10-20260928. Viewer and exported model remain baseline500.

Completed100 updates. At25/50/100: centred25/25 throughout; varied21/22/21; frontlesson0/3/0 outof25; general/delayed0throughout. At50 seed5000014 completed3passes and survived600; seeds5000003/5000021 completed2 and survived600. These selected validation successes are not acceptance. Final preupdate lapMSE .093247 despite disappearance of successes. All frozen hashes passed. Preservecheckpoint50 for further diagnostic evaluation; do not extend imitation unchanged or claim winner. No active trainer.
