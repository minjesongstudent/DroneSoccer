# Independent-rollout baseline actor trial

Critic fitting did not produce reliable return predictions. This bounded trial
uses a detached baseline from other on-policy trajectories of the same start.
It does not load a calibrated critic for advantage estimates.

Each batch has23 starts x3 independently seeded rollouts. Eleven authentic
return starts use FIT indices1,2,4,5,6,7,10,11,12,13,14; parent0/3 exclusions
remain. Eight new natural full starts (four delayed) and four basic starts
provide acquisition and retention coverage. Natural reset seeds are
2980000+100*update+group_id. Torch seeds995000+1000*update+entry_index.
Actual observations/prefixes are asserted equal across the three starts.
Every action comes from the learned policy, and each update collects fresh
trajectories from the current weights. No old off-policy trajectory is reused
for policy gradients.

At time t, baseline is mean return-to-go at t from the OTHER two trajectories,
with zero after those peers terminate. Own actions and rewards never enter its
baseline. Loss is negative log-probability times (return-baseline)*gamma^t/40,
summed over frames and averaged across trajectories. Fixed scale avoids a
normalizer depending on the same sampled outcome. Baseline stays detached.

Maximum4updates,69*600*4=165600 environment steps, lr1e-5, gamma.999,
gradient cap.5. Empirical source-policy KL reference on first batch histories:
mean<=.001,max<=.01; rejected steps restore weights/optimizer and stop.
Fixed evaluation after each accepted update includes centered,varied,return
lesson,full and delayed full. Stop if centered<24/25 or varied<20/25.
All full reports retain outcome/crash detail. No automatic model promotion.
This intervention also changes sampling and objective scaling; it is not a
controlled attribution to baseline choice alone.

Tests: test_independent_return_baseline.py (three passed: own-outcome
independence, absorbing zeros/group isolation, invalid inputs), and
 test_natural_recurrent_rollout.py (reset likelihood/value parity, gradients,
isolation and continuation-history checks passed). Actual batch replay
log-probability discrepancy must remain<.001 before every update.

Runner diagnostics/run_independent_baseline_actor.py. Output
 diagnostics/independent-baseline-actor-20260929 retains source snapshot,
runner/protocol/trajectories/checkpoints/evaluations/rollback diagnostics.
User-facing viewer stays on preserved source. No reserved acceptance seeds,
physical commands, publishing or group master changes.
