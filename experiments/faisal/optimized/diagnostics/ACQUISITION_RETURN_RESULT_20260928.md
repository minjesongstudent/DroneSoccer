# Acquisition/return mixture result

The run finished after 12,438 collected steps. Three updates were accepted;
the fourth exceeded maximum frame KL (0.0115244 versus the predefined 0.01)
and restored weights and optimizer state. No limit was relaxed. All accepted
checks retained 25/25 centered and 21/25 varied success, with zero complete
lesson, general or delayed missions. The source remains the active policy.

All 108 saved rollout natural-start flags and prefix shapes were verified:
natural episodes have empty prefixes, authentic continuations have nonempty
prefixes. The training records contain actual learned-policy movements.

## Credit distribution audit

The timestep-weighted batch mixture varied with episode duration. Early-return
frames accounted for 39–59%, re-approach 17–33%, natural full-task 14–27%, and
basic passes 4–9%. Nominal episode counts therefore do not describe gradient
weighting. Every natural full-task training episode crashed in these batches.

Basic-pass targets averaged +4.23 and +3.13 normalized units in batches 0 and 2,
while re-approach targets averaged -0.43 and -0.38. These are pooled reward-to-go
weights, not estimates of action advantage relative to the same state. They
mix task difficulty, progress, terminal reward and behavior. This does not alone
prove biased policy gradients or that a different baseline will solve flight;
it identifies a possible variance issue to test rather than another rate sweep.

Detailed task shares, rewards, passages, failures and guard values are stored
in `acquisition-return-mixture-20260928/credit_balance_audit.json`.

## Next justified experiment

Calibrate a public-history recurrent value baseline on training trajectories,
with episode-disjoint held-out trajectories and actor weights frozen. Verify
both held-out prediction quality and exact actor-output retention before using
it for actor updates. Reconstruct critic memory under current critic weights;
do not treat old snapshot states as valid. Do not use the 5M validation or 7M
acceptance episodes for this fit. Failure to predict held-out returns is a
reason to reconsider the baseline, not to train the actor with an unverified
critic. Preserve all source and experiment checkpoints.

No training remains active from this run. Physical flight and simulation
acceptance remain unachieved; no publishing or hardware commands were issued.
