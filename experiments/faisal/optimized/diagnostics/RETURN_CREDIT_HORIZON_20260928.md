# Delayed return credit audit and bounded comparison

Computed discounted returns from all 32 unchanged reward arrays in the prior
authentic-return trial. At gamma 0.99, events 300 steps ahead retain 4.90% weight;
at 600 steps, 0.24%. At gamma 0.999 those weights are 74.07% and 54.86%.

In these recorded paths, the additional-pass trajectory that later collided
has return 3.91 at gamma 0.99, versus 1.73 for the timeout survivor. With gamma
0.999 the respective returns are 1.19 and 12.67. One zero-additional-pass crash
also has positive return 0.257 at gamma 0.99. These trajectories have different
starting states; the ranking is descriptive evidence about credit assignment,
not proof of a policy exploiting rewards or a causal comparison of actions.

The environment's undiscounted distance-difference shaping and its rewards
remain unchanged. No success criterion is relaxed. This audit motivates testing
a longer credit horizon, not assuming that it will improve training.

## Comparison protocol

Runner: `run_authentic_return_long_horizon.py`.
Output: `authentic-return-pg-long-horizon-20260928`.

Same preserved source, eight authentic early-return snapshots, Torch seed rule,
learning rate 1e-6, reward-only policy-gradient objective, four-update maximum,
and fixed 125-episode validation as the previous gamma 0.99 trial. Only gamma is
changed to 0.999. Using the same snapshot set isolates that change; the newly
collected re-approach bank is intentionally not mixed into this comparison.
Both runs continue to the original mission endpoint after each saved state.

Stop if centered performance falls below 24/25 or varied below 20/25. Preserve
every source snapshot, model and rollout. No promotion based on training reward
or isolated continuation successes. Compare natural-start outcomes, crashes,
return-to-front and repeated passes when results arrive. No independent
acceptance seeds or physical hardware are used.
