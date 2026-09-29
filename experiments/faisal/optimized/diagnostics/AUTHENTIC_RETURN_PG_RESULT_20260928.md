# Authentic return reward-gradient result

Completed the bounded CUDA trial in
`authentic-return-pg-stepwise-v2-20260928`: four updates, 5,782 training steps,
eight authentic return snapshots per update. All actual movements came from
the recurrent learned policy. No scripted actions, observation changes or
reward changes were introduced. The source model and viewer remain unchanged.

The trial used normalized discounted Monte Carlo reward-to-go, Adam 1e-6,
and detached current-weight prefix reconstruction. It did not rely on the
snapshot critic as a baseline. This was a small diagnostic policy-gradient
trial, not a standard full RecurrentPPO training run.

| Update | Centered /25 | Varied /25 | Front-start lap lesson /25 | General /25 | Delayed /25 |
|---|---:|---:|---:|---:|---:|
| 0 | 25 | 21 | 0 | 0 | 0 |
| 1 | 25 | 21 | 1 | 0 | 0 |
| 2 | 25 | 21 | 1 | 0 | 0 |
| 3 | 25 | 21 | 0 | 0 | 0 |
| 4 | 25 | 21 | 0 | 0 | 0 |

The lesson gain was transient. Single-pass skill was retained on these fixed
tests, but general full-mission performance did not improve. No candidate is
promoted and the independent acceptance seeds remain unused.

Training batches had 8, 8, 7 and 8 crashes out of eight continuations,
respectively. The third batch produced two additional passages in total;
saved historical passages are excluded from this count. Snapshot continuations
are not natural-start acceptance tests.

The first attempt stopped before optimization because batched sequence
likelihoods differed from stepwise collection. The stepwise evaluator preserves
within-segment backpropagation and produced exactly zero pre-update log-probability
error in all four completed updates. An intermediate retry encountered a NumPy
JSON serialization error after an unsaved in-memory update; its artifacts were
preserved separately. Both failures are documented in their status files.

Next inspect successful and failed return trajectories from these saved batches,
including whether passage progress ends in a crash or fails to sustain another
lap. Do not extend this trial on the basis of transient lesson success.
