# Authentic second-passage reset trial: closed without qualification

The checkpoint implementation worked, but this training mixture did not improve whole-flight reliability. The final controller completed 46 clean randomized repeat missions out of 100 with 54 crashes; the original controller achieved 48 clean repeats and 52 crashes. The unchanged advancement gate required at least 53 clean repeats and at most 47 crashes. Keep the original `seenflag-s1-level1` controller selected. Do not extend this variant, promote an intermediate checkpoint or run acceptance on it.

## Completed experiment

The [frozen plan](AUTHENTIC_LATE_RESET_TRIAL_20261008.md) and [launch record](AUTHENTIC_LATE_RESET_LAUNCH_20261008.md) describe the exact algorithm, FIT seeds, budgets and source identities. The single RTX 4070 Ti CUDA run completed in 623.984 seconds, within its 3,600-second limit. All five rounds completed: 120 fresh starts plus 120 continuation tails, 71,655 training transitions, 282 accepted updates and zero rejected updates. The 120 continuations share 35 parents; there are 155 distinct training parent layouts, not 240 independent layouts.

Exactly 64 natural source flights produced 35 eligible second-passage checkpoints. Every complete restored suffix matched its original learned flight byte for byte before optimization. The closure independently recounted those 35 comparisons from saved arrays, including public observations, memory features, actions, float64 rewards, positions, headings, terminal flags, absolute clock and passage count. All 240 saved training trajectories matched the frozen schedule, retained the original step-600 limit and accounted for inherited versus new passages separately. Parent usage was balanced, three or four times each.

No source actor, GRU or frozen source identity changed. Both recorded feature-use probes passed. The final residual used all 1,032 parameters; final source guard KL was 0.000417 mean and 0.045138 maximum, within the unchanged limits. These findings rule out a failed checkpoint restoration or fully rejected update stream as explanations for this result. They do not establish that weakening the guard or taking larger updates would improve flight.

## Final normal-start development evaluation

| Suite, 100 cases each | Original first / return / clean repeat / crash | Final first / return / clean repeat / crash |
|---|---|---|
| Aligned | 84 / 84 / 84 / 16 | 84 / 84 / 84 / 16 |
| Nominal | 80 / 80 / 74 / 26 | 80 / 80 / 75 / 25 |
| Randomized | 73 / 70 / 48 / 52 | 73 / 71 / 46 / 54 |

The randomized total hides four rescued missions and six newly lost missions. Rescued seeds: 9430031, 9430059, 9430075, 9430087. Lost seeds: 9430010, 9430012, 9430024, 9430046, 9430070, 9430094. Of the six added crashes, five were ring hits and one was a net collision; four occurred after at least two passages, and two after one passage. Thus first-pass totals were retained while later reliability worsened. Nominal seed9420056 was rescued, with no nominal losses.

Checkpoint-tail training survived the original horizon without wall contact in 18, 15, 14, 21 and 15 of 24 tails across the five rounds. These are stochastic practice episodes with different parent mixtures and changing weights, not a matched learning curve or independent validation. No monotonic improvement can be claimed. Comparisons with the historical full-horizon run also use different FIT/RNG seeds and cannot isolate the causal effect of checkpoint sampling.

## Preserved evidence and next decision

Run folder: `D:/DroneSoccerData/runs/authentic-late-reset-20261008`.

- `decision.json`: `reject_close_variant`; complete training and final evaluation, no promotion, no acceptance, no physical flight.
- `closure_review.py` and `closure_review.json`: reproducible saved-artifact review, no policy inference, pickle loading, training or new simulation. Verified all 1,216 decision evidence hashes and 1,559 unique input hashes overall, including source identities, original comparisons and the reused saved-data diagnostic code.
- Final decision SHA-256: `a46a2934db8b58dc268b6c4deb069a44132a35797601da6d82cdba30fd791549`.
- Closure report SHA-256: `f01b5f71a7652905e0584f6da6d6824da9cfef31dd72dcb21fe734301803431b`.
- Original source flights, whole-state checkpoints, parity traces, training ancestry and tails, optimizer/RNG states, all five models and all guard/final evaluations remain preserved.

This reset mixture is closed. No additional trainer was launched after closure; the only remaining Python processes serve the read-only viewer. A further training run needs a distinct, measured hypothesis and fixed decision rule. Repeating this closure audit, extending the failed run or searching its intermediate models is not progress. The software tests and exact checkpoint restoration remain reusable work. Simulation acceptance and physical readiness remain incomplete; this algorithmic failure is not evidence of an unavailable-lab blocker.
