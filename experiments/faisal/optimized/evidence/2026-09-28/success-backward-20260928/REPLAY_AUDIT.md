# Replay and frozen-policy value audit

Training remains stopped; no model parameters or success criteria changed.

All59 authentic near-pass training episodes were replayed from their saved full-history reset states using their actually collected learned actions. Every observation, next observation, reward and episode boundary matched the saved replay.

Before10k:40 complete authentic episodes,35 with a new pass,14 surviving successes,21 post-pass crashes.
From10k:19 complete authentic episodes,17 with a new pass,5 surviving successes,12 post-pass crashes.
Thus positive passage and subsequent crash experience both exist in replay. This does not establish adequate sampling weight, corrected trace influence, or accurate learned values. These are episodes on four repeated FIT layouts, not independent layouts.

## Frozen15k policy

32 stochastic continuations from four FIT near-pass snapshots (eight per layout):24 achieved a new pass,3 passed and survived200steps,29 exited the cage. Every action came from the unchanged learned Gaussian policy on CUDA. Camera/command observations were preserved. No reserved-check or independent acceptance layouts were used.

For29 crash-terminal trajectories, initial min-Q minus realized discounted soft return had mean+8.3847 and mean absolute error8.4361. Actual soft return includes successor action entropy, excludes current action entropy consistently with SAC Q, gamma.995, and stops at the true crash terminal. The three200step timeouts were excluded because their finite returns omit a bootstrap continuation. Positive bias by layout was6.17,18.84,4.79,0.95.

These are terminal-selected conditional measurements and correlated starts; they are not an unbiased estimate of overall Q error. They show optimism for these realized failures, not proof that Q action ranking or a specific optimizer defect caused regression. No causal attribution or new reward loophole is established.

## Next diagnostic

Before another training launch, test the usefulness of finite-return targets with positive passage-plus-survival and crash data, using disjoint FIT-layout folds and a frozen actor. Compare against the existing critic and a constant prediction baseline, retain censoring/entropy semantics, and require held-out predictive improvement. Do not assume better critic calibration yields better flight or relax retention/mission tests. Any actor update trial needs separate measured flight gates. Avoid simply extending failed SAC training.

Reproduce with audit_replay.py using the frozen source_snapshot, replay.pkl and model_15000.zip. Data and full results: replay_audit.json and replay_audit_trajectories.npz. All inputs and model stayed unchanged; no new training is active.
