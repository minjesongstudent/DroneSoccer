# Critic accuracy experiment

**Result: not selected.** The completed comparison is preserved in `evidence/diagnostics/asymmetric-comparison-v1`. At 5,000 updates, held-out crash-terminal MAE was 18.59 for the public critic and 22.51 for the asymmetric critic. This fails the predeclared evidence gate; no asymmetric policy run was launched. Actors were unchanged. The 16 terminal development trajectories came from eight source flights in the side-return phase; the 16 later-return trajectories timed out, so their true continuing values remain unassessed. The optional implementation is retained for reproducibility, not recommended as a demonstrated improvement.

The 20k authentic return-practice trial retained first-pass skill but made no new returns. Development crashes rose from 1/50 to 25/50. Read-only, frozen-policy trajectories showed substantial optimism about terminal crashes: minimum Q around -13.18 versus realized soft return -33.32. A small residual against the critic's own bootstrapped target did not predict this failure.

The experimental treatment provides training-only simulator information to the critic, alongside the actor's existing observation/history and action. The actor still receives exactly 100 camera/history, command-memory and requested-task values. True position, phase, velocity and wall information never enter actor prediction. The original single-pass reference remains frozen. All actual movements are sampled or predicted by the learned actor.

This is a hypothesis about value estimation, not evidence of a solved task or deployable sensing. A history-and-state critic has better justification than replacing history with simulator state alone; see [Baisero and Amato](https://arxiv.org/abs/2105.11674). The implementation is an adaptation to this SAC setup, not a claim that every assumption of that paper is satisfied. The actor's limited history may still limit its decisions.

## Diagnostic before policy learning

Compare fresh public-observation and asymmetric critics using the same real stochastic trajectories from a frozen policy. Train networks on CUDA, preserve the actor and reference byte-for-byte, and reserve development source flights. Compare terminal soft returns where no unknown timeout tail exists; report censored/time-limit trajectories separately. Never turn a practice truncation into an artificial terminal penalty. Keep calibration budgets, discount, entropy, network width and task balancing comparable. Record data, code, model and bank hashes and selection limits.

No actor training should start merely because unit tests pass. Review held-out calibration and the limitations of this diagnostic first. A clear calibration improvement permits a bounded behavioral trial; it does not establish return skill.

## Behavioral trial, if justified

Use the same good source checkpoint as the previous trial (`return-route-40k-20260927/best_model.zip`), authentic banks, five-way mixture, fixed entropy, reward v3, actor/reference protection and validation schedule. Fresh critics/replay. Save outputs on D:. Add `--asymmetric-critic` to the live command and use a new run name/port. Only one policy run may train.

Maximum 40k steps, evaluation every 5k. Preserve the existing 20k no-sustained-return-progress gate and 40k no-new-pass-gain gate. Full mission selection and independent acceptance remain separate and unchanged. Keep the 7M final seeds untouched until the existing candidate criteria are satisfied. Do not report critic calibration or diagnostic controller passages as learned-policy success.

The simulator-context wrapper must retain true pre-action and post-action context, including the terminal state before vector auto-reset. Saved replay must preserve timeout bootstrapping and matching context. Prediction, evaluation and model replay must work on plain 100-value observations without any context provider.
