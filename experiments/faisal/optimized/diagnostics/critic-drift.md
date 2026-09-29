# Actor and critic drift in the first optimized trial

Run: `completion-20260927-135016`. This audit used saved 10,000- and 20,000-step checkpoints on CPU; it did not modify checkpoints or train a model. All dimensions, collision rules and success criteria remained unchanged.

## Observed approach failure

Deterministic evaluation used the same 25 seeds starting at 5,000,000 for each single-pass task.

| Measure | 10k centered | 20k centered | 10k varied | 20k varied |
|---|---:|---:|---:|---:|
| Clean passages | 24/25 | 1/25 | 23/25 | 0/25 |
| Ring hits | 1 | 23 | 2 | 23 |
| Left cage | 0 | 1 | 0 | 2 |
| Mean lateral offset at ending | -1.4 cm | +26.6 cm | +0.3 cm | +29.6 cm |
| Mean forward command | 0.735 | 0.223 | 0.634 | 0.218 |

The 20k policy often retreats first, then drifts sideways and hits the ring. This is not simply a policy waiting motionlessly for more training.

On 167 identical front-visible, marker-visible observations sampled every five steps from 10k trajectories, changing only the policy produced mean absolute command changes of 0.315 lateral, 0.975 forward, 0.424 vertical and 0.314 yaw. Mean Gaussian KL(20k || 10k) was 28.48 nats. The actor's new-input weight norm rose from 0.00187 to 1.998. These are measured changes, not proof that memory features alone caused the regression.

## Critic estimates were poorly calibrated

Each checkpoint was tested with stochastic actions on 50 fresh single-pass trials: both centered and varied tasks, seeds 6,000,100–6,000,124. The discounted return includes the fixed SAC entropy term from subsequent actions, matching the soft Q target convention.

| Measure | 10k | 20k |
|---|---:|---:|
| Passes | 43/50 | 1/50 |
| Mean initial minimum critic Q | 0.94 | 38.33 |
| Mean realized discounted soft return | 16.19 | -22.95 |
| Mean predicted-minus-realized difference | -15.25 | +61.28 |

Four 20k episodes timed out, so their reported returns are finite-horizon/censored rather than estimates with a continuation bootstrap. The other 46 terminated: 45 crashes and one passage. This limitation does not explain away the large mismatch on mostly terminal trajectories. This sample establishes poor calibration, not a mathematical estimate of error on every state.

## Episode probabilities did not balance replay transitions

The saved replay at 30k retained the chronological prefix without capacity wraparound. Single-pass transitions constituted only 4.54% of the first 5k samples and 7.04% of the first 10k, despite a 40% probability of selecting single-pass episodes. Long lap episodes contributed disproportionately many transitions. The first 10k samples contained 17 single-pass success reward events. By 20k, the cumulative single-pass fraction was 15.25%.

## Controlled next change

The next implementation samples half of each training batch from single-pass transitions and half from lap transitions whenever both are available. It retains the same replay data, rewards and termination flags; no teacher trajectories or actions are added.

The actor is also regularized toward its own frozen inherited policy on public single-pass tasks, currently visible front approaches, and up to 1.8 seconds after a front detection. A Gaussian KL penalty has coefficient 10. An attempted actor update is rejected, including rollback of its Adam state, if the protected minibatch's mean KL exceeds 0.10. Other lap observations are not anchored. These are initial experimental settings, not validated optimal hyperparameters.

The frozen prior contributes a training regularizer, not simulator actions. This algorithm is SAC with prior-policy regularization, not unmodified SAC. The hard check applies to the current protected minibatch only; it is not a global or statewise guarantee. It can restrict improvement, and a high rejection rate can stall actor learning. Fixed validation and independent tests must determine whether passing is retained and lap completion improves. The change does not promise success or resolve critic overestimation by itself.
