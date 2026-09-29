# Protected-run diagnosis

Run: `completion-protected-20260927-140307`, stopped at 100,000 new steps for stagnation. This report analyzes saved checkpoints and fresh read-only simulator rollouts. No model weights, replay, source code or running training were changed by this audit.

## Findings

The retention constraint preserved passing but did not teach a reliable return route. Across every checkpoint from 20k through 100k, centered validation remained 24/25 and varied validation 23/25. Full-task validation remained 0/25 throughout. The final policy made 23 passages in 25 front-lap trials, with exactly one passage in each of 23 trials; it never completed a second passage.

The learner was not globally frozen: it accepted 59,828 of 90,001 attempted actor updates (66.5%) and rejected 30,173. The accepted/rejected counts alone cannot prove whether individual rejected steps would have helped, but they rule out a total update stall.

Detailed deterministic front-lap rollouts used seeds 5,000,000–5,000,024 at four saved checkpoints. JSON results are in `protected-run-audit.json`, reproduced with `audit_protected_run.py`.

| Checkpoint | First clear passage | Reach side waypoint | Reach front waypoint / alignment phase | Final outcomes |
|---|---:|---:|---:|---|
| Initial | 23/25 | 3/25 | 0/25 | 16 timeouts, 9 crashes |
| 20k | 23/25 | 5/25 | 0/25 | 2 timeouts, 23 crashes |
| 60k | 23/25 | 12/25 | 0/25 | 23 timeouts, 2 crashes |
| 100k | 23/25 | 11/25 | 0/25 | 21 timeouts, 4 crashes |

The initial policy happened to achieve two passages in two episodes without following every shaping waypoint. Subsequent checkpoints lost these incidental laps. Passing itself is scored by physical clearance, not by completing the shaping phase sequence.

At 100k, the 25 trajectories spent 9,327 steps in `return_side` and 2,779 steps in `return_front`. **None of these sampled return steps was protected by the anchor mask.** Their mean current-policy/reference-policy Gaussian KL was 7.42 and 14.78 nats respectively, confirming that return actions changed substantially. The return failure therefore is not explained by the policy being directly anchored to the old return actions on these trajectories. Shared parameters and minibatch rollback can still indirectly restrict learning; this audit does not exclude that effect.

The final policy tended to move slowly or oscillate after its first passage. Mean speed was 0.062 m/s in `return_side` and 0.055 m/s in `return_front`, while mean distance to the active shaping target stayed 0.729 m and 0.533 m. It never reached the front waypoint in these tests. The 60k checkpoint had a higher mean episode return (19.62 versus the initial 4.76) despite zero completed laps: fewer crashes and retaining the first passage raised reward without satisfying the task.

## What the evidence supports next

Do not simply relax the approach anchor or extend this run. The demonstrated bottleneck is the transition from the rear-side return waypoint back to the front, while the existing approach skill is retained.

Before another training run, verify the return route is feasible in the exact current geometry and inspect the reward around each return milestone with a deterministic diagnostic route. That route must remain a test, not evaluation results attributed to the policy. Then test a short return-only curriculum with authentic observation/memory state captured after successful policy passages, so the learner can encounter later return states and eventual second-passage reward. Fresh blind resets with fabricated observation history would change the task and should not be used.

The current return reward depends on latched phase/side targets that are not explicit policy observations. This is an additional partial-observability concern, not a proven sole cause. The present camera/command memory stores the latest detection and recent command estimate; it does not guarantee recovery of every hidden phase or chosen side target. Any curriculum or observation change should be checked for consistent deployable information before training.

Keep the original full-task scoring and independent test seeds unchanged. Only actual repeated clean passages with low crash rates would establish improvement; reward gains or successful diagnostic controllers would not.
