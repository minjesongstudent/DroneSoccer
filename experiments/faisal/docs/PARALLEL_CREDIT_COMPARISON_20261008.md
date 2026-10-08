# Three-arm parallel credit comparison

Faisal explicitly requested concurrent variable comparisons on 8 October Sydney. This replaces the earlier one-trainer preference for this bounded cohort. It does not authorize changed success criteria, physical operation or publication. The original85 policy remains selected until the required evaluations establish a better candidate.

The preceding estimated-motion residual failed its fifth retention check: randomized clean repeats 13 to 9 and crashes 12 to 16 in the fixed 25-case guard. All additional failures occurred after two or more passages. Motion estimates stayed accurate on the new training data, and no optimizer update was fully rejected. This motivates comparing credit construction; it does not establish a unique cause. The unchanged failed residual is not resumed or rerun as an arm.

## Frozen comparison

All three arms use the same zero-initialized, 1,032-parameter bilinear correction, frozen original85 actor, frozen action variance, frozen GRU and its FIT normalizers. Every movement is produced by the learned policy. Public inputs and startup/reset handling remain unchanged; simulator truth is limited to reset, rewards, scoring and diagnostic labels.

| Arm | Return and advantage construction | Hypothesis |
|---|---|---|
| `full_horizon` | Dense finite Monte Carlo reward-to-go with gamma 1.0, minus the fitted public89 value baseline | Giving late collisions their full weight may improve sustained laps. |
| `mission` | Undiscounted binary target: at least two full-body passages, all 600 steps survived, no crash or wall contact; minus the fitted public89 value baseline | Directly learning the scored mission may improve reliability. |
| `local_credit` | Dense gamma .995 Monte Carlo targets for the value baseline; GAE gamma .995/lambda .95 with that baseline frozen during advantage construction | More local credit may work better with the estimated-motion correction. |

Gamma .995 weights an outcome 400 steps ahead by approximately .135 and 599 steps ahead by .050. This is a mathematical property, not proof that discounting caused the observed failures. Mission and GAE objectives previously failed with a trainable full85 actor; the current comparison tests their interaction with the frozen-base motion correction. They are not presented as wholly new algorithms or guaranteed improvements.

Each arm has five rounds of 48 complete learned-policy episodes, maximum 600 steps per episode, one actor epoch per round, minibatch256, actor Adam1e-5, and 100 value batches with Adam3e-4. Existing source KL limits .01 mean/.1 maximum, boundary penalty .1, four-step backtracking, approach/full-task curriculum, geometry, dynamics, rewards and passage/crash definitions remain unchanged. A GAE tail is zero at both crashes and the finite 600-step horizon; no reset-crossing bootstrap or Q action gradient is permitted.

## Matched samples and concurrency

The coordinator must scan existing allocations before reserving the shared FIT seeds 11244000 through 11244239. Sharing these seeds between the three explicitly registered arms is intentional. They use algorithm seed11592000, applied after loading the source policy, and independent process-local random states. The software smoke uses only 11244900 and11244901 and cannot pass a policy gate. No development or reserved acceptance layout is used for fitting.

Before the first update, the three arms' saved features, issued actions and episode lengths must agree exactly. Returns and later trajectories can differ as intended. Subsequent differences estimate a combined target/advantage choice, not three independent replications or a confidence interval. The preceding failed experiment is historical context, not a matched concurrent control.

One coordinator holds the existing project training lock. At most three registered workers run simultaneously. A Windows Job Object ties their lifetime to the coordinator; each worker must receive a valid private start grant before any CUDA work. Merely supplying a manifest must not bypass the lock. Each worker has its own output directory, log, model/optimizer/RNG state and STOP file. The coordinator has a global STOP. Other standalone trainers and a second cohort remain excluded.

Each worker uses one CPU thread and CUDA float32 with autocast and both TF32 flags off. Its PyTorch allocator limit is .15 of the GPU's total memory; this does not cap CUDA context or other library allocations. Actual GPU usage must be recorded. The machine has one RTX4070Ti with 12GB; parallel speedup is an empirical outcome because rollouts also consume CPU time. Out-of-memory, nonfinite values, stop requests, retention regressions or the 60-minute worker budget close the affected arm; no automatic retry, budget extension or memory-limit increase. The coordinator has a 65-minute bound including cleanup.

## Final-only decision

Each round retains the existing 25-case-per-suite stop check. Only a final fifth checkpoint that completes the whole protocol without regression is eligible for the existing 100-case-per-suite screen and offline feature-use checks. Require at least five more randomized clean repeats and five fewer crashes than the hash-bound original, with no suite losing more than two first passages or repeats or adding more than two crashes. Keep the same feature-use check and full-body passage definitions.

Record all arms, including failures. If more than one final model qualifies, select at most one by randomized clean repeats descending, randomized crashes ascending, randomized first passages descending, then fixed arm order `full_horizon`, `mission`, `local_credit`. An arm with a missing, failed or incomplete decision cannot be selected. The best failing arm is never eligible.

Selection permits only one separate fresh paired development comparison with the original policy. It is not promotion, sustained qualification or independent acceptance. Reserved acceptance seeds remain untouched. No intermediate checkpoint selection, extra favorable cohorts, unchanged extensions or post-result changes to this selection rule.

Preserve the immutable coordinator/worker manifests, source snapshots, allocations, full trajectories, targets, advantages, final model identities and machine decisions. All actual physical flight remains unvalidated.
