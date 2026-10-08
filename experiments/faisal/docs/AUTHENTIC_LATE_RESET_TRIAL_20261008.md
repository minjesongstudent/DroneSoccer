# Authentic second-passage reset trial

## Question and scope

Can more training exposure to authentic states reached after two complete passages improve whole-flight reliability? The closed parallel credit comparison changed actual behavior but traded rescued flights for crashes after two or more passages. Full horizon added four randomized crashes and rescued four. Local credit added seven and rescued five. Those records justify testing a different reset distribution; they do not justify increasing update size, weakening guards or extending a failed run.

This trial applies Faisal's checkpoint suggestion to the current 85-input controller and current group simulator. Historical authentic-state training used different legacy policies, observation stacks, wrappers and short practice horizons. Those results remain negative historical evidence, not current-controller results. No historical bank is reused.

The experimental controller starts again from the registered original85 actor plus the zero 1,032-parameter motion-dependent residual. The base actor, action variance, GRU and FIT normalizers stay frozen. The credit construction remains full-horizon dense Monte Carlo, gamma 1.0, with the same 89-input fitted value baseline. This tests a new reset mixture, not a continuation of the rejected Full horizon checkpoint.

## Bank collected before optimization

Use exactly 64 fresh natural randomized `HoopEnv(level=1, randomise=True, hanger=True)` starts, FIT seeds 11245000..11245063. The original zero-head learned policy produces every command deterministically. Every source flight is preserved to its original crash or step-600 endpoint, including flights without a checkpoint.

Capture at most one checkpoint per parent: immediately after its second audited full-body clearance, provided the flight is active, has no wall contact and the original elapsed step is at most 450. This rule cannot depend on the later endpoint or later success. Snapshot the entire `ClearanceTrainingEnv` with tracing disabled, including simulator RNG, delay queue, velocity, route, geometry, both histories, audit state and original elapsed steps. Preserve the authentic public85 prefix and current observation, with source and payload hashes. These are trusted local simulation snapshots, not a format for accepting untrusted pickle files.

Before any optimization, restore each checkpoint without reset, placement, reconstructed geometry or synthetic history. Rebuild frozen GRU state from the recorded public prefix, then process the current observation once. Its full deterministic remaining suffix must exactly match the uninterrupted source flight's features, actions, rewards, positions and terminal flags. Fail closed on any mismatch. At least 16 distinct source parents must qualify; insufficient coverage closes the trial without changing the capture rule or collecting extra layouts.

The resumed episode retains its original step counter and ends at the original step-600 horizon or crash. It does not receive 600 additional steps. Inherited passage counts remain diagnostic provenance; new passages are reported separately. No simulator truth enters the deployable actor input.

## One bounded training run

Five rounds, 48 trajectories per round: 24 fresh starts and 24 checkpoint continuations. Fresh starts use the existing alternating approach/full-task curriculum, seeds 11244600..11244719 exactly once. This preserves that curriculum within the fresh-start half; the overall reset mixture changes explicitly. Checkpoint selection rotates in sorted parent order across rounds so use differs by at most one. Every eligible parent is retained regardless of source outcome. Descendants share parent identity and are not independent layouts.

There are 120 fresh training starts and 120 continuations, not 240 new independent flights. Only continuation-tail transitions receive actor/value updates; the historical prefix is used solely to reconstruct causal memory. Dense reward accounting remains unchanged, and Monte Carlo tails are zero at crash or the original finite horizon. All issued movement remains sampled from the learned policy.

Retain Adam1e-5 on the residual only, value Adam3e-4 with 100 batches, one actor epoch, minibatch256, clipping .1, source KL mean .01/max .1, boundary penalty .1 and four-step backtracking. Use the existing 8,192-frame FIT source guard and unchanged offline feature-use check. Seed Python/NumPy/Torch with 11594000 after loading the frozen artifacts. No learning-rate, seed, architecture or credit sweep is authorized by this plan.

One CUDA worker holds the project training lock, with allocator fraction .15, one CPU thread, float32, no autocast and both TF32 flags off. No other trainer/cohort may run. A 3,600-second wall budget includes preparation, bank collection/parity, training and evaluations. STOP is checked at every frame and update batch. Nonfinite values, source changes, failed parity, insufficient bank coverage, 100 consecutive rejected updates, retention regression or budget expiry close the trial. No automatic retries or budget extensions.

## Software smoke and final decision

Before MAIN, run a separate software-only smoke: eight bank source seeds 11244810..11244817, minimum one eligible parent, one fresh start 11244800 plus one bank tail, one update round, two value batches and two cases per development suite. Algorithm seed11594900, maximum600 seconds. Full snapshot/parity, save/reload and evidence checks still apply. A smoke can never qualify a policy. Scan every proposed seed against prior actual and declared allocations before reserving it; any conflict blocks launch rather than silently choosing another seed.

MAIN uses the unchanged 25-case retention stop after each round. Only checkpoint five after all prescribed work may receive the fixed 100-case-per-suite development screen and feature-use checks. Require randomized clean repeats at least53/100 and crashes at most47/100, with no suite losing more than two first passages or clean repeats or adding more than two crashes versus the hash-bound original. A checkpoint-start result can never substitute for a normal-start result.

Failure closes this reset-mixture variant: no earlier checkpoint selection, source-outcome filtering, extra bank or favorable evaluation cohort. Passing permits one separate fresh paired development screen only. Sustained fixed validation and independent fresh-seed acceptance remain separate, unchanged requirements. Physical flight remains unvalidated.

Preserve plan, runner/tests/source snapshots, every parent flight/checkpoint, every descendant with ancestry, actual observations/actions, value targets, advantages, optimizer/RNG states, final identities and machine decisions. No group master edits, push, hardware operation or outside messages.
