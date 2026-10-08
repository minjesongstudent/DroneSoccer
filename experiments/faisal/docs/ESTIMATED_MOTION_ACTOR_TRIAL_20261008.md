# Estimated-motion residual actor: predeclared trial, 8 October 2026

Status: core model and feature builder implemented in `src/estimated_motion_actor.py`, with 21 focused CPU tests passing in `tests/test_estimated_motion_actor.py`; no experiment runner, policy fitting or flight rollouts launched for this trial. These include a small gradient/optimizer unit check, not a learning experiment. Original85 remains selected. Estimator accuracy is not flight success, simulation acceptance, or physical-flight validation.

## Evidence and one hypothesis

The frozen GRU64 in `D:/DroneSoccerData/runs/recurrent-estimator-20261008` reduced pre-action previous-displacement RMSE against the equal-budget feedforward estimator by 30.69% overall / 39.65% blind on the historical development-heldout episodes and 31.26% / 39.59% on the separately preserved GAE cohort. Visible RMSE improved too. The target is forward/right/up millimetres per completed interval, not metres per second. These are recorded simulation cohorts, not independent flight acceptance.

Earlier added-command actor columns had little action influence and did not produce reliable flight improvement. This single experiment tests whether an explicit, frozen estimated-motion feature can support useful context-dependent corrections to the original actor. It does not repeat critic-ranking, GAE, reward changes, history-width sweeps, or unrestricted source-actor fine-tuning.

Frozen identities:

- Source85 `trainedModels/seenflag-s1-level1.zip`: `8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef`.
- Estimator `gru64.pt`: `b48d6b26136a5e2e6cbfcd615a1a819eb801b495fba4cfa38130c0e2da54f554`.
- Estimator `normalizers.npz`: `3bb05728cf3eab59e34733be32971397bb2df8614134448cc2e8c705ee9386db`.

## Explicit input and causal state contract

Use a separately named `estimated-motion-residual-v1` 89-input layout: unchanged original public85, then three standardized estimated displacements, then a binary valid-history flag. It is incompatible with the older 89-input four-command layout; dimensionality must never select a profile. Do not normalize or reorder original85.

At each new pre-action observation, feed `(public85 - saved FIT x_mean) / saved FIT x_scale` once through the frozen unidirectional GRU. Its three outputs already equal standardized targets: `(estimated displacement_mm - saved FIT y_mean) / saved FIT y_scale`. Do not normalize twice. Reset hidden state at every real episode reset. Feed startup frames through the GRU; mask appended estimates to zero and validity to zero for frames 0 and 1, becoming valid on frame 2. Zero standardized features represent removal of the residual contribution, not a claim that physical displacement is zero.

The inference contract is explicit `strict_float32`. The future runner must set `torch.backends.cudnn.allow_tf32=False` and `torch.backends.cuda.matmul.allow_tf32=False` once at startup and record these values. Components reject incompatible CUDA flags; they never mutate global precision settings. Default cuDNN TF32 produced about .008175mm packed/stream CPU differences on a saved600-frame trajectory; strict FP32 reduced CPU-versus-CUDA-stream and packed-versus-stream maxima to .000108mm and .000115mm, below the unchanged .001mm numerical check. Original estimator training/results used their recorded defaults and are retained unchanged; this is a documented inference precision difference, not modified weights or an accuracy retest.

Only camera history and previously actually issued commands enter public85. Proposed/current actions, simulator positions/headings, target labels, rewards and future frames never enter actor or estimator inputs. Advance recurrence once per observation; cache an already processed next observation instead of feeding it a second time. Save the exact features used for each action. Estimator weights, scales and source actor remain frozen throughout.

## Small residual with exact initialization

Let `o` be public85, `e` the standardized three-vector, `v` the validity flag and `c = [1, o]`. Flatten `c outer e` in a documented row-major order (258 entries). Train only a zero-initialized matrix `W` with shape 4 x 258, without bias:

`mu(o,e,v) = mu_source(o) + v * W @ vec(c outer e)`

Keep the original log standard deviation and tanh-Gaussian action distribution. There are 1,032 trainable parameters. Every correction depends on estimated motion; removing the estimate removes the correction. Gradients reach W immediately at zero initialization. Original85 matrix operations remain exactly unchanged, avoiding the rounding changes previously observed when widening a source layer. No hand-written controller, action cancellation rule or oracle-state substitution is introduced.

Before any gradients, require exact original distribution, deterministic and seeded stochastic action parity, save/reload parity, startup/zero-estimate parity, and exact trajectories on the existing 25-per-suite guard cases. Check recurrence reset, prefix causality, actual-issued-action alignment and no double advancement separately. Reject a mismatched source, estimator, profile, nonfinite estimate or malformed validity flag.

The wrapper preserves SB3's final action unscaling arithmetic, even for [-1,1], because returning raw tanh can differ by float32 ulps. Head checkpoints contain safe float32 arrays plus the explicit layout, source/estimator hashes and inference precision contract; restoration requires the supplied candidate checksum and separately verified frozen artifacts. Episode hidden state is reset on load.

## One bounded learning run

- CUDA only, one trainer lock; fixed initialization seed 11590000, head Adam 1e-5, no seed or learning-rate sweep.
- Five rounds of 48 fresh complete FIT episodes, max 600 steps each (at most 144,000 transitions). Proposed FIT seeds 11242000..11242239 are within the existing `[11000000,11250000)` guard; do not widen that guard. Preflight metadata had no 11242xxx allocation, and a scan of 593 `episodes.jsonl` logs / 20,570 rows under `D:/DroneSoccerData/runs` found zero collisions. Launch must confirm and save the scan before using these seeds. Smoke uses separate declared FIT seeds and cannot pass a learning gate.
- Preserve the existing alternating approach/full-task reset curriculum, dynamics, rewards, passage/crash definitions and actually learned stochastic actions.
- Use complete finite-horizon Monte Carlo reward-to-go, gamma .995, zero tails at crash and step600. Fit an action-independent 89-input 128/128 state-value baseline (100 Adam3e-4 batches per round) only on that round's FIT rollouts. Standardize detached advantages. No SAC Q gradient, GAE, scripted rollout or development-trajectory fitting.
- One actor epoch per round, minibatches256, clipped ratio [.9,1.1], same source KL penalty and boundary penalty .1; source mean KL<=.01 and maximum<=.1 with existing four-step optimizer backtracking and complete rollback of Adam moments. Use 8,192 fixed warmup FIT guard frames after reconstructing their GRU states from complete reset episodes, not from shuffled frame starts.
- Stop on STOP, nonfinite values, 100 consecutive rejected proposals, retention regression, or the fixed 60-minute run budget. Save final state on bounded exit; do not extend or select an earlier checkpoint. Incomplete main runs cannot pass the final screen.

The original warmup FIT episodes provide source guards only. Historical heldout and the separately preserved GAE240 episodes are never actor fitting, normalization or checkpoint-selection data. Save model/head, optimizers/value baseline, RNG, episode seeds, exact public/estimated inputs, actions, rewards and source/input hashes; write the protocol before fitting. Record CUDA/runtime versions and verify frozen parameters/checksums after fitting.

## Final-only go/no-go

Keep the existing 25-per-suite retention stop after each round. Only the final completed model is eligible for a fixed100-per-suite comparison with the hash-bound actual original report `lab-data/policy-validation-20261006/results.json` and episode logs. Recount outcomes rather than trusting summary numbers. Require randomized clean repeats >= original+5 and crashes <= original-5; in every suite allow no first/repeat loss greater than2 or crash increase greater than2. The current baseline is aligned84/84/16, nominal80/74/26, random73/48/52 (first/repeat/crash), so randomized repeat>=53 and crash<=47, but the machine gate must read preserved evidence rather than hardcode these counts.

Also report an offline feature-use check on 8,192 fixed valid FIT frames and 8,192 fixed valid development-heldout frames. Freeze sample/permutation seeds before training (11591000/11591001) and shuffle the three estimates together only within visible/blind strata. No simulator is stepped with altered observations. Define D_source as mean absolute deterministic-action difference from source and D_shuffle as mean absolute difference when only estimates are permuted. Require both finite and >1e-6 and D_shuffle >= .5 * D_source on each cohort; zero estimates must recover source actions. This checks feature dependence, not flight improvement, and does not impose an artificial large-action target.

Passing all software, integrity, feature-use and fixed-flight gates permits only the previously specified single fresh paired development comparison. It does not promote the policy or replace sustained qualification and reserved independent acceptance. Failure closes this variant: no unchanged extension, sweep, selected checkpoint or additional cohort seeking a favorable outcome. Physical flight remains unvalidated and no hardware or transport is operated by this trial.
