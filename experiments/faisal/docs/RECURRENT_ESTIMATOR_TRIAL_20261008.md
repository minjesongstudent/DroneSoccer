# Offline recurrent pre-action motion estimator

Status: completed on the RTX 4070 Ti and passed the prespecified estimator comparison. This is a diagnostic estimation result; it does not train or replace the flight policy. Both models completed exactly 3,000 updates in one bounded run, taking 85.2 seconds in total. The original policy remained unchanged.

## Final result

| Recorded cohort | Metric | Feedforward RMSE | GRU RMSE | Reduction |
|---|---|---:|---:|---:|
| Warmup development-heldout, 111 episodes | All intervals | 5.693 mm | 3.946 mm | 30.7% |
| Warmup development-heldout | Ring not visible | 6.269 mm | 3.784 mm | 39.6% |
| Warmup development-heldout | Ring visible | 5.284 mm | 4.048 mm | 23.4% |
| Separate recorded GAE cohort, 240 episodes | All intervals | 6.173 mm | 4.243 mm | 31.3% |
| Separate recorded GAE cohort | Ring not visible | 6.740 mm | 4.071 mm | 39.6% |
| Separate recorded GAE cohort | Ring visible | 5.768 mm | 4.353 mm | 24.5% |

These are pooled per-axis errors in **millimetres per completed interval**, not speed errors. Both cohorts exceeded the required 15% improvement overall and during invisible-ring intervals; visible-interval errors improved. The saved machine decision is `supports_future_estimated_state_integration_review`. It records actor integration, policy promotion, simulation acceptance and physical flight as false. No trajectories were generated during this offline experiment.

Root verification passed 17 focused tests (15 new estimator checks plus two existing motion-prediction checks). The two-update CUDA smoke completed as `software_only`, before the full experiment. Evidence is preserved in `D:/DroneSoccerData/runs/recurrent-estimator-20261008`, with separate smoke artifacts in `recurrent-estimator-smoke-20261008`. The pre-fit protocol and source documentation are snapshotted in that run; this section was added afterward.

Final GRU checkpoint SHA-256: `b48d6b26136a5e2e6cbfcd615a1a819eb801b495fba4cfa38130c0e2da54f554`. Feedforward checkpoint SHA-256: `536a0bacedb9d4c48b188e49ca1a96f33cb414f64cabb6d2a9556606950a5298`. Protocol SHA-256: `c9857d474d3af76fe54d1b904a2ca6f74be3ec72b5e4ffba7eae6f0e7fee5f86`. `decision.json` binds all final predictions, models, split, schedule and results.

Independent closure verification reproduced all prediction metrics, causal targets and six gate comparisons without importing the runner or loading model weights. It verified 15 decision evidence hashes, nine frozen sources, 805 input files and four protocol artifacts, including FIT normalizers, the 447/111 split and shared 3,000 by eight sampling schedule. `closure_review.py` and `closure_review.json` in the run directory preserve that check. No new rollouts were used.

The next justified step is a frozen streaming estimator with explicit state resets, followed by a separately specified policy integration and flight-outcome experiment. Estimator error reduction alone cannot establish better ring passages, returns or crash rates. Neither the original selected policy nor the Mac preview package has been replaced.

## Streaming integration and precision

The frozen runtime is now implemented in `src/recurrent_motion.py`. It accepts only the named public85 profile, preserves those inputs, maintains one private GRU state per episode and returns both millimetre and FIT-standardized displacement estimates. The first two outputs carry `valid_history=False`; recurrence still consumes those frames. No current action, position, heading or target enters the runtime. Runtime failures do not advance state or substitute an estimate.

A full 600-frame recorded episode exposed a backend difference missed by the short unit check. CPU streaming differed from the experiment's saved CUDA packed predictions by up to 0.008175 mm, failing the predeclared 0.001 mm numerical-parity tolerance. The historical comparison and failure remain in `streaming_review.py` and `streaming_review_failure.json`. Reconstructing the original eight-episode CUDA packed batch with cuDNN TF32 enabled reproduced the saved predictions exactly; this isolated a backend precision and batch-shape effect, rather than an adapter state bug. Neither weights nor historical predictions were changed.

The runtime now requires explicit `strict_float32` inference. CUDA callers must disable both cuDNN and matrix-multiplication TF32; all callers must disable autocast. The adapter checks these conditions and raises before state advancement, without changing process-global flags. In the new strict check on the same recorded 600 frames, CPU batch/stream differed by 0.00001145 mm, CUDA batch/stream by 0.00011063 mm, and CPU/CUDA streams by 0.00010777 mm. All passed the original 0.001 mm tolerance. Input bytes, reset behavior and startup validity also matched exactly. This is numerical consistency evidence on one episode, not a new estimator-accuracy or flight test.

Reproduction artifacts are `backend_precision_diagnostic_snapshot.py`, `backend_precision_diagnostic.json`, `strict_float32_contract_manifest.json` and `strict_streaming_review.py/json` inside the estimator run. The old runtime is preserved under `integration_source_before_precision`; the strict runtime snapshot is under `integration_source`. Runtime SHA-256 is `460c5b29753f2a94384ea37536a7a9b7933111373286d05416d8841f37f4f010`. The combined estimator, adapter, residual-actor, motion and history test run passed 85 tests after this fix; counts overlap earlier checks.

## Why this single change

The GAE curriculum candidate was rejected: randomized repeat/crash outcomes deteriorated to 45/55 per 100 versus the retained policy's 48/52. No GAE continuation or lambda sweep is justified. Earlier diagnostics found that fixing command delay and response time together improved paired repeats from 54 to 66 per 100, while twelve actual command-history entries improved next-motion prediction about 23%. Simply adding those columns to the actor did not produce a reliable flight improvement. A recurrent estimator now tests whether explicitly carrying state across complete camera/command sequences yields better motion estimates. It is a distinct hypothesis, not a demonstrated collision fix.

The older prediction experiment used the **next displacement and current action**. Its numbers are not a valid comparator for a pre-action state estimate. This experiment fits both new estimators against the same explicitly shifted target.

## Inputs and target

Both models receive exactly the saved public 85-input observation before action `t`: nine camera-feature samples at the original documented lags, three actually issued prior commands, and the seen-ring flag. Full camera/history reconstruction is required to match every saved observation exactly. Neither model receives `action[t]`, simulator position, heading, velocity, randomized dynamics or any future observation. The GRU alone carries a hidden state from earlier public observations.

Saved `positions[t]` and `headings[t]` are after action `t`. For rows `t >= 2`, the supervised target is `positions[t-1] - positions[t-2]`, rotated into body axes using `heading[t-1]` and multiplied by 1,000. Axes are **forward, right, up**, in millimetres per completed interval. This is displacement, **not metres per second**: simulator control intervals vary. The first two public observations still initialize recurrence; their targets are masked because initial position/heading before the first stored transition are unavailable. Simulator truth supplies diagnostic labels only, never model inputs.

## Fixed comparison and data separation

- One unidirectional GRU with 64 hidden units and a linear three-output head versus a current-observation feed-forward `85 -> 128 tanh -> 128 tanh -> 3` model.
- One initialization seed, 11580000; Adam learning rate 0.0003, gradient norm clip 10, 3,000 updates per variant.
- Exactly the same precomputed eight-whole-episode minibatch at each update, sampled with replacement from fitting episodes. This matches update and episode budgets, not parameter count or FLOPs.
- Unroll each full episode, up to 600 steps. Hidden state starts at zero at every actual reset; no arbitrary truncated-backpropagation reset. Packed sequences and target masks exclude padding and the first two targets.
- Use only complete episodes wholly inside the original learned policy's first 200,000 warmup transitions from `clearance-reward-20261007/clearance/training_trajectories`. The crossing-boundary partial episode is excluded. The independently checked source contains 558 complete episodes and 199,929 stored steps.
- Exactly the prior motion audit's every-fifth-episode split: 447 FIT and 111 development-heldout episodes. All 198,813 valid target rows and episode ordering must match its saved split. This heldout set has been inspected in prior diagnostics and is not an untouched acceptance set.
- A separate final check uses the preserved 240 complete GAE trajectories, seeds 11241000–11241239, totaling 67,211 steps and 66,731 valid targets. These do not enter fitting, normalization, checkpoint selection or hyperparameter selection. Their reset distribution and generating policy differ from the warmup cohort; results are reported separately.
- Input moments use only actual FIT frames; target moments use only valid FIT targets. Padding, heldout and GAE data never set normalization statistics.

One CUDA process holds the existing exclusive training lock. The whole operation has a 30-minute wall-clock cap plus a `STOP` file check; either ends the comparison without a passing gate. The bounded `--smoke` uses two updates, the first ten warmup episodes and four GAE episodes, with a three-minute cap. It remains explicitly software-only regardless of errors.

```text
python -m tools.run_recurrent_estimator --out D:/DroneSoccerData/runs/recurrent-estimator-20261008
```

Use a fresh output directory. Smoke uses the same command plus `--smoke` and a different output directory.

## Prespecified final decision

Fit both models to their fixed budgets before any final performance evaluation. Report FIT, development-heldout and separate GAE metrics at the final models only. On **both** the development-heldout warmup cohort and separate GAE cohort, the GRU must:

1. Reduce per-axis displacement RMSE by at least 15% overall.
2. Reduce per-axis displacement RMSE by at least 15% while the current ring feature is not visible.
3. Increase visible-interval RMSE by no more than 5%.

An empty stratum, incomplete run, changed original policy or invalid evidence cannot pass. Failure closes this estimator variant; no architecture/seed sweep, best-checkpoint selection, extension or extra rollouts. A pass supports only a future review of explicitly estimated-state integration. It does not authorize actor integration automatically, promote a model, establish flight improvement or satisfy simulation acceptance.

These errors pool correlated frame samples within episodes. One initialization and one preserved domain-check cohort are not generalization proof or confidence intervals. Both variants receive the same data and masks so their comparison is controlled, but a prediction gain alone cannot establish better collision avoidance.

## Evidence and checks

Before fitting, save the protocol, source snapshots/hashes, original flight-policy snapshot/hash, every used trajectory and metadata input hash, episode split and counts, shared episode sampling schedule and FIT normalizers. Record Python/package versions, CUDA version, GPU identity and device; require all estimator parameters on CUDA. Preserve final model, optimizer and the variant's own end-of-training Python/NumPy/Torch/CUDA RNG states. Stop/failure saves partial models separately through the same final checkpoint names within the fresh run directory; completed checkpoints retain their original RNG state.

Final prediction files retain per-episode identities, seeds, row numbers, masks, visibility, physical-unit targets and predictions. `decision.json` binds all result/checkpoint artifacts and records every gate result. Inputs are rehashed after fitting; the selected original policy must remain unchanged. No simulator movement, physical command, publication, group master modification, or new flight-policy training is part of this comparison.

Unit tests cover target indexing and axes, current/future truth/action exclusion, exact issued-history validation, recurrent reset isolation and prefix causality, first-frame state contribution, padding/loss/evaluation masks, FIT-only normalization, shared FIT-only sampling, complete/cohort gate conditions, smoke rejection, STOP and wall-clock limits.

Implementation verification: all 15 focused unit tests passed on CPU. No CUDA smoke, model fitting, saved-trajectory evaluation or simulator rollout was launched during implementation. A separate read-only review found no causal-input, target-alignment, split, padding or gate blocker; execution still requires the root process preflight and bounded CUDA smoke.
