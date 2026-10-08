# Bounded GAE advantage experiment

Status: **rejected and closed** on 8 October 2026 (Sydney). All five rounds completed; no policy promotion, continuation, lambda sweep or fresh cohort is authorized for this variant.

The final checkpoint failed the preregistered development screen. It increased randomized first passes by one, but clean repeats fell by three and crashes rose by three. Aligned and nominal aggregate success/crash counts were unchanged. The original model remains selected.

| Suite, 100 episodes each | Original first / return / repeat / crash | GAE95 final first / return / repeat / crash |
|---|---|---|
| Aligned | 84 / 84 / 84 / 16 | 84 / 84 / 84 / 16 |
| Nominal | 80 / 80 / 74 / 26 | 80 / 80 / 74 / 26 |
| Randomized | 73 / 70 / 48 / 52 | 74 / 72 / 45 / 55 |

Training used 240 complete learned episodes and 67,211 transitions, with 264 accepted updates and zero rejected updates. The final model hash is `c206dba6e47052cc8ce5ced95eb92b0cbc99ade7040712fbafb2c9c485a3ed9e`. The saved decision is `reject_close_variant`: randomized repeat loss and crash rise exceed the allowed two, and neither required five-count improvement was met. Source policy and all 15 decision evidence hashes were verified unchanged during closure.

Paired saved outcomes show four randomized repeat gains and seven losses. Randomized ring-contact endings rose from29 to33; net contacts fell from23 to22. Crashes before any full passage fell from27 to26, after exactly one remained15, and after two or more rose from10 to14. This locates the regression later in the mission; it does not establish the causal control error or justify another advantage-estimator variant. The reproducible `closure_review.py` and its `closure_review.json` are preserved inside the run directory and use only saved episode logs, with no new rollout or selection. No training process remained at the closure inspection.

The root integration check completed `gae95-curriculum-smoke-20261007`: two complete learned episodes, 632 transitions, three accepted updates, zero rejected, source model unchanged. Its final decision correctly reports `software_only`, with no evidence-validation failure. This verifies execution, not improved flight reliability. The 33 targeted training/screen tests passed before launch.

Main run: `D:/DroneSoccerData/runs/gae95-curriculum-20261007`. Launch record and redirected stdout/stderr sit alongside that directory. Historical launcher PID12720 and worker PID88512 formed one job. The completed protocol records CUDA on NVIDIA GeForce RTX4070Ti; stderr was empty. All five rounds of48 episodes and bounded evaluations completed within the training budget. `result.json` and `final_screen_decision.json` contain the completed outcome. The project heartbeat remains ACTIVE for distinct justified work, not a restart of this failed variant.

The completed backtracking curriculum trial accepted 267 optimizer updates but achieved only 49 randomized clean repeats and 51 crashes per 100, versus the original policy's 48 repeats and 52 crashes. Its fitted public-observation value baseline explained only 0.062–0.079 of return variance over the five rounds. Earlier complete-episode gradient checks also showed weak direction consistency. This motivates one estimator change; it does not identify a confirmed episode-accounting bug or guarantee better flight.

Change only the actor advantage estimator to generalized advantage estimation (GAE) with gamma 0.995 and lambda 0.95. The existing value network is still fitted to the complete Monte Carlo reward-to-go targets with the same 100 minibatches. After fitting, freeze its predictions for the complete rollout. For each episode separately, compute `delta_t = reward_t + gamma * V_(t+1) - V_t`, then `A_t = delta_t + gamma * lambda * A_(t+1)`. Both a crash and the finite 600-step cutoff end with zero tail value and zero tail advantage. No bootstrap reaches across an episode reset. The actor uses the same normalized advantages and clipped surrogate thereafter.

GAE trades variance for bias when the value approximation is imperfect; the weak baseline could therefore make performance worse. This is an empirical test, not a claim that reduced estimator variance establishes mission success. Reference: [Schulman et al., High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).

## Fixed protocol

Restart from the original 85-input learned policy. Keep the latest approach curriculum distribution, randomized simulator, full-clearance rewards, MC value fitting, original source KL limits 0.01 mean/0.1 max, 8,192 FIT guard observations, backtracking, learning rate 1e-5 and one actor epoch. Retain five rounds of 48 complete episodes, at most 144,000 learned transitions and a one-hour training cap plus bounded evaluations. Fresh FIT reset seeds are 11241000–11241239; changed samples are explicitly distinct from the previous trial. All simulator movements are learned actions on actual public camera/history observations. No development or acceptance trajectory becomes training data.

The option is accepted only with this paired configuration:

```text
python -m tools.run_onpolicy_trial --out D:/DroneSoccerData/runs/gae95-curriculum-20261007 --approach-curriculum --backtracking --actor-lr 1e-5 --episodes-per-iteration 48 --actor-epochs 1 --advantage-estimator gae95
```

Default `monte-carlo` behavior remains unchanged. `--smoke` is a separate software check and does not qualify the model. Every GAE round additionally saves its actual rewards, frozen baseline values, episode lengths and unnormalized advantages in `gae_inputs_NN.npz`. Existing actual trajectories, complete MC targets, normalized advantages, likelihoods, model checkpoints, RNG and source hashes remain preserved.

Before collecting episodes, the runner snapshots `../lab-data/policy-validation-20261006/results.json`, its `protocol.json`, and the original 85-input policy's three saved episode logs into `screen_reference/`. It verifies the original model hash, observation lags, suite seeds, finite horizon and exactly 100 unique layouts per suite, and recounts first passages, repeats and crashes against the saved report. No baseline counts are hardcoded into the screen. The run protocol binds the frozen reference manifest hash and records Python, Torch, NumPy, Stable Baselines3 and Gymnasium versions, the Torch CUDA version, GPU name and actual model device. Actor and value parameters must both be on CUDA.

Review of the historical baseline sources confirmed identical simulator, parameters, observation contract, passage audit and policy runtime hashes. The evaluation helper's later `diagnostic_dynamics=None` option is unused here; CameraHistory's later optional four-action mode leaves the default three-action behavior unchanged. These reviewed additions do not replace the baseline with a different experiment.

## Go/no-go set before the experiment

Use the existing 25-per-suite retention stops during training. Assess only the final checkpoint on the unchanged 100-per-suite development sets; do not select a favorable intermediate checkpoint.

Continue to one fresh paired development comparison only if all these conditions hold relative to the original policy:

- Randomized clean repeat improves by at least 5/100: at least 53, versus 48.
- Randomized crashes fall by at least 5/100: at most 47, versus 52.
- No suite loses more than two first-pass or clean-repeat successes, or gains more than two crashes.

Otherwise reject and close this variant. No lambda sweep, unchanged extension, or extra fresh cohort to seek a favorable result. Passing this development screen authorizes only a single independent paired development comparison, with its existing paired significance and retention gates. It does not promote the policy or replace the four qualification suites, sustained validation, reserved independent acceptance, or physical verification. Original policy remains selected until sufficient evidence supports a later decision.

`final_screen_decision.json` records the machine-readable decision, reasons, recounted reference/candidate counts and SHA-256 evidence bindings. Only `final_development/` from the last saved checkpoint is eligible; favorable intermediate metrics are ignored. The screen recounts exactly 100 unique candidate layouts per suite, checks both saved summaries, verifies candidate/model identity and untouched frozen baseline evidence, and rejects an altered original source or any retention-regression stop. Smoke runs always receive `software_only` and `passed: false`.

Tests cover lambda-one equivalence to complete MC returns minus frozen values, isolation between episodes, zero crash/timeout tails, invalid inputs, restricted CLI configuration and evidence-bound final screening. The combined on-policy/curriculum/constrained-optimizer test command passed 33 tests; compilation also passed. A read-only recount of the actual baseline logs reproduced aligned 84/84/16, nominal 80/74/26 and randomized 73/48/52 first/repeat/crash counts. No CUDA smoke or training was launched during implementation. No physical operation, publication or group master modification is part of this experiment.
