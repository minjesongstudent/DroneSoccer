# Ring-position residual actor result — 8 October 2026

The experiment completed but failed its unchanged advancement gate. Retain the original `seenflag-s1-level1` policy. Do not extend this variant, select an intermediate checkpoint, run a fresh development screen or claim physical-flight readiness.

All five CUDA rounds completed in 397.828 seconds: 240 distinct FIT episodes, 64,021 transitions, 253 accepted updates and zero rejected updates. The original actor and ring-position estimator stayed frozen. All retention stops, source checks and final feature-use checks passed, but the final randomized flight improvement requirements did not.

## Final 100-case results

Counts are first full-body passage / return to front / clean repeated episode / crash. Clean repeats require the unchanged full horizon and safety criteria.

| Suite | Original | Final ring-position actor |
|---|---|---|
| Aligned | 84 / 84 / 84 / 16 | 84 / 84 / 84 / 16 |
| Nominal | 80 / 80 / 74 / 26 | 79 / 79 / 73 / 27 |
| Randomized | 73 / 70 / 48 / 52 | 75 / 73 / 46 / 54 |

The randomized gate required at least 53 clean repeats and at most 47 crashes. Two additional first passages and three additional returns did not compensate for two fewer safe repeat flights.

Paired saved trajectories show six randomized repeat rescues and eight losses. Seven added crashes occurred after two or more full-body passages; one occurred after one passage. Nominal results had one rescue and two losses, including one new crash before the first passage. This is a reliability regression, not evidence that a successful displayed flight qualifies the policy.

Final mean/max source KL was 0.0001385 / 0.0212395. No update-rejection stall occurred. The residual genuinely depended on the estimate: FIT source/shuffle action differences were 0.000623/0.000804; historical heldout differences were 0.000638/0.000811. Removing estimates exactly recovered source actions. These checks establish correct feature use, not useful flight control.

## Saved-state diagnosis and limits

A read-only calculation compared the actual saved estimates with correctly timed simulator truth. No model inference, simulator step or new flight was performed. Truth was used only for diagnostics.

Across final randomized episodes, position RMSE was 55.8 mm per axis overall and 79.6 mm while the ring was invisible. Invisible-ring error was 122.3 mm on trajectories that eventually crashed versus 55.8 mm on trajectories that survived; the crash-trajectory 95th-percentile vector error was 408.1 mm. These are outcome-conditioned, state-distribution-dependent summaries. They do **not** establish that estimator error caused the crashes or justify fitting to development trajectories.

FIT round invisible RMSE was 68.9, 72.0, 59.1, 61.5 and 77.2 mm per axis. Complete-horizon survivors were 13, 17, 17, 12 and 12 of48 stochastic practice episodes. Different rounds use different layouts; these counts are not an independent learning curve. The results support investigating how estimates behave near difficult FIT states, but do not authorize an unchanged extension or arbitrary larger policy steps.

## Preservation and reproduction

Run: `D:/DroneSoccerData/runs/ring-position-actor-20261008`.

- All source snapshots, plans, input hashes, initial and intermediate heads, final head, value/optimizer/RNG states and full trajectories are preserved.
- `decision.json` SHA256: `6491cbf539b60cc66d805aedea79a7e6e683054fdb6514b9933fab9cc6e5e111`.
- `closure_review.py` independently checked all 1,283 decision evidence hashes and 2,191 unique files, reconstructed paired outcomes and feature-use metrics, and checked public-feature/action/reward alignment in all240 FIT trajectories. Its report is `closure_review.json`. The script initially used a snapshot folder without baseline NPZ files; it was corrected to read the recorded original baseline path. No run artifact or outcome was changed.
- Reproduce the saved-data review with the project Python interpreter and that script. No training or hardware access is required. Do not repeat this completed review during routine heartbeat checks.

This quantity/architecture/training-budget variant is closed. Estimator accuracy remains a valid offline improvement, but its integration did not deliver safer flight. No policy was promoted, no acceptance seeds were consumed, no physical hardware was operated and no master branch or remote repository was modified. No trainer remains active after closure. A further training run needs a distinct, measured intervention; unavailable laboratory measurements are not an explanation for this simulation failure.
