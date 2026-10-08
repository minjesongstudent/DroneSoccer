# Dynamics identification result — 8 October 2026

The fixed offline experiment failed its predeclared complete gate. Do not integrate this model into the actor, promote its successful delay output alone, extend training, or fit to the separate cohort. Original85 remains selected. No new flight performance was measured.

Both networks completed3,000 CUDA updates in76.437 seconds total. Integrity checks passed; no actor was loaded or changed and no simulator step was performed. Simulator resets supplied verified labels only. The actual input was recorded camera and issued-command history.

| Cohort / target | Feedforward RMSE | Recurrent RMSE | FIT-mean prior RMSE | Improvement vs FF / prior |
|---|---:|---:|---:|---:|
| Historical heldout, delay | 0.9819 steps | 0.6749 steps | 1.0486 steps | 31.3% / 35.6% |
| Historical heldout, response time | 0.1195 s | 0.0806 s | 0.1277 s | 32.5% / 36.9% |
| Separate GAE, delay | 1.0260 steps | 0.6951 steps | 1.1293 steps | 32.3% / 38.5% |
| Separate GAE, response time | 0.1304 s | 0.1225 s | 0.1351 s | **6.1% / 9.4% — failed** |

Each target on each cohort needed at least15% improvement versus feedforward and20% versus the FIT-mean prior. The response-time estimate did not transfer well enough to the separate recorded cohort. The partial delay result is informative but does not satisfy the declared joint condition. The data do not show whether the remaining error comes from insufficient information, estimator capacity, limited training coverage or another cause; none justifies an automatic retry.

Metrics weight episodes equally after32 observed transitions. Coverage was426/447 FIT episodes,106/111 historical heldout and197/240 separate GAE. Short episodes are excluded only from this prediction test, not from flight evaluation. Rounded-delay accuracy was62.7% and64.1% on the two non-fitting cohorts. These historically inspected cohorts are not independent flight acceptance, and no physical response-time calibration is implied.

Saved evidence: `D:/DroneSoccerData/runs/dynamics-identification-20261008`. Models, optimizer/RNG states, reset labels, normalizers, schedule, source snapshots and all predictions are preserved. `closure_review.py` independently verified51 evidence files,36 current sources and807 input files, then checked534,280 saved prediction rows, split/schedule membership, FIT target means, equal-episode metrics and every gate. It performs no inference, reset, training or flight. The review is complete; routine heartbeats should not rerun it.

Decision SHA256: `0f400c22c699a9a3b9b7f13ba9bfa3af8440485dea0798dbd01df42d28c8dbac`.

Close this fixed architecture/data/budget test. No trainer remains active. A subsequent experiment needs a distinct measured intervention; do not relax the joint gate, assume known simulator dynamics are deployable observations, or reduce randomization. Simulation acceptance remains incomplete and physical flight unvalidated. No master branch, remote repository, hardware or outside messages were affected.
