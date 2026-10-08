# Ring-relative position comparison launched

The [fixed estimator comparison](RING_POSITION_ESTIMATOR_TRIAL_20261008.md) is running on one CUDA worker. It tests a specific additional quantity: current ring-relative position estimated from public camera and command history, including when the ring leaves view. It does not repeat recent-displacement estimation or resume a failed flight actor.

The supporting FIT-bank visibility count is reproducible in `D:/DroneSoccerData/runs/ring-position-evidence-20261008.py` and its JSON output. Of64 original learned flights,36 crashed;27 crashes occurred with no current ring feature,6 had all nine camera-history slots invisible, and6 of7 crashes after two passages were currently blind. This is descriptive evidence, not proof that blindness caused the collisions.

Five new label/indexing tests and15 existing estimator tests passed (20 total). The original all-zero synthetic fixture was corrected to construct valid camera histories through the real history builder; runtime history checks were preserved. No training ran before those tests passed. The CUDA software smoke completed two updates for each model and all final evaluations in6.844seconds. Independent verification passed all51 saved evidence hashes. Its software-only decision SHA-256 is `1ce839006f5602f28c1bc7a8e804791fa2ec449d55bf45c106ac34ce29bfc53c`. Short-smoke performance is not an estimator qualification.

Main output: `D:/DroneSoccerData/runs/ring-position-estimator-20261008`; managed session68826. Admission observed8847MiB free and no other trainer. Command from the repository:

```text
python -m tools.run_ring_position_estimator --out D:/DroneSoccerData/runs/ring-position-estimator-20261008
```

Both networks train sequentially in the same locked worker, on identical eight-episode minibatches and3,000 updates each. No simulator is stepped. All source movements are historical learned-policy trajectories. Inputs are public85 only; pre-action simulator truth supplies explicit supervised labels only. FIT-only moments and whole-episode boundaries are preserved. Original actor, existing displacement estimator and historical artifacts remain unchanged.

The fixed1,800-second total cap covers preparation, training and final checks; STOP is checked throughout. Read compact `status.json` first. Let a healthy batch finish before detailed analysis; inspect final `decision.json` once. No source/plan/test edits or second trainer while this run is active. The two final diagnostic cohorts must each show>=15% overall and invisible RMSE reduction, with<=5% visible degradation. Passing supports an integration review only, not automatic actor fitting or flight readiness. Failure closes the target/architecture/budget without a sweep or extension.
