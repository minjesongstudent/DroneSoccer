# Dynamics identification launch — 8 October 2026

Final status: complete, joint advancement gate failed. See [the final result](DYNAMICS_IDENTIFICATION_RESULT_20261008.md). No trainer remains active; the launch details below are historical. The failed ring-position actor remains closed and original85 remains selected.

Plan: [DYNAMICS_IDENTIFICATION_TRIAL_20261008.md](DYNAMICS_IDENTIFICATION_TRIAL_20261008.md). Runner: `tools/run_dynamics_identification.py`.

Fourteen focused checks passed: reset-label provenance without simulator movement, corrupted first-observation rejection, normal/curriculum resets, no future access, no seed/truth input columns, burn-in/padding, equal-episode weighting and all final gate conditions.

CUDA software smoke `D:/DroneSoccerData/runs/dynamics-identification-smoke-20261008` completed in1.672 seconds with two updates per model. Both models/evaluations and integrity checks completed. Its accuracy failed the final criteria as expected for an untrained software probe; it does not qualify an estimator. All51 evidence hashes and36 current sources verified. Decision SHA256 `9b47ebaf59ae77f32fcbc9dff6ad5e9fac0c434ee1fa4a2b8a83c1778a411525`; verification saved inside that run.

Main output: `D:/DroneSoccerData/runs/dynamics-identification-20261008`, sibling `.stdout.log`, managed execution session94849. GPU admission: RTX4070Ti with8,859 MiB free. One worker, sequential FF andGRU, 0.15 allocator fraction, one thread per CPU pool, strict float32/no TF32. No other trainer/cohort is active.

```powershell
& '../DroneSoccer/.venv/Scripts/python.exe' -m tools.run_dynamics_identification --out 'D:/DroneSoccerData/runs/dynamics-identification-20261008'
```

Read compact `status.json` or final `decision.json` first. Let a healthy batch finish without repeated logs/tests/audits. Do not edit bound sources/tests/plan while active. The main budget is3,000 updates per model and1,800 seconds including preparation/evaluation. No actor loading, simulated steps, hardware, master edits or publication occurs. Verified simulator resets provide labels only; the networks consume actual public85 history.

After completion, inspect results once against the declared per-parameter/per-cohort gates. Passing permits only a separate integration review, not flight-policy training or qualification automatically. Failure closes this fixed test without extension or heldout fitting. Physical flight remains unvalidated.
