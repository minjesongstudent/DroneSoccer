# Ring-position residual actor launch — 8 October 2026

Final status: completed and rejected by the advancement gate. See [the result and saved-trajectory diagnosis](RING_POSITION_ACTOR_RESULT_20261008.md). No trainer remains active. The launch details below are historical. Original85 remains selected; simulation acceptance is incomplete and physical flight is unvalidated.

Plan: [RING_POSITION_ACTOR_TRIAL_20261008.md](RING_POSITION_ACTOR_TRIAL_20261008.md). Estimator/integration evidence: [RING_POSITION_ESTIMATOR_RESULT_20261008.md](RING_POSITION_ESTIMATOR_RESULT_20261008.md).

## Preflight

- 75 initialization flights and 31,634 frames exactly preserve original actions, trajectories and outcomes. Independent saved-evidence check verified 206 hashes and 34 current source files.
- 48 actor-runner tests passed, including actual curriculum reset endpoints and rejection of conflicting coordinator seed reservations. Component tests previously passed 136 checks, with six initialization-runner checks.
- First smoke `D:/DroneSoccerData/runs/ring-position-actor-smoke-20261008` failed before any flight/update because proposed FIT seeds were outside the curriculum's narrower interval. Its complete snapshot is retained. Corrected seeds are within the unchanged interval; no success criteria changed.
- Corrected smoke `D:/DroneSoccerData/runs/ring-position-actor-smoke-20261008-v2` completed in 8.172 seconds: two stochastic FIT episodes, 42 transitions, one accepted update, zero rejected. Software pipeline and frozen checks passed. Smoke outcomes do not establish improvement.
- Independently verified all 96 smoke evidence hashes and 40 current/snapshotted sources. Decision SHA256 `dd86c95c74b72e2fbd9c73a97dd2a938b81e8f1dce96078a2d5f79ce0509b13b`; verification saved as `launch_verification.json` inside the smoke directory.
- GPU admission: RTX4070Ti, 12,282 MiB total, 8,874 MiB free. Only the read-only viewer was otherwise running. Worker uses the project training lock, 0.15 allocator fraction, one CPU thread per pool, strict float32 and disabled TF32.

## Running batch

Output: `D:/DroneSoccerData/runs/ring-position-actor-20261008`; sibling `.stdout.log`; managed execution session94448. Read `status.json` or final `decision.json` first. Do not edit bound sources/tests/plan while active, start another trainer, or perform repeated detailed audits while healthy.

Exact command from repository root:

```powershell
& '../DroneSoccer/.venv/Scripts/python.exe' -m tools.run_ring_position_actor_trial --out 'D:/DroneSoccerData/runs/ring-position-actor-20261008' --initialization-report 'D:/DroneSoccerData/runs/ring-position-initialization-20261008/decision.json'
```

Run uses seeds11243200..11243439, algorithm11599000, five rounds of48 complete learned-policy episodes, full-horizon MC gamma1 and the unchanged approach/full-task reset curriculum. Only the residual learns. Maximum 3,600 seconds includes preparation, training and evaluation. STOP, nonfinite data, integrity failure, retention regression or100 consecutive rejected updates stop the trial.

After completion, inspect saved results once. Only final checkpoint5 may advance if randomized clean repeats reach at least53/100 and crashes fall to at most47/100, all suite retention and feature-use checks pass, and saved evidence verifies. Report first passages, returns, clean repeats and crashes. No intermediate or best-failing promotion, extension or extra-seed retry. A pass permits only one future fresh paired development screen; sustained validation and independent acceptance are still required. No physical hardware, master changes, publication or outside messages.
