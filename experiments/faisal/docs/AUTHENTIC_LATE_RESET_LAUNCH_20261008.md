# Authentic second-passage reset trial: launch record

Update after completion: the [main result](AUTHENTIC_LATE_RESET_RESULT_20261008.md) is closed without qualification. All five rounds finished in 623.984 seconds. Randomized clean repeats were46/100 with54 crashes versus the original48/52. All35 checkpoints restored exactly. No trainer remains; the launch details below are historical, not a live status.

The [fixed trial plan](AUTHENTIC_LATE_RESET_TRIAL_20261008.md) is now implemented. The earlier three credit variants remain closed. This trial starts a fresh zero residual and changes training exposure through authentic second-passage checkpoints; it does not resume their failed models.

## Software verification

The combined snapshot, checkpoint runner, estimated-motion runner and on-policy tests passed: **100 tests in 6.27 seconds**. Independent source review checked memory replay boundaries, the original horizon, suffix-only reward/gradient inputs, FIT allocation and final-only gates. Exact parity retains float64 environment rewards; the existing learning-target conversion stays float32.

CUDA software smoke `D:/DroneSoccerData/runs/authentic-late-reset-smoke-20261008` completed in 51.265 seconds. Eight fresh source flights yielded four eligible checkpoints; all four complete deterministic restored suffixes matched the original flights exactly. One fresh start plus one continuation yielded 1,049 training transitions and five accepted updates, with none rejected. The fixed small guard/final evaluations, save/reload and feature-interface checks completed. This is software evidence only, not a policy performance result or acceptance.

Root verified all 118 smoke evidence hashes, current bound sources, complete/frozen flags and the exact software-only decision. The result is preserved in `launch_verification.json`. Smoke decision SHA-256: `c07171aadf2e90a63574bb23caaefdce3857c1448b28f99e650def5bf689d050`.

## Main run

Started one managed CUDA experiment, session90484, launcher73648/interpreter88556, output `D:/DroneSoccerData/runs/authentic-late-reset-20261008`. Initial compact status was `preparing`. The sibling `.launch.json` records command/process information and `.stdout.log` records output. The project training lock excludes another trainer or cohort.

Recorded launch command from the repository (the runner refuses an existing output directory or previously reserved FIT seeds; do not treat a replay of this experiment as new independent evidence):

```text
python -m tools.run_checkpoint_reset_trial --out D:/DroneSoccerData/runs/authentic-late-reset-20261008 --initialization-report D:/DroneSoccerData/runs/estimated-motion-initialization-20261008/decision.json
```

The 3,600-second bound covers preparation, the fixed 64-parent collection, every restored suffix check, five training rounds and final evaluation. At least 16 authentic parents are required. There are 120 fresh starts and 120 continuations; descendants are explicitly correlated by parent. Failed bank coverage/parity, retention, budget or integrity closes the run. No source/plan edits while active, duplicate trainer, automatic retry, extra bank or relaxed gate.

At the user's request, let this bounded batch run unattended. Scheduled checks read `status.json` or `decision.json` first; review saved results after completion or a meaningful failure. The selected original85 policy, independent acceptance seeds and physical drone remain unchanged.
