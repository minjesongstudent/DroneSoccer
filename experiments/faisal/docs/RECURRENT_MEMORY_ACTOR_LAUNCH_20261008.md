# Recurrent-memory actor launch — 8 October 2026

The direct-memory experiment completed and failed its advancement gate; see [final result](RECURRENT_MEMORY_ACTOR_RESULT_20261008.md). It ran under the [frozen plan](RECURRENT_MEMORY_ACTOR_TRIAL_20261008.md).
This tests a 256-parameter correction driven by the frozen GRU hidden64, with
unchanged real public observations, source policy, simulator and outcome gates.
Earlier failed trials remain closed. This is not a claim of improved flight.

## Verified launch prerequisites

- 76 focused component, initialization and runner tests passed. Includes a600-frame
  CPU/CUDA hidden-state comparison at absolute1e-6 tolerance, exact episode reset,
  learned-only actions, zero-head parity, source freezing and final-only gates.
- Two initial test runs caught stale three-axis/seed/units fixtures in the new
  tests; corrected to the explicit150-input dimensionless-memory contract before
  any initialization flight or training. No outcome gate was changed.
- Initialization completed75 cases / 31634 frames in32.312seconds, all source
  actions and baseline trajectories bitwise exact. Decision SHA256
  `9eef62840f2d712f1c21940d15292d2c4321fca3f1e6acd865c1ea2461365106`.
- CUDA smoke completed2 episodes,33 transitions,1 accepted/0 rejected update in
  7.922seconds. Its complete evaluation/integrity pipeline passed software checks;
  the decision intentionally remains `software_only`/`passed:false`, with sole
  reason `Software-only smoke can never pass`. This is not an outcome qualification.
  All99 decision evidence hashes and43 source hashes verified; saved
  `launch_verification.json`. Decision SHA256
  `683d5544ba993e883231e4fd0efcfc8182666df287843804e61725e7da2473d3`.
- GPU admission: RTX4070Ti,8784MiB free,0% utilization; only read-only UI Python
  processes before launch. One standalone locked CUDA worker,0.15 allocator,
  one intra/inter-op thread, strictFP32 and TF32 disabled.

## Current run and next review

Output `D:/DroneSoccerData/runs/recurrent-memory-actor-20261008`;
stdout sibling `recurrent-memory-actor-20261008.stdout.log`; managed session51786.
Start phase: preparing_public_features. Read compact `status.json` or `decision.json` first on the
next heartbeat. Do not edit bound sources, tests or plan while running. Let the
bounded batch finish before detailed inspection; do not add another trainer.

Five48-episode FIT rounds, seeds11243500..11243739, algorithm11603000, total3600s
including preparation and evaluation. Stop/retention/integrity rules and full
100-per-suite final-only screen remain unchanged. Next review once on completion
or meaningful failure. Failure closes this representation/architecture/budget;
no intermediate promotion or unchanged extension. Passing permits only one fresh
paired development screen; full simulation qualification remains later.

Original85 remains selected. No physical flight validation, hardware operation,
master edits, remote publication or outside messages. Heartbeat remains ACTIVE.
