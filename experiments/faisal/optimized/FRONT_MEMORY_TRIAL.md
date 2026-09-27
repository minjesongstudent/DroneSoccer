# Additive public front-sighting memory: bounded trial

Status: integration complete; 423 tests passed (41.80s, two existing Gymnasium warnings). CUDA baseline validation passed before any actor updates. The new115D contract preserves the original100 inputs and task index99. Old100D archives remain separate.

The recent/all-replay trial stopped at20k with24/25 centered and22/25 varied passes,0/50 return-development crashes, but zero new passes or safe front returns. Read-only continuation audit (`evidence/diagnostics/return-retrace-audit/recent-checkpoint-20000.json`) found all16 deterministic development continuations timed out. Stochastic continuations had29 timeouts and3 crashes in32 episodes. Later-return starts saw the back continuously and never crossed back to the front. Improved survival has not solved return flight.

The existing last-any-detection memory resets on every back detection. `src/front_memory.py` preserves all original100 inputs and appends15 separate features from the last genuine front-marker sighting. They contain the old detection, visibility history, age and nominal movement/heading integrated from issued commands. Requested-task index remains99; new features occupy100:115. No true pose, target phase, walls, velocities or rewards enter the helper. Command integration is uncertain and may drift physically. This adds an available history rather than a sensor.

The wrapper intentionally rejects old mid-flight reset environments: an old snapshot cannot reconstruct the earlier front sighting from its current view. Training requires freshly captured full memory histories from genuine learned-policy flights. Do not populate missing histories using simulator geometry or invented zeros and describe them as authentic.

Five focused tests passed: retaining the front anchor during back detection, reanchoring on actual front sightings without resetting nominal velocity, no fabricated earlier front view, reset/pickle consistency, unchanged100D prefix/rewards despite arbitrary diagnostic truth, and rejection of old return-practice reset histories.

`diagnostics/capture_front_memory_evidence.py` ran the preserved good100D actor on CUDA from normal starts with seeds2000000–2000015. The appended features did not enter its actions. Of eligible genuine entries,13/13 side returns and10/10 later returns retained a real earlier front sighting. At later-return entry, front-memory age averaged29.8 seconds while the original memory age was zero. See `evidence/diagnostics/front-memory-capture-v1.json`. This is an availability probe, not a training bank, demonstration set, learned improvement or proof that the extra memory is sufficient.

## Completed integration requirements

1. Preserve original100-feature meanings and task index99 explicitly in every sampler/protection mask. Make115D a separately versioned contract; old100D resume must remain valid only with its original class.
2. Transfer the chosen source actor and its original frozen reference with zero-initialized added first-layer columns. Test action-distribution equivalence for arbitrary appended values. Critics/replay/optimizers must be fresh; no failed-run weights should silently replace the selected source.
3. Capture new training/development banks by executing the preserved source actor from normal starts and saving the complete115D wrapper history on genuine phase entries. Keep2M training and8M development partitions. Validate actual saved observations, hashes and provenance; preserve old banks.
4. Integrate matching regular-start, practice, fixed evaluation and viewer wrappers; never reset new memory when restoring a mid-flight snapshot. Keep original physical/reward/success rules and50 distinct-flight return development protocol.
5. Test true terminal/time-limit handling,100D-prefix and frozen-reference equivalence, replay sequence/density correctness, new bank restore, actual CUDA live save/resume and incompatible contract rejection. Audit fixed baseline equivalence before actor learning.
6. Predeclare the bounded comparison and source. A logical control is the completed recent-replay run, with the same algorithm/sampling and only the memory addition changed. Keep40k ceiling,5k checks and existing20k/40k gates. Do not launch until integration and baseline tests pass.

No independent7M acceptance test has been used. Simulation and physical flight remain incomplete.

## Predeclared run and verification

Run `front-memory-20260927`, port8781, runs. Source remains runs/return-route-40k-20260927/best_model.zip (SHA884834222681c545623110f8b00fe6b7def32322ec069906d728a2373dd39337). Fresh critics/replay/optimizers; original frozen reference transferred independently. Both current/reference first layers use separate100-column and15-column sums with zero new columns: this preserves the original CUDA arithmetic at transfer while permitting learned use of the new features. Initial naive widened matrix multiplication caused small rounding changes and two extra long-flight collisions; that failed baseline remains preserved in diagnostics/front-memory-baseline-v1 on D:. The corrected result is v2, with all five whole-flight suites' episode records exactly matching the control. Return-development remains0/50 new/safe returns and1/50 crashes, identical outcome counts. Centered24/25 and varied23/25 passing are retained. This establishes a comparable start, not improvement.

New authentic banks on D:/DroneSoccerData/return-libraries: front-memory-train-v1 (93 snapshots, SHA2c798367ed060c468bd86e9947b3d7f1341810139e9d0542ddb6ba556b3f3804), front-memory-development-v1 (190, SHA50e300258a708d476deb130e0f6c48159cba653729bc9111911af3128ad574c7). Every original100-input prefix, physical position and RNG state matches its old-bank entry. New front-history is captured from normal source-policy flights, never reconstructed from truth.

The control is recent-retrace-20260927. Retain recent/all replay, eight-step Retrace, lambda.95, fixed entropy, reward, geometry, action limits, passing protection and600-step/two-pass mission success. Maximum40k, checks every5k;20k stops without >=5/50 new-pass or safe-front improvement sustained across two checks;40k stops without new-pass progress. First5000 steps refill and next5000 calibrate critics before actor updates. No scripted actions, new demonstrations or independent7M tuning. Only one production training worker.

Launch from project root:
```powershell
..\DroneSoccer\.venv\Scripts\python.exe -m src.live --steps 40000 --name front-memory-20260927 --port 8781 --eval-every 5000 --runs-dir runs --front-memory --source-checkpoint runs/return-route-40k-20260927/best_model.zip --source-config runs/return-route-40k-20260927/config.json --return-library D:/DroneSoccerData/return-libraries/front-memory-train-v1 --return-validation D:/DroneSoccerData/return-libraries/front-memory-development-v1
```
