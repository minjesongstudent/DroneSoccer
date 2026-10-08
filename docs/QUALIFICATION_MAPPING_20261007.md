# Legacy qualification compatibility, 7 October 2026

## Decision

The selected original85 policy can now be evaluated against the four original
qualification scenarios through a source-pinned, isolated compatibility runner.
This is **development only**, not current-group mission acceptance and not a
physical-flight release. The 5M fixed-validation and 7M reserved-acceptance seeds
are blocked in both the launcher and worker. There is no automatic promotion.

The first bounded software smoke used two episodes per suite, seeds 10,100,000
and 10,100,001. All eight episodes ended in a crash before any credited full-body
pass. This tiny sample verifies the integration and records the domain mismatch;
it does not estimate general performance or diagnose the cause of each failure.
The selected model was not modified.

| Original suite | Success / episodes | Full-body passes | Endings |
| --- | ---: | ---: | --- |
| original_centered | 0 / 2 | 0 | 2 hit |
| varied_approach | 0 / 2 | 0 | 1 hit, 1 left_cage |
| full_task | 0 / 2 | 0 | 2 hit |
| full_task_delay_200ms | 0 / 2 | 0 | 2 hit |

Evidence: `D:/DroneSoccerData/runs/qualification-legacy-compatibility-smoke-20261007/`.
`results.json` contains protocols, runtime versions and measured outcomes;
`manifest.json` binds source/model hashes; `execution.json` verifies frozen files
and original input files remained unchanged. Every episode has a JSON record,
initial geometry, and NPZ public-camera features, full actor observations, learned
actions, positions, headings and cumulative full-body passes.

Frozen-source reproduction in
`D:/DroneSoccerData/runs/qualification-legacy-compatibility-reproduce-20261007/`
produced byte-identical results, all four episode JSONL files and all eight NPZ
trajectories. `reproduction_checks.json` records the twelve per-file comparisons.
This was a software reproducibility check on the same seeds, not new evidence.
The smoke manifest SHA256 is
`4d0a1670108bd0e8b7a70e875f2b542832d200410661f37fbc79276453ddf3d1`;
both results files have SHA256
`fbfa80f7321d07953698971db2cda0aa437ec3d1112b505ccda450103e56e1a7`.

## Exact historical reference

Authority is the frozen `DroneSoccer-Optimized/src/unified_evaluation.py`,
`curriculum.py`, `sim.py`, `params.py`, and `evaluation_guard.py`. The live driver
uses 100 independent episodes per suite from seed 7,000,000 after sustained fixed
validation. This development tool neither invokes that driver nor makes that
acceptance claim.

All four historical suites use deterministic learned actions, a 0.1s time step,
no dynamics/measurement randomization, and the hanger. Nominal dynamics use
tau=.3s, max speed=.4m/s and max yaw=60deg/s. Reset velocity, yaw rate, command
queue and history use the native environment reset. Each suite shares its seed
sequence; these are paired scenarios, not 400 independent random draws.

| Suite | Reset and horizon | Success |
| --- | --- | --- |
| original_centered | Native LessonEnv stage0. Ring center `[.60*length,.50*width,U(.9,1.15)]`; front distance U(.50,.65)m; normal bearing +/- .08rad; lateral/vertical offsets +/- .01m; heading bearing +/- .02rad; independently perturbed outer diameters .45 +/- .03m. 150 steps maximum. | Native `passed` outcome, which terminates this lesson only after full-body clearance. |
| varied_approach | Native LessonEnv stage1. Same ring-center/diameter distribution; distance U(.55,.90)m; bearing +/- .25rad; lateral/vertical +/- .06m; heading bearing +/- .30rad. 150 steps maximum. | Same full-body single-pass outcome. |
| full_task | Native level1 corner start. Random corner inset .25m; ring XY rejection-sampled from the cage inset rectangle; ring height U(.6,1.4)m; normal facing away from pad with +/-75deg deviation; heading towards cage center +/-15deg; initial height .8m. One return-side route must fit. 600 steps. Zero command-delay steps. | Timeout with at least two full-body passes. |
| full_task_delay_200ms | Identical full-task reset/physics, except two delayed command steps, totaling200ms. 600 steps. | Same complete surviving mission. |

The historical full-body clearance includes the yaw-projected body reach, half
the noodle thickness **and an additional .01m**. Native ring/hanger collision,
wrong-way passage and leaving the inset cage terminate before success.

Fixed reference validation normally uses25 episodes per suite, seed5,000,000.
The required rates are85% centered,80% varied,80% full and80% delayed, meaning
22/25,20/25,20/25,20/25. Three qualifying checkpoint reports request the independent
100-per-suite test with85/80/80/80 successful episodes. Retention eligibility also
requires centered >=max(.70,baseline-.12) and varied >=max(.60,baseline-.12).
Candidate selection ranks full success, delayed success, then lower full crash
rate. These remain reference thresholds, not a claim that this smoke satisfied
the fixed-validation requirement.

## Why the current development suites are different

Current `tools/validate_flight_policies.py` evaluates aligned, nominal and
randomised600-step episodes. Those reports cannot be renamed or reused as the
four original suites. Important differences are explicit here:

| Property | Historical optimized reference | Current group simulator |
| --- | --- | --- |
| Nominal delay / tau | 0steps / .3s | 2steps / .55s |
| Minimum altitude | .3m | .5m |
| Camera pitch / forward offset | 0deg / .05m | 12.8deg / .045m |
| Guard box width / length | provisional .20 / .20m | .17 / .105m |
| Hanger half-width | .20m fixed | U(.025,.20)m in native layouts |
| Marker visibility angle | 70deg | 60deg |
| Ring XY | inset cage rectangle with rejection | central radius-.15m disc with rejection |
| Return-route fit | one side is sufficient | both sides must fit |
| Boundary | center leaving .15m inset is failure | box touching physical net is failure; floor clamps |
| Passage | native full body plus .01m clearance | native center crossing; PassageAudit adds full-body metric without that extra .01m |

The two guard-box descriptions conflict: optimized documents why inferring a
.105m length from a photographed .20m span was not established, while group
params labels that derivation measured. This requires an explicit evidence and
geometry decision before a unified physical claim. This tool preserves the
historical geometry and does not silently select the smaller envelope.

Nominal current-group physics already has200ms delay. Setting `delay_steps=2`
again would duplicate it, not reproduce the old no-delay-versus200ms comparison.
A future additional200ms stressor means four steps total and needs a distinct,
documented protocol version. Current measured geometry should not be rolled back
merely to obtain better legacy scores.

## Public actor bridge and source isolation

The world/action signs are compatible: action is `[right,forward,up,yaw_cw]`,
world heading is counter-clockwise, and camera normalized coordinates are
x-right/y-down/z-forward. Eight-feature meanings match. Observation dimensions do
not: the old native input is84, its mission wrapper is100, and selected85 has a
different final flag.

The worker takes only the newest eight public camera values from the unwrapped
legacy observation. The frozen group `CameraHistory` constructs the selected
lags `(24,20,16,12,8,4,2,1,0)`, the previous three actually issued learned actions,
and the group's **unseen flag**. No old mission flag, privileged position, ring
geometry, reward phase or simulator diagnostics enter the actor. No padding or
truncating model inputs is used. All movements are learned-policy outputs.

The model is `seenflag-s1-level1.zip`, SHA256:
`8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef`.

Sources and model are frozen before any episode. The worker runs in a separate
Python process with `-I`; only the frozen legacy and bridge paths are added.
Exact group source copies are retained. The executable bridge copies change only
two package imports from `src` to `group_bridge`; the manifest records these
changes and all original/derived hashes. The old environment receives explicit
`Camera()` defaults and refuses a calibration file in its frozen tree, preventing
ambient calibration from changing this historical protocol.

## Verification and reproduction

`tests/test_qualification_compatibility.py`: **12 tests passed**. Tests include
reserved/bounded seed rejection, overwrite prevention, source/model/calibration
integrity, public-history and learned-action parity against the actual group85
runtime, and outcome/count/termination parity with the **unmodified original
unified evaluator** on a nonreserved fixture. The reference evaluator's extra
wrappers are ignored by the bridge; only their original newest8 prefix enters the
selected85 runtime. No scripted flight actions are used in these episodes.

From the Publish checkout:

```powershell
& '..\DroneSoccer\.venv\Scripts\python.exe' -m pytest tests/test_qualification_compatibility.py -q
& '..\DroneSoccer\.venv\Scripts\python.exe' -m tools.evaluate_qualification_compatibility --out 'D:\DroneSoccerData\runs\qualification-legacy-compatibility-smoke-20261007' --episodes 2 --seed-start 10100000
```

The smoke directory now exists; the launcher deliberately refuses to overwrite
it. Reproduction from the preserved source, without using the current checkout:
create a new empty run directory, copy only the saved `source` directory and
`manifest.json` into it, then run the copied frozen
`source/qualification_legacy_worker.py --run <new-directory>` with the same
Python runtime and `-I`. Its hash checks execute before evaluation. Do not copy
existing episode/result directories. This reuses development seeds, not reserved
acceptance data, and must not be presented as new independent evidence.

## Remaining qualification work

1. Freeze a separately named current-group qualification protocol, including an
   explicit geometry decision, reset distribution, camera contract, delay
   stressor and collision-free full-body success definition. Keep the original
   thresholds and report differences rather than claiming equivalence.
2. Bind checkpoint/model hashes, environment sources and complete protocol to
   sustained validation. Do not count the same model relabeled at three steps as
   sustained training progress. The old guard counts step0 in its candidate
   streak; the new driver must apply the documented post-training requirement.
3. Run reserved acceptance only after the frozen candidate qualifies. Avoid the
   old path-only result cache: verify candidate/protocol hashes on resume.
4. Keep physical release separate. Neither compatibility success nor current
   simulation acceptance demonstrates real camera calibration, command timing,
   RC response or a supervised flight.

No training, physical operation, publication, or master-branch change occurred.
