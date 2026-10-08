# Selected policy for group review: 52% simulated crash rate

This branch contains the exact learned policy that repeatedly passes the ring in
simulation: **`trainedModels/seenflag-s1-level1.zip`**. It is not one of the failed
replacement checkpoints. The model is already tracked in Git and is included
when this branch is cloned.

SHA256: `8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef`.

The branch name `52%-crash-policy` describes the saved randomized simulation
result, not a measured physical crash probability. Physical flight is unvalidated.
This is an experimental handoff for group assessment, not flight approval.

## Saved results

100 deterministic episodes per suite, up to600steps in the2.05m cage:

| Suite | Clean first passage | Returned to front | Clean repeated mission | Crashes |
|---|---:|---:|---:|---:|
| Aligned |84|84|84|16|
| Nominal pad starts |80|80|74|26|
| Randomized |73|70|48|52|

Passages require the full collision box to clear the far noodle face. A later
crash still fails the mission even after several passes. See
[evaluation report](POLICY_FLIGHT_READINESS_20261006.md) and
[saved results](policy-validation-20261006/results.json).
The latest attempted improvements failed their checks; none replaced this model.

## Load the correct model

From the repository root, install `requirements.txt` into a Python virtual
environment. The model uses85inputs: nine eight-feature camera observations at
lags `(24,20,16,12,8,4,2,1,0)`, the last three actually issued four-axis commands,
and the seen-hoop flag. Do not pass the default109-input simulator vector directly
to it, invent unavailable state, or insert a suggested but unissued action into
the history.

```python
from pathlib import Path
import numpy as np
from src.policy_runtime import LearnedPolicy, SPECS

policy = LearnedPolicy(Path('trainedModels/seenflag-s1-level1.zip'),
                       SPECS['seenflag-s1-level1'])
policy.reset()  # repeat for every episode/session
previous_issued_action = np.zeros(4, dtype=np.float32)
# Supply the detector's actual eight-feature output:
# public85, suggested_action = policy.predict(detection8, previous_issued_action)
```

`LearnedPolicy` verifies the checkpoint hash and uses the correct history. Output
is normalized `[right, forward, up, clockwise_yaw]` in[-1,1]. Camera and telemetry
timestamps, command scaling/quantization, stale-data handling and operator stop
behavior need a validated physical integration; this loading example sends no
commands. The provided command-session tests use memory/loopback transports.

## Replay in the group simulator

```sh
python -m src.viewer --model trainedModels/seenflag-s1-level1.zip --port 8793
```

Open `http://127.0.0.1:8793/`. The viewer now defaults to this85-input policy,
applies its recorded camera history and reports full-body passage counts. It
connects to no drone. To reproduce the fixed development comparison, choose a new
output directory and run `python -m tools.validate_flight_policies --out <new-directory>`.
This evaluates both named historical policies; it is not the independent acceptance test.

## Mac camera preview package

The ready-built [Mac preview ZIP](../lab-tools/Mac-52-Percent-Policy-Preview-20261008.zip)
is included in this branch. Download the ZIP as a raw file and extract it on the
Mac. Its model identity and archive checksum are recorded in
[the selected-policy manifest](SELECTED_POLICY_20261008.json).
To build another copy from source:

```sh
python tools/build_mac_lab_preview.py --profile seenflag-s1-level1 --out ../Mac-52-Percent-Policy-Preview
```

The builder creates a portable folder and ZIP containing this exact model,
manifest, camera recorder and replay tools. Use a new output name each time.
Run `1-Setup.command` while on normal internet, then follow its `START-HERE.txt`.
The package displays camera images and policy suggestions, and records evidence;
**it does not run autonomous flight or send movement commands**. Keep motors off
for this preview. Supply an actual calibration with the builder's `--calibration`
option when available; otherwise its manifest explicitly labels team defaults.

For the group's proposed physical test, review this performance and integration
with the supervising operator first. No physical flight launcher is enabled by
this branch. Sustained simulation validation and fresh-seed acceptance have not
passed, and the hand-carried lab recording was not a flying-policy validation.

## What is preserved

This branch contains current research code, tests, detector/calibration work,
protocols, results and the original model. Compact recent result evidence is
under `experiments/faisal/evidence/share-20261008/`. Historical large trajectory
archives and failed-run checkpoints remain on the original workstation at
`D:/DroneSoccerData/runs`; they are not downloaded with this branch. Some research
runners/tests require those local archives. The selected-policy viewer and package
builder use the tracked model and repository files.
