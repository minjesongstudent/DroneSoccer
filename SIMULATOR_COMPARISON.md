# Simulator selection - 6 October 2026

The repository-root group simulator is the default for new work on Faisal's
branch, based on master revision 71c577e. The prior experiment simulator and all
models remain under experiments/faisal/optimized for reproducibility.

## Comparison

| Area | Group simulator | Preserved optimized simulator |
|---|---|---|
| Camera tilt | 12.8 degrees, recorded as measured on 1 October | 0 degrees, unmeasured |
| Velocity response | 0.55 s, based on recorded step measurements | 0.3 s, unmeasured |
| Nominal command delay | 2 steps | 0 steps |
| Minimum altitude | 0.5 m, recorded lab observation | 0.3 m, unmeasured |
| Detector | Live ring and front-marker detector plus lab tools | Separate experimental camera bridge |
| Actor interface | 109 inputs: camera history, action history, unseen flag | Preserved recurrent policy uses 119 inputs with experimental memory |
| Passage scoring | Centre crosses ring plane while collision checks pass | Pending crossing until body clears the plane |
| Collision envelope | 0.17 by 0.105 m; length derived from assumed diagonal | Conservative provisional 0.20 by 0.20 m |

The group implementation is the better-supported base for measured dynamics and
shared lab development. It is not uniformly stricter or independently validated
against real flight. In particular, the diagonal-derived length needs direct
measurement; its smaller box and centre-crossing count must not substitute for
full-body acceptance. Camera intrinsics remain defaults, and a shared response
constant does not capture every axis's measured response.

The older simulator remains useful for its strict passage evaluation and
historical reproduction. Selecting master does not change historical models or
their scores. No cross-simulator policy ranking is claimed: physics, observations,
layouts and success metrics differ. No padding, truncating or fabricating model
inputs is used to make old checkpoints load.

## Run the selected simulator

From the repository root, with requirements.txt installed:

    python -m src.viewer

Open http://127.0.0.1:8793/. This uses the compatible 109-input learned SAC model
trainedModels/lags60-scratch.zip in the actual group HoopEnv, nominal level 1,
with the hanger enabled. Replay seeds begin at 9200000. Every movement comes from
the model. It is a demonstration, not a training run or acceptance evaluation.
The model is selected for compatibility, not superior performance: its published
report says it underperformed the baseline. The stronger reported seenflag model
uses 85 inputs and cannot be dropped into the current 109-input environment.
The viewer checks shape compatibility and reports errors instead of adapting inputs.

The viewer only serves loopback HTTP and has no drone command connection.
Training and unattended automation remain paused.

## Verification

The merge preview and actual merge were conflict-free. The merged group's
simulation, training, detector and baseline suites passed: 61 tests, with two
existing warnings about unbounded observation-space limits. These baseline tests
test code; scripted controllers are not evidence of learned-policy success.

Existing full-body acceptance requirements remain unmet. Before claiming progress
against them, port and test a separate strict metric and evaluate compatible
learned policies under common conditions. No physical flight validation is claimed.
