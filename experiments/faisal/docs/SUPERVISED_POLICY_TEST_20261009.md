# Experimental supervised policy flight with synchronized recording

The user explicitly requested real learned-policy flight commands and recording
for today's supervised lab test. This adds a separate physical-flight launcher;
the existing shadow recorder and blocked legacy transport remain unchanged.
No physical device was contacted or flown by the development tools.

The exact seenflag-s1-level1 85-input policy uses camera features and quantized
locally sent action history. No simulator truth enters its observations.
Takeoff/landing are operator/SDK actions; navigation RC comes from the model.
The model remains unqualified: 48 clean repeats and 52 crashes in 100 simulated
development episodes is not a physical crash probability or acceptance result.

A local browser hold control authorizes one takeoff and at most 10 seconds of
policy control. Release, lost fresh inputs, stalled inference or the deadline
latches landing with up to 20 seconds of retries. Late inference cannot reverse
that latch. Emergency motor cut is separate. Software cannot guarantee UDP
execution, landing or safety, and no global position/geofence is claimed.

Capture retains H264, packet offsets/timestamps, raw telemetry, SDK sends and
responses, policy proposals/results, full observations, calibration/model
provenance and final errors. Offline comparison joins only preceding fresh
telemetry; missing data remains missing. Logs distinguish socket acceptance from
physical movement; actual landing remains unconfirmed by software.

## Reproduce

From repository root, with the project Python dependencies installed:

```text
python tools/build_mac_supervised_flight.py --out ../Mac-Supervised-Policy-Test-NEW
```

Use a new output name. The builder preserves a distinct shadow source package,
adds the explicit physical launcher and tests, and verifies archive checksums.
Mac instructions are in the generated START-HERE.txt and the lab-tools README.

Artifact: Mac-Supervised-Policy-Test-20261009-v2.zip
SHA256: 4b10daaafad8e8b9126ca7d4408d4e70d650d9a5c8bc093ca9f5958249f75385
Model SHA256: 8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef

## Verification

- Generated package: `python -m unittest test_recorder test_live_preview test_calibration test_compare_session test_flight_control test_supervised_flight -q`: 37 passed, 11.896 seconds.
- Repository: `python -m pytest tests/test_mac_lab_package.py tests/test_learned_policy_session.py tests/test_command_rehearsal.py -q`: 124 passed, 32.46 seconds, exit 0. A native Windows WMI diagnostic appeared during collection; the run nevertheless completed successfully.
- Tests cover release during inference, stale results, lost inputs, takeoff
  cancellation, landing retries, no automatic restart, bad actions, logging and
  send failures, UI token checks and real-model quantized action generation.
- All hardware transport tests use mocks. UI uses loopback only. Model inputs
  in tests are synthetic. No actual drone movement or Mac launch was tested.
- Default camera intrinsics and vertical response remain imperfectly calibrated.
  Simulation acceptance and supervised physical validation are still outstanding.

Published only to the requested 52%-crash-policy branch; master is not modified.
