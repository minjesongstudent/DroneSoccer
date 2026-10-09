# Synchronized policy and lab recording

The user requested recording drone movement signals while running the selected
policy, for later simulation comparison. This historical deliverable preceded the user's explicit physical-flight
clarification. See SUPERVISED_POLICY_TEST_20261009.md for the new physical test. It
implements recording plus inference only; the existing movement-command block
is unchanged. No physical device was contacted during development.

Recorder v3 adds a host monotonic anchor and timestamps to telemetry/video and
commands. The policy log records camera8 features, actor observation, raw model
output, usable suggestion, actual zero issued actions, analysis times and resets.
The immutable preview outcome captures decoding/dropped-packet/error information.
Repeated close no longer rewrites exported evidence.

After capture, comparison.csv joins each processed frame to the last telemetry
packet received at or before it, within 250 ms. Missing/stale and legacy
unanchored values stay empty. comparison-summary.json includes matching coverage,
quality counts, model/calibration provenance and input hashes. Raw SDK values
are preserved; no global position or response to unsent commands is inferred.
This is a data collection tool, not a flight readiness declaration.

## Artifact

Workspace sibling: Mac-Record-Policy-Compare-20261009-v3.zip (3,240,400 bytes,
37 files). Source checkpoint seenflag-s1-level1, 85 inputs, SHA256
8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef.
Intrinsics remain explicitly labeled team defaults (uncalibrated).

Build from repository root, selecting a fresh destination:

```text
python tools/build_mac_lab_preview.py --profile seenflag-s1-level1 --out ../Mac-Record-Policy-Compare-NEW
```

On Mac, run 1-Setup.command with internet, then join Tello Wi-Fi and run
2-Record.command with motors off. COMPARISON-GUIDE.txt explains outputs and limits.

## Verification

- Five package-builder tests passed, including calibration/hash preservation.
- Final generated v3 package: `python -m unittest test_recorder test_live_preview test_calibration test_compare_session -q`: 21 tests passed (3.197 s).
- Camera integration uses synthetic imagery and the real bundled policy; only
  loopback HTTP is exercised. SDK tests use mocks. No real telemetry fabricated.
- Causal joining tests cover future data exclusion, stale data, absent fields,
  legacy timestamp refusal and source/output preservation.
- An initial integration test stopped at an arbitrary three frames before a
  confirmed detection; it now waits for both required assertions, bounded to
  15 seconds. No detector or production timing thresholds were weakened.
- Initial cleanup exposed repeated close after temporary folder deletion;
  an outcome-saved flag now makes close idempotent without rewriting evidence.
- macOS shell launch and actual camera capture are not tested on the remote Mac;
  this chat has only a Windows execution host.

The shadow ZIP and instructions were published at 591f6d0 on 52%-crash-policy.
The shared recording source is included with the subsequent supervised-flight
deliverable. Master remains unchanged.
