# Ellipse arc coverage correction

The detector now measures occupied angular sectors in the fitted ellipse's normalized coordinates. The previous image-polar-angle calculation depended on aspect ratio and which axis tip contained the visible arc. The correction passed its predeclared retention gate, but **recorded-video detection recall did not improve**. The selected policy and simulator are unchanged; no physical-flight release is implied.

Only `src/detect.py::_coverage` changed. It rotates inlier pixels into the fitted ellipse axes, divides by their semiaxes and uses `atan2(y/b,x/a)`. A modulo operation also removes a duplicate sector at the -pi/+pi seam. It retains the existing9/36 sector threshold and all band, hole, backdrop, marker, minimum-size and temporal-confirmation checks. AST comparison found no other detector logic change.

Analytic100-degree arcs previously occupied8 versus14 sectors at opposite axis tips for aspect ratio0.5, and4 versus16 for ratio0.25. Normalized coverage removes that difference. This is an ellipse-parameter correction, not exact equivalence to perspective projection of uniformly sampled physical hoop points.

## Verification and retained limits

The117 focused tests passed, including50 added coverage checks for aspect/rotation invariance, short arcs, swapped ellipse axes, noninliers, seam wrapping, synthetic clipped rings, negative blue objects and confirmation behavior. The declared synthetic comparison retained the same5/7 ring detections and0/3 false positives. The two unresolved synthetic clipped cases remain failures; no additional parameter tuning followed.

After source freeze, exactly one full3583-frame sequential replay was scored against the existing48-frame visual review. All reviewed-image pixel hashes and source/label hashes were checked. Existing H264 decode warnings were present, but the decoded frame count and selected pixels matched the frozen reference; warnings were not used to excuse misses.

| Target | Original TP / FN / FP / TN | Corrected TP / FN / FP / TN |
|---|---|---|
| Ring presence | 7 / 7 / 0 / 33 | 7 / 7 / 0 / 33 |
| Front marker | 4 / 5 / 0 / 37 | 4 / 5 / 0 / 37 |

No previously correct positive frame was lost, no new positive frame was recovered, and no false positive was added. One ring and two marker labels remain unknown/excluded. All14 positive ring samples are clipped; there is no full-ring accuracy estimate. These are retrospective presence labels from a previously used recording, not independent pose, clearance, camera-performance or flight validation.

The predeclared gate required retaining every previously correct positive frame, adding no false positives, and preserving the synthetic safeguards. `frozen_decision.json` records `retain_math_fix` with no failure reasons. The correction was retained for its demonstrated geometry consistency, not as a solution to the seven recorded ring misses.

## Reproduction and preservation

Evidence directory: `D:/DroneSoccerData/runs/detector-ellipse-coverage-20261008/`.

- `source_before/` and `source_after/` preserve detector/dependency/test sources.
- `compare_coverage.py`, `protocol.json` and `synthetic-comparison.json` preserve the declared procedural cases, analytic results and gate.
- `final-manifest.json` records source freeze, exact changed function and focused test result.
- `frozen-report.json`, `review_frozen.py` and `frozen_decision.json` bind the real replay and retention decision.
- `package-verification.json` binds the delivered archive.

Detector hashes: original `62d32d5c315d07828eca8238a31dac2d62d8645c7615831f0f5a7359a45ed06a`; corrected `e8d3103cb6c24a06ba87d75374c563fdc537f3ca43e22a8f43b2315c7f19640e`.

Reproduce the tests from Publish with:

```text
python -m pytest tests/test_detector_coverage.py tests/test_detect.py tests/test_sim.py tests/test_camera_history.py tests/test_policy_validation.py -q
```

The v3 package is `C:/Users/Faisal/OneDrive/Documents/ChatGPT/RL 2/Mac-Policy-Preview-ReadyCheck-20261008-v3.zip`:34 files,3235093 bytes,SHA256 `1f1ea654601eb87a57c4ca256ec06947b557daa29fbc3da62430d92f15252238`. ZIP CRC,33 manifest hashes, source parity and executable LF launchers were verified. All17 bundled tests passed. Earlier archives and recordings remain preserved.

The package still uses the original85-input policy, labeled uncalibrated defaults when no measured calibration is supplied, and shadow/offline operation. It does not send learned flight commands to hardware. No GitHub publication or master-branch change occurred.
