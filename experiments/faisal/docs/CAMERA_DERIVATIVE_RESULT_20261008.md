# Camera feedback derivative check: gate not met

The bounded check completed in2.266seconds on1,245 saved poses from64 original
learned-policy FIT flights. It did not justify a new feedback-gradient training
trial. The policy and simulator are unchanged.

| Metric | Result | Frozen requirement |
|---|---:|---:|
| Exact source-camera forward outputs | 1,245/1,245 | all |
| Visible poses | 702 | reported |
| Stable visible poses | 604 | at least100 |
| Stable visible derivative agreement | 604/604,100% | at least95% |
| Visible poses crossing a processing branch | 98/702,13.96% | at most10% |
| Stable invisible poses with zero derivative | 540/540 | all |

All stable visible estimates agreed at the fixed coarse/half-size and independent
directional comparisons. Maximum relative errors there were1.044% for the
Jacobian and.734% for the directional estimate. This partial result does not
override the failed branch-coverage requirement. Processing-branch checks include
visible projected-arc membership, visibility, marker and output clipping; the
aggregate result does not isolate which branch caused every excluded case.

The hypothesis was that the failed clearance-only training neglected how motion
changes subsequent public observations. This test evaluated a proposed numerical
component for that feedback. It neither establishes that missing feedback caused
the training regression nor proves all camera differentiation impossible. It
closes this numerical route at the declared scales without a tolerance/stencil
sweep or actor fitting. Discrete switches and episode termination remain
unresolved even where a local derivative is accurate.

No neural network was loaded, updated or used for inference. No simulator step or
physical command was executed. Nearby poses were numerical camera queries only;
they were never flown, inserted into deployable observations or scored as passages.
The existing source camera's forward outputs were checked bitwise against saved
actual post-step camera features. Geometry, detector contract and actor input
layout were not changed.

Files: `src/camera_pose_derivative.py`,
`tools/check_camera_pose_derivative.py`, `tests/test_camera_pose_derivative.py`.
Eight focused tests passed, covering source output, state restoration including
exceptions, finite inputs, visible/invisible derivatives and failure gates.

Run: `D:/DroneSoccerData/runs/camera-pose-derivative-20261008`.
[Predeclared protocol](CAMERA_DERIVATIVE_PROTOCOL_20261008.md).
Original command from `DroneSoccer-Publish`:

```powershell
& '../DroneSoccer/.venv/Scripts/python.exe' -m tools.check_camera_pose_derivative --out D:/DroneSoccerData/runs/camera-pose-derivative-20261008
```

Existing results are preserved and refuse overwrite. Source snapshots, input
hashes, pose-level results and all numeric derivatives are saved in the run.
One verification checked45evidence entries,40source files and131input files.
Its `verification.json` is complete; do not repeat the review.
Result SHA256:
`7d36c2760962ed50a56c6a466d063ae3dd47bc0576f035f1cd8cd788096e7e8f`.

No trainer is active. Original policy remains selected. Simulation acceptance is
incomplete; physical flight remains unvalidated.
