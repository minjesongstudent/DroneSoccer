# Publication verification — 2026-10-09

- 44 recorder, camera, calibration, transport, watchdog and PC-policy tests passed on this Windows PC. The archive regression test was then added; all 5 composite/archive tests passed. See lab-runner-tests.txt and pc-policy-tests.txt.
- All 15 checkpoints loaded with the exact baseline + GRU + correction identities. Nonzero PC residual weights were verified; sampled sequential outputs match the direct training composite and differ from the baseline.
- Actual learned navigation outputs passed through the production RC encoder and flight state machine into a mocked command sink, with 89-input observations and complete identities recorded. No physical drone was contacted.
- First blind policy step requests landing without navigation RC. Camera/telemetry/operator/inference failures, late inference, malformed actions, logging/send failures and emergency/landing paths tested.
- OpenCV default 24-thread native GaussianBlur failed on this PC. Bounded single-thread processing passed detector and streaming tests. This is a runtime fix, not a change to ring thresholds or learned weights.
- Policy-only CPU timing: median 0.513 ms, max 0.750 ms over 100 steps. Excludes decoding, detection, UI and UDP. The Mac must run its own timing check.
- Tested environment: Python 3.12, Torch 2.11.0+cu128, NumPy 2.5.3, SB3 2.9.0, Gymnasium 1.3.0, OpenCV 5.0.0.93, PyAV 18.1.0. The lab adapter explicitly uses CPU float32; no training ran during organization.
- Historical Git model bytes were compared directly with the lab baseline hash. A complete pre-cleanup Git ZIP was CRC-verified before removing current-branch files.

Mac installation, real Wi-Fi, real actuator response and policy flight success were not validated by these tests. Recorded physical flights belong to the earlier baseline. No candidate passed simulation acceptance; no promotion is claimed.

The final ZIP was extracted to a separate D: directory and all **45 tests passed** there, confirming portable artifact paths and all-component loading. See extracted-package-tests.txt.
