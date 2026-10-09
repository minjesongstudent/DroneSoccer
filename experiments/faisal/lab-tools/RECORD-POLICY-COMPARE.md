# Supervised physical policy test and recording

[Download the physical-flight ZIP](https://github.com/minjesongstudent/DroneSoccer/raw/refs/heads/52%25-crash-policy/experiments/faisal/lab-tools/Mac-Supervised-Policy-Test-20261009-v2.zip)

**This new package sends real takeoff, learned-policy RC, and landing commands.**
It records video, telemetry, observations, policy proposals, commands sent and
transport responses together. It is an experimental supervised test interface;
simulation acceptance and physical flight validation remain incomplete.

1. Extract the new ZIP on the Mac. With normal internet, run `1-Setup.command`.
2. Use the enclosed lab cage and guards, with an operator watching the drone.
   Join Tello Wi-Fi and close other controllers/recorders.
3. Run `2-Supervised-Flight.command`, type `FLIGHT`, and check live inputs.
4. Check the supervision box and continuously **hold the green button** to take
   off and run the learned policy for up to 10 seconds.
5. Release, Escape, or LAND NOW requests landing. Keep the terminal/page open
   during landing retries and visually confirm motors stop. The separate red
   emergency control cuts motors and the drone will drop.
6. Keep the saved session folder and exported ZIP for comparison.

Run locally on the Mac beside the drone. Camera/telemetry/operator loss and
inference stalls request landing; a broken network cannot guarantee delivery.
Read `START-HERE.txt` for limits, recorded fields and uncalibrated assumptions.
Commands use the existing full team RC mapping, not a reduced-speed mapping.
Telemetry does not provide a measured global XYZ trajectory. Socket acceptance
is recorded separately from any evidence of physical execution.

Verification: 37 generated-package tests and 124 repository regression tests
passed on Windows using mocks, loopback HTTP and the real model with synthetic
camera inputs. No hardware was contacted; macOS and physical behavior remain
unvalidated.

ZIP SHA256: `4b10daaafad8e8b9126ca7d4408d4e70d650d9a5c8bc093ca9f5958249f75385`

Model SHA256: `8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef`

---

The old shadow-only download is preserved below. Use the new ZIP above for the
explicitly requested physical test.

# Mac lab recorder with the selected policy

[Download the ZIP](https://github.com/minjesongstudent/DroneSoccer/raw/refs/heads/52%25-crash-policy/experiments/faisal/lab-tools/Mac-Record-Policy-Compare-20261009-v3.zip)

**Runs learned-policy inference and records suggestions. Does not send policy
movement commands, take off, or fly the drone.** The selected policy has not
passed simulation acceptance or physical flight validation.

1. Download and extract `Mac-Record-Policy-Compare-20261009-v3.zip` on the Mac.
2. While connected to normal internet, run `1-Setup.command` once.
3. Join the drone Wi-Fi, keep motors off, and close other controllers/recorders.
4. Run `2-Record.command`. The camera window opens and records for two minutes;
   Ctrl+C stops early and saves. The policy computes suggestions during capture.
5. Preserve the session folder and exported ZIP. `comparison.csv` and
   `comparison-summary.json` are generated automatically for later review.

If Finder does not launch a command file, open Terminal, type `bash `, drag
that command file into Terminal, and press Return.

The package contains the exact `seenflag-s1-level1` 85-input policy selected on
this branch. It logs video, telemetry, full actor observations, camera features,
raw policy outputs, fresh suggestions, actually issued zero actions, timing gaps,
and model/calibration provenance. Telemetry is matched to preceding frame receipt
timestamps within 250 ms; unavailable fields stay blank. It does not reconstruct
a measured global XYZ trajectory or infer the physical effect of unsent actions.

Camera intrinsics are explicitly labeled team defaults (uncalibrated).
See `COMPARISON-GUIDE.txt` inside the package for all files and limitations.
The earlier 52/100 simulated crashes is a development result, not a measured
physical crash probability. Lab recordings do not change that qualification.

Validated on Windows with 21 package tests; actual Mac capture remains to be
verified in the lab. No hardware was contacted during packaging or testing.

ZIP SHA256: `fe5e1ea59b6ec5c8282fc4fc6bd89281181c98213c3b544b1eb85764d225d6f5`

Model SHA256: `8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef`
