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
