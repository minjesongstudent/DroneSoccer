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
