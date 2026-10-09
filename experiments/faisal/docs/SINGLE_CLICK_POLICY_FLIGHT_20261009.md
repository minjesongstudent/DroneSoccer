# Single-click continuous policy flight

User explicitly requested one press to take off and run the policy, then rejected
the previously imposed ten-second limit. This deliverable changes that interface
and removes both policy-duration checks. A policy is a continuously evaluated
controller, not a script with a known end. No physical ring-passage completion is
invented. Recording continues with policy inference until operator or safety stop.

The UI sends browser-liveness heartbeats automatically after the explicit click;
mouse release does not stop flight. LAND NOW, Escape, loss of focus/visibility or
connection, and server safety checks still stop it. A double click sends no second
takeoff. A late start response cannot override landing. No restart after stopping.
The three-minute idle limit now applies only in ground state, checked under the
same lock as start to avoid an idle-expiry/takeoff race. The existing 12-second
takeoff wait and 20-second landing retry bounds remain unchanged.

Camera, telemetry, battery, temperature, height, inference, recording-failure
checks, full RC mapping and the selected model are unchanged. UI heartbeat is
browser liveness, not evidence of human attention. Operator supervision remains
required. Simulation acceptance and physical flight validation remain incomplete.

Tests: ten mocked-DOM/HTTP JS scenarios passed using the real UI script. Forty
generated-package tests passed (4.880 seconds), including policy steps beyond
10, 180 and 360 seconds with fresh inputs, takeoff-vs-idle checks, and manual
landing afterwards. These are software fault-injection tests, not flown results.
The final named ZIP has byte-identical Python, launchers and policy files to the
tested v6 staging package; only explanatory guide wording was subsequently cleaned.
All archive checksums verified. No physical hardware was contacted.

Build: python tools/build_mac_supervised_flight.py --out ../Mac-Single-Click-Policy-Flight-NEW
Artifact: Mac-Single-Click-Policy-Flight-20261009.zip
SHA256: e200ca3811dcefd5388815bc1506d45bea9667bd9edd64b7abd92b33a7166155
Model SHA256: 8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef

Run 1-Setup.command, then 2-Supervised-Flight.command. Type FLIGHT to connect.
Check supervision and click Take off + run policy once in its newly opened page.
Only group branch 52%-crash-policy is updated; master is untouched.
