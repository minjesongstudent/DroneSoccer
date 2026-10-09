# Supervised flight UI connection and preflight fixes

User reported ground state, blank readiness, 72% battery, height 0, temperature
68 C, and "UI disconnected; watchdog requests landing". The snapshot proves
telemetry was received and no readiness blocker was displayed then; it does not
prove the server was still running or that takeoff was requested. Actual Mac
Terminal/session evidence is pending. No physical failure cause is asserted.

Confirmed defects found in code:
- A failed poll wrote a disconnect warning that a successful later poll never cleared.
- The bounded 180-second session closed its server without a visible countdown
  or specific stop reason in Terminal, leaving a stale ground-state display.
- A rejected start invoked land even while still on the ground, closing the
  session. Pointer release without a valid hold also closed it.

v3 separates connection status from errors and clears it on recovery, disables
start with no current connection/readiness, shows the unchanged recording
countdown and prints the final stop reason. A server-confirmed ground preflight
rejection permits a new deliberate attempt. Release after a possible takeoff,
lost contact while held, Escape, LAND and the existing watchdogs retain stop
behavior; a stopped flight cannot restart. No policy, thresholds, RC scaling,
flight time, or model hash changed. No hardware was contacted.

Validation: 38 tests passed in the generated package (2.687 seconds), including
an HTTP preflight rejection followed by a valid mocked takeoff and monotonic
countdown coverage. `node tests/test_supervised_flight_ui.cjs` passed six real
inline-JavaScript handler scenarios with a mocked DOM/transport: unchecked
clicks, rejected-start retry, connection recovery, release during a pending
start, loss while held and no restart after stopping. No Mac/hardware validation.

Artifact: Mac-Supervised-Policy-Test-20261009-v3.zip
SHA256: c73e74ac2c24e50b9b95d0d5aff3557bdbfd1ea19f1b2245da817d72c362ccba
