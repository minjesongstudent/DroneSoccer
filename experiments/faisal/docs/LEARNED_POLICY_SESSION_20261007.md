# Learned-policy command session — 7 October 2026

The selected learned-policy command path can now be exercised entirely offline.
Physical flight is still blocked. The new adapter neither imports the labcheck
controller nor uses its scripted baseline. It has no takeoff or launch-on-connect
path, and it does not establish policy acceptance or flight readiness.

## Reusable session

`src/learned_policy_session.py` wraps the existing checksum-bound `LearnedPolicy`.
`from_checkpoint(path, profile)` requires an explicit supported profile, verifies
the unchanged model hash and observation/action dimensions, and creates separate
policy history for that session. Do not share a policy instance between sessions.

The default transport records commands in memory. The only other accepted
transport is the existing IPv4-loopback-only `LoopbackLink`. The explicit
`open_physical_transport(...)` interface always raises `PhysicalFlightBlocked`
before opening sockets or loading an SDK. There is no approval flag or override.

The session transitions from `disarmed` to `armed` to `active` only through two
separate calls, `arm()` and `start()`. Fresh camera, telemetry and operator
heartbeat inputs are required at both transitions. Neither transition sends a
command. `process_latest()` is the only policy RC path. Closing or stopping latches
the session; another arm/start cannot revive it.

`receive_detection(CameraSample(sequence, received_at, detection))` accepts the
same eight detector features from live or recorded producers. `receive_frame`
also accepts a raw uint8 BGR image and a callable detector exposing `.obs`; it
retains the caller's original timestamp. Neither API opens a camera or connection.
Sequences and receipt timestamps must increase. Repeated polls of one accepted
frame cannot update history or freshen its timestamp. The features are copied.

Timestamps refer to the original **host receipt** of the video, not measured
exposure time. Network age before host receipt is unknown. Live producers must
share the session's monotonic clock domain. Playback must use recorded receipt
times, recorded receipt ages and a compatible virtual clock.

Every subsequent model observation receives the last successfully handed-off,
quantized RC action, using the existing `(50, 50, 100, 70)` scale. The eight camera
features, documented lags, three-action history and seen-hoop flag stay unchanged.
Audit records distinguish proposed actions, issued actions and failed sends. A
failed neutral command retains the previous `last_issued`; it does not falsely
record that zero reached the transport. Successful handoff is not physical
receipt, execution or landing acknowledgment.

## Stops and fault coverage

An independent watchdog starts automatically when the operator starts the
session. Detection and inference execute outside the state lock, so a blocked
model cannot prevent an operator stop or watchdog stop. Only one inference may
run at once, and late results are discarded after a stop. Freshness is checked
again after inference. The supported transports send without blocking; supporting
any future transport would require a bounded send contract and independent
verification of its failure behavior.

The limits are 250 ms for video and inference, 500 ms for telemetry and operator
heartbeat, and 30 seconds for one session. Stale, future or duplicate timestamps,
regressing frame sequences, unhealthy telemetry, malformed detector/model output,
detector/inference exceptions, send failure, shutdown and operator stop all
prevent further motion. Space, Escape, L, Q and Ctrl+C request zero RC followed
by land. No motor-cut command is bound to ordinary stop. Landing is attempted
even if neutral RC fails; the stop latch is set before either send.

Manual `process_latest()` calls retain the existing 10 Hz cap and skip duplicate
frames. The optional `start_control_loop()` now supplies the software scheduler
after explicit `arm()` and `start()`. It spaces target ticks by 100 ms from each
tick's start, so variable inference latency does not add another 100 ms wait or
halve the command rate. Work spanning an interval skips to the next future tick;
late wakeups rebase on their actual start instead of producing catch-up bursts.
No camera timestamp is regenerated, and one cached frame cannot advance history
twice. The default manual/replay behavior remains unchanged.

The runner owns inference while enabled. Starting it during an in-flight manual
prediction is explicitly rejected; a later start between ticks is permitted.
Stop and watchdog handling remain independent of the control worker. Shutdown
waits at most one second across both workers. If a model remains blocked, its
daemon worker is reported as still running, but the closed latch prevents its
eventual result from issuing a command. There is no automatic restart.

This is a nominal 10 Hz software scheduler, subject to host wakeup jitter and
inference time. Physical camera timing and RC calibration remain unverified.

## Reproducible saved-feature rehearsal

The new command reads saved preview observations, verifies their metadata against
an explicit source profile, extracts that profile's documented newest feature
block, and rebuilds the selected model's own history. It never reuses the saved
suggested actions or zero-action history as newly issued actions. This is feature
replay, not image re-detection or a closed-loop flight simulation.

From the repository root, with the existing project Python environment:

```powershell
& '..\DroneSoccer\.venv\Scripts\python.exe' -m tools.rehearse_learned_policy `
  --preview '..\lab-data\20261006-081306Z-mac-preview-review\policy_preview.jsonl' `
  --source-profile lags60-scratch --profile seenflag-s1-level1 `
  --arm --start `
  --out '..\lab-data\learned-policy-session-repeat\recorded-memory-rehearsal.json'
```

Use a new output file. The CLI requires both explicit arm/start switches and has
no network destination option. Recorded video receipt times and ages are kept;
fresh telemetry and operator heartbeats are clearly labeled synthetic fixtures.
They provide no telemetry-health or operator-interface evidence. It uses a
virtual clock, so it does not measure inference speed or transport latency.

The recorded source contains 970 samples over approximately 101 seconds. One
bounded session consumed 289 samples and issued 225 learned RC commands in memory
before latching at its 30-second limit. It then issued neutral RC and land to the
memory sink. It did not automatically restart to consume the remainder. The
recorded arrival schedule plus the 10 Hz cap yielded about 7.5 policy commands
per second; this result does not validate regular 10 Hz operation. The hand-carried
source images do not respond to the generated memory commands, so this result is
not closed-loop performance evidence.

The full local report is
`lab-data/learned-policy-session-20261007/recorded-memory-rehearsal-v2.json`
(beside the repository). It includes source/model/code hashes, producer
timestamps, model observations, proposed actions, quantized actions, all command
attempts and stop reason. Original recordings, models and prior evidence remain
unchanged.

## Short software scheduler timing check

`tools/check_policy_scheduler.py` loads the real selected checkpoint, feeds
synthetic eight-feature camera samples plus explicitly synthetic telemetry and
operator heartbeats, and runs the new scheduler against memory only. It uses the
real host clock, opens no camera or network, and accepts a bounded 0.5–5 second
duration. The two-second check is reproducible with:

```powershell
& '..\DroneSoccer\.venv\Scripts\python.exe' -m tools.check_policy_scheduler `
  --profile seenflag-s1-level1 --seconds 2 --arm --start `
  --out '..\lab-data\learned-policy-session-repeat\scheduler-timing.json'
```

The current Windows host issued 19 learned memory commands in two seconds.
Tick-start intervals were 109–110 ms, median 109 ms, approximately 9.2 Hz between
ticks. Thus the configured 100 ms period is not a claim of exact measured 10 Hz:
host wait/wakeup granularity added roughly 9 ms per tick. No work intervals were
skipped, no watchdog fault occurred, the stop sent neutral/land, and both producer
and control worker exited. The observed work durations were 0–16 ms at the host
clock's available resolution; zero in this report does not mean free inference.

The report is
`lab-data/learned-policy-session-20261007/scheduler-real-clock-v2-final.json`.
It records actual tick times, all model observations/actions and memory commands,
source hashes and the explicit synthetic-input limitation. This is a measured
software check on the current host, **not** actual Mac timing, detector accuracy,
closed-loop performance or flight acceptance. The earlier recorded-feature
replay still produces 225 commands because its default manual polling is kept.

## Verification and remaining release conditions

The relevant suite passed 132 checks: 96 session/replay/scheduler checks plus 36 existing
command/history/profile checks. Tests cover separate arm/start, stale inputs,
sequence regressions, malformed values, inference and detector failures, heartbeat
loss, timeout, operator stops, failed neutral/land sends, context shutdown, late
results after stops, real watchdog behavior during blocked inference, memory-only
replay and IPv4 loopback wire order. Scheduler tests explicitly cover varying
inference delays without half-rate gating, skipped intervals without bursts,
duplicate-frame suppression, failed worker startup, manual-to-scheduled ownership,
independent watchdog/operator stops and bounded shutdown during blocked inference.
Both unchanged real checkpoints were also
loaded and run through the session, checking their observation sizes and actual
quantized action histories. Fault-injection model doubles are confined to tests;
they are not policy-success evidence.

```powershell
& '..\DroneSoccer\.venv\Scripts\python.exe' -m pytest `
  tests/test_learned_policy_session.py tests/test_command_rehearsal.py `
  tests/test_policy_validation.py tests/test_camera_history.py -q
```

The [6 October readiness assessment](POLICY_FLIGHT_READINESS_20261006.md) still
applies: policy acceptance, calibration and real stop/landing verification are
incomplete. The software scheduler is implemented and checked offline; a separately
reviewed physical transport remains blocked until release conditions are satisfied. This work
does not run training, operate hardware, publish a package or resume automation.
