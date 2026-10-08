# Policy and command-path validation — 6 October 2026

**Physical flight remains disabled.** The camera-to-policy preview works on the
Mac, but neither tested checkpoint is sufficiently reliable for a lab flight.
This assessment does not claim completion of the repeated-passage mission.

## Simulator comparison

600 deterministic learned-policy episodes: two unchanged checkpoints, three
100-episode suites, up to 600 steps per episode. The current group simulator,
2.05 m cage and hanger were retained. Every movement action came from the model.
No scripted controller, optimization, reserved acceptance seeds or hardware was used.

| Checkpoint / starts | At least one full passage | Crashed during episode | At least two full passages and survived |
|---|---:|---:|---:|
| lags60-scratch / aligned | 23/100 | 100/100 | 0/100 |
| lags60-scratch / normal pad starts | 7/100 | 100/100 | 0/100 |
| lags60-scratch / randomized conditions | 6/100 | 100/100 | 0/100 |
| seenflag-s1-level1 / aligned | 84/100 | 16/100 | 84/100 |
| seenflag-s1-level1 / normal pad starts | 80/100 | 26/100 | 74/100 |
| seenflag-s1-level1 / randomized conditions | 73/100 | 52/100 | 48/100 |

A full passage requires starting with the whole body in front, crossing through
the opening, and clearing the far noodle face with the entire yawed collision
box. Contacts invalidate the transit. Centre-plane counts alone do not qualify.
Later crashes remain in the crash column even when an earlier passage succeeded.
This uses the group's discrete collision model; it is not continuous physical
collision certification. Simulator truth is used for reset and scoring only.

Aligned starts face a ring 0.6–0.8 m away, with lateral offsets up to 8 cm,
vertical offsets up to 4 cm and heading errors up to 10 degrees. Ring heights
range 0.8–1.2 m in that suite. Normal starts retain the group's level-1 layout
distribution, including ring heights 0.6–1.4 m. The randomized suite uses the
existing Randomise parameter ranges and timing/delay variation, unchanged.

The development seed ranges are 9410000–9410099, 9420000–9420099 and
9430000–9430099 respectively. These are development results, not the project's
sustained fixed validation or independent final acceptance. They must not be
used as training examples or relabeled as a fresh acceptance set.

## What failed

For the stronger checkpoint, all 16 aligned-start crashes hit the hoop/hanger
before a first clearance. With normal starts, 20 crashed before clearance and
6 afterwards (21 net and 5 hoop/hanger endings). With randomized conditions,
27 crashed before clearance and 25 afterwards (23 net and 29 hoop/hanger endings).
Thus, both the initial approach and subsequent flight still need improvement.
Ring detection working on a camera does not remove those policy failures.

## Exact model compatibility

- `lags60-scratch`: 109 inputs, lags 60,50,40,32,24,16,12,8,4,2,1,0.
  SHA256 `832e75fcf52998aa2b4b5dd1fa67ec0573d389816db5c5875e85099c5391ffe7`.
- `seenflag-s1-level1`: 85 inputs, lags 24,20,16,12,8,4,2,1,0, as recorded in
  its group training report. SHA256
  `8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef`.

Both profiles use the same eight public camera features, three actually issued
actions and seen-hoop flag. The runtime checks both the checksum and input/action
dimensions. It rebuilds each documented history explicitly, without changing
the global simulator contract or guessing a history from its dimension.

The 970 saved Mac detector samples were also replayed through the stronger
profile: all produced finite 85-input predictions. Issued-action history remained
zero. This was replay of recorded features, not image re-detection or closed-loop
flight, and supplies no new detection-accuracy or flight-success estimate.

## Command rehearsal

`src/command_rehearsal.py` exercises RC formatting, quantized issued-action
history, stop/land dispatch, and an independent watchdog against a local UDP
stand-in. It rejects all non-loopback destinations and has no takeoff command.
There is no operational physical flight launcher in this change.

Stop, Escape, L, Q and Ctrl+C dispatch zero RC followed by land. Invalid actions,
old/out-of-order frames, stale telemetry/operator input, inference stalls, send
errors and a 30-second rehearsal deadline latch the session stopped. Fresh
predictions cannot silently restart it. An unreceived landing packet cannot
guarantee a real landing; physical delivery and operator-interface behavior have
not been tested here.

The command ordering follows the [official Tello SDK guide](https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf)
and [DJITelloPy's RC implementation](https://djitellopy.readthedocs.io/en/latest/tello/).
The team's RC scales (50,50,100,70) are preserved for offline tests. The reported
vertical response still falls short of the simulation assumption in the existing
lab report; no new calibration or physical performance is claimed.

## Deliverables and checks

- Local `Mac-Policy-Preview-85.zip`: separate shadow camera/replay package with
  the stronger model and explicit 85-input profile. No movement commands.
  Original Mac-Lab-Recorder-Live package and all old models remain preserved.
- 100 relevant repository checks passed, including 23 command-rehearsal checks.
- Nine recorder/browser checks passed for each of the 85- and 109-input builds.
- Both package ZIPs passed CRC and file-hash checks; Mac launchers retain LF line
  endings and executable ZIP permissions. New 85-input build has not been run on
  the user's Mac yet. The earlier 109-input live session was verified on Mac.

Reproduce from the repository root (use a new output directory):

```bash
python -m pytest tests/test_sim.py tests/test_train.py tests/test_detect.py tests/test_baseline.py tests/test_camera_history.py tests/test_policy_validation.py tests/test_command_rehearsal.py -q
python -m tools.validate_flight_policies --out ../lab-data/policy-validation-repeat --episodes 100
python -m tools.analyse_flight_failures ../lab-data/policy-validation-repeat
python tools/build_mac_lab_preview.py --profile seenflag-s1-level1 --out ../Mac-Policy-Preview-85
```

Local evidence is under `lab-data/policy-validation-20261006`: pre-run protocol,
source/model snapshots and hashes, all 600 trajectories and episode outcomes,
failure analysis and the camera-feature replay. Small result files are copied
beside this report under `policy-validation-20261006/`. Raw lab video stays local.

The next learning work should address measured approach collisions and sensitivity
to randomized conditions, using separate training seeds and preserving single-pass
skill. The existing mission acceptance criteria remain unchanged. Training and the
recurring heartbeat were not resumed, and no physical flight test was launched.
