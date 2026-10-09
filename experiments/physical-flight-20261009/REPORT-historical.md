# Physical flight review - 9 October 2026

## Finding

The recorded model is the requested seenflag-s1-level1 checkpoint, but the physical
closed loop fails. Six sessions sent takeoff and policy RC; three sent neither
takeoff nor policy RC. All six active sessions started with all-zero camera8
features (ring not detected). Only 48 of 410 policy steps had a detected ring:
88.3% of commands were selected without a current ring detection.

Every first policy command was `rc 8 15 38 -2`; after sustained unseen input it
often approached `rc -3 8 47 -9` (upward), or after losing an acquired ring,
`rc -2 -25 -2 -23` (backward and yaw). These are actual model outputs, not
commands substituted by the launcher. Offline replay of all 410 stored actor
observations with the exact checkpoint reproduced raw outputs to max absolute
error 0.00000155. Quantized issued-action history matched exactly.

## Session evidence

| Start (Mac time) | Policy commands | Command span (s) | Ring detected | First landing evidence |
|---|---:|---:|---:|---|
| 17:02:12 | 161 | 16.61 | 11/161 | Land sent after 16.6 s of policy commands; trigger not recoverable from final reason. |
| 17:03:47 | 0 | 0.00 | 0/0 | No takeoff sent; temperature 80-84 C. |
| 17:07:51 | 0 | 0.00 | 0/0 | No takeoff sent; final temperature 88 C. |
| 17:10:58 | 99 | 10.15 | 16/99 | Height collapsed to 0 cm; final reason: height below test floor. |
| 17:13:21 | 21 | 2.10 | 0/21 | First land immediately after reported height reached 160 cm. |
| 17:14:17 | 26 | 2.61 | 3/26 | First land immediately after reported height reached 160 cm. |
| 17:16:29 | 72 | 7.36 | 13/72 | First land immediately after reported height reached 160 cm. |
| 17:17:29 | 31 | 3.15 | 5/31 | First land immediately after reported height reached 160 cm. |
| 17:18:14 | 0 | 0.00 | 0/0 | No takeoff sent; final temperature 82 C. |

## Video cross-check

The generic watch helper cannot establish a duration for raw H264; it returned
only an initial frame. The reproducible analyse.py instead decoded each recorded
UDP packet with PyAV using the same parser as the recorder. Selected frame IDs
and packet receipt times matched the policy log exactly. Four flights were
visually sampled at first policy command, first detection, intermediate and final
command (15 frames total; duplicates removed).

The first frames show net/walls/floor, without the ring in frame. Later images
show the actual blue ring entering from above or the side, frequently clipped.
Some visible ring imagery is rejected by the detector; absence of detection is
not universally a false negative because the ring is often outside the frame.
The 17:10 flight ends with a sharply rotated/blurred view and falling reported
height. That is consistent with a physical upset; the exact contact/cause cannot
be established from these samples. Do not claim completed physical passages.

## What is established, and what remains uncertain

- SDK takeoff replies and rising height support actual takeoff in six sessions.
  UDP acceptance does not establish execution of every individual RC command.
- The camera/actor path ran at about 10 Hz: recorded policy gaps at most 0.186 s,
  with max reported host frame ages below 0.147 s. This does not measure optical
  exposure delay or rule out unmeasured network/actuation delay.
- All nine archive checksum manifests verify; all nine metadata records identify
  the intended model hash 8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef.
- Four flights requested landing just after h reached 160 cm, consistent with the
  150 cm ceiling. A later emergency overwrote the original state-machine reason,
  so those initial triggers are inferred, not logged independently.
- Three zero-policy sessions ended above the launcher's 80 C threshold. Their
  complete start-attempt histories are not logged; do not invent rejected clicks.
- Camera calibration remains explicitly default/unverified. These logs cannot
  uniquely identify camera intrinsics or all dynamics parameters.
- Mission-pad x/y/z reports are often invalid (-100 with mid=-1); no continuous
  global trajectory is inferred. Negative height transients also limit that data.
- Final policy_seconds includes time spent landing/stopped before export; it is
  not the duration of actual policy command output. The table uses command span.

## Corrective work supported by the evidence

Pause unchanged physical policy trials. First add an explicit target-acquisition
condition before handing navigation to the learned actor after takeoff. If the
ring cannot be acquired within a defined window, request landing; do not send
blind learned search commands into the cage. Temporary ring loss during a passage
is expected, so a later loss policy must be tested against the full mission rather
than banning every invisible frame. Test using the captured clips without motors.

Replay and label the ring/clipped-ring sequences to assess detector continuity and
feature accuracy. Compare the recorded startup view and response to the actual
training reset assumptions. Calibrate/update the simulation where evidence supports
it, then train/evaluate blind-start and lost-target recovery without weakening
passage/crash criteria. Do not invent missing physical calibration or promise that
moving the ring alone will solve transfer.

Record immutable state transitions and initial stop reason before any later
emergency, plus start rejection reasons. Keep the existing height, temperature,
input-loss and emergency controls; raising limits would hide this failure.
No policy/software fix or successful physical validation is claimed by this review.
