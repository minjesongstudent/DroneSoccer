# Bounded startup search — 9 October 2026

The user reported that the ring was above the camera after takeoff and explicitly
requested upward movement and looking around before handing over to the policy.
The preceding nine exports contained four takeoffs and zero learned navigation
steps. Three stopped immediately for missed ring detection; one for negative
height telemetry. This change addresses acquisition, not learned-policy quality.

After takeoff acknowledgement and fresh height >=30 cm:

1. Enter `acquiring`, keeping all camera, telemetry, operator, battery, temperature
   and height checks active. Valid yaw is required before takeoff.
2. Request neutral RC for at least 0.5 seconds. If no confirmed ring is visible,
   request slow upward RC 15 until measured height reaches 120 cm. Stop ascent
   if the ring appears. Five seconds maximum ascent attempt; never move forward
   or sideways during search. The 150 cm ceiling remains unchanged.
3. At search height, request yaw RC +15 for up to three seconds, then -15 for up
   to three seconds. Measured yaw excursion is bounded to 60 degrees from its
   initial heading, with wraparound handling. These are bounded commands, not
   calibrated guarantees of a particular trajectory or sweep angle.
4. Any accepted ring detection requests neutral RC. Require at least three fresh
   unique detector observations across 0.4 seconds, with no gap over 0.25 seconds,
   before handoff. A flicker resets confirmation. Detection still uses the existing
   detector; this does not establish its physical accuracy or reject every distractor.
5. After recording the neutral command, start learned control on the next fresh
   observation. If the ring is lost before the first learned command, land. There
   is no fixed policy-flight duration limit.

The entire acquisition stage has a 14-second maximum; completing both sweeps or
failing ascent can end it earlier. Timeouts, invalid input, logging/send failures,
lost UI/contact or an operator landing request end the attempt. No restart loop.

Startup movement is **scripted**, explicitly requested by the user and distinct
from trained-policy output. It is logged to `startup_search.jsonl` and shared
command/telemetry/video logs. `policy_preview.jsonl` contains only learned-policy
commands; comparison summaries exclude startup steps from learned-policy counts.
The real camera/previous issued command/GRU history is updated throughout search
without feeding measured height, yaw or simulator truth into the actor. It is
preserved at handoff. The original checkpoint and both shared models are unchanged.

The UI shows `Startup search v1`, the current phase and exact policy identity.
`policy_steps` counts issued learned actions. `policy_seconds` now reports the
span between first and last issued learned commands, excluding landing retries.

Settings are experimental search bounds, not newly measured ring height or new
training results. No simulation score or flight-qualification claim is changed.
No physical drone was contacted during development. Test evidence covers mocked
transport, real-model handoff, acquisition, timeout, recording separation,
watchdogs, late results, bad telemetry and existing recorder/camera behavior.

All **59 tests passed**, both in the checkout and from the complete ZIP extracted to a separate directory. See tests.txt and extracted-package-tests.txt. Build reproducibly with `python policies/build_lab_package.py /path/to/new-output.zip` from the repository root; the builder refuses overwrites and validates model hashes.
