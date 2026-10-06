# First successful camera recording: initial detector review

Session: 20261006-175222-714047. The operator confirmed hand-carried motion with
motors off. This is perception data, not autonomous flight or motion calibration.

Both command and streamon received valid SDK ok replies. The session recorded
120.119 seconds, 1,195 telemetry packets, 28,110 video packets and 37,498,116 video
bytes. All manifest hashes matched. OpenCV decoded 3,583 frames at 960 by 720;
decoder warnings include missing initial parameter sets and later damaged blocks.
The measurements template is blank.

## Initial sample review

The unmodified group HoopDetector was run on decoded frame indices 0, 30, 60,
and so on (120 samples), with confirm=1 for individual-image inspection. This is
not the deployed consecutive-frame confirmation mode and is not a detector
accuracy measurement: the frames have not been exhaustively labelled.

The detector returned ring visibility on 17 samples and marker visibility on 7.
Positive ring detections cluster at frame/30 estimates 34-40, 55-60, 74-77 and
93-96 seconds. These are approximate positions based on an assumed 30 fps,
not camera exposure timestamps or synchronized telemetry times.

Visual overview: the first approximately 30 seconds show the floor/cage, followed
by repeated ring approaches and views away from it. At frame 1200 (~40 s) the
ring is found but the visible white cross marker is not reported. At frame 2250
(~75 s) a cropped ring is rejected; at frame 3000 (~100 s) a close partial ring
is likewise rejected. Clearer views such as frames 1050 and 2850 are accepted.
The hand-carried sequence is consistent with the operator's account of moving
through and around the ring; sampled monocular images do not certify full-body
clearance or an exact passage count.

## Decision

Retain this recording: it supplies useful real ring and marker examples. No repeat
capture is required just to establish that the ring was recorded. Do not change
thresholds based on these samples alone. Next, label continuous approach/return
segments and evaluate the live confirm=2 detector on every decoded frame. Measure
false positives, ring misses and marker misses separately, including partial views
and compression damage; preserve separate segments for checking any proposed fix.

The original recording and review images remain local, rather than publishing
identifiable lab footage. Only numeric diagnostic samples accompany this report.
No policy inputs were fabricated, no model trained, and no flight commands sent.
