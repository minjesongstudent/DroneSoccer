# Geometry audit for staged search v4

The actual full_horizon training source snapshot (not just the current legacy
checkout) uses a 205 cm cage and hoop CENTRE heights uniformly sampled from
60 to 140 cm. Initial drone z placement clips to cage height minus 15 cm, but
runtime wall collision clamps the drone body at cage height minus half its
vertical size. Thus 190 cm was an initial-placement bound, not that training
simulator's universal in-flight ceiling. See geometry-audit.json for exact lines.

User reports the hoop 15-50 cm below the 205 cm roof. Which part of the hoop
this references remains unconfirmed. With the configured nominal 45 cm outer
diameter, if this is its TOP EDGE, centre heights are 132.5-167.5 cm. If it is
its CENTRE, heights are 155-190 cm; centres above 182.5 cm put part of an upright
45 cm hoop above the 205 cm roof. Do not silently encode the latter as a valid
layout distribution. The +/-3 cm per-axis simulation deformation would also
need to be included when deriving centre height from top-edge clearance.

The packaged/training camera pitch is 12.8 degrees down. The default fy=920 at
720 pixels gives about21.37 degrees vertical half-field, so the upper edge looks
only about8.57 degrees above horizontal. These intrinsics are defaults with
randomization, not a verified current-camera calibration. A high ring can remain
partially cropped even after a modest climb; distance also affects visibility.
No unique physical ring height is inferred from the supplied frames.

Option1 startup uses independently bounded staged ascent/yaw. It does not alter
actor inputs, learned weights or historical simulator evidence. New startup
search is scripted and separately recorded; it is not learned-policy success.
The requested190cm is a hard telemetry ceiling; upward search stops at180cm
as a10cm stopping allowance. This allowance is a design choice, not verified
physical clearance, position holding or guaranteed overshoot prevention.

The simulator hoop range remains a documented sim-to-lab mismatch. Existing
52/100 crash results describe the original distribution only. Before reporting
reliability at higher placements, use a versioned simulator configuration with
confirmed height reference and re-evaluate/train on properly separated seeds.
Historical source snapshots and their original scores must not be rewritten.
