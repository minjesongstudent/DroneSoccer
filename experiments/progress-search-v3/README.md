# Progress-aware search v3 - 2026-10-09

Four uploaded takeoffs all stopped with startup search ascent timeout. At RC up=15,
search-start/end telemetry heights were 70/100, 40/70, 50/80 and 40/70 cm. Each
received 46-48 upward sends. Two isolated detected frames occurred in the third
flight; none sustained confirmation. All five exports had zero learned-policy rows.
The fifth had no takeoff. All file manifests verified. See summary.json and analyse.py.

The first recording's sampled frames show the blue ring entering the top edge
as height rises; the five-second abort prevents completion of the planned climb.
Raw videos remain local. Review MP4 was remuxed at 24 fps for frame inspection;
its approximate times are not a synchronized physical position measurement.

Fix: same RC=15 and target=120 cm, ascent cap 16 seconds; require a new 10 cm
measured-height gain within four seconds of continued upward commands, otherwise
request landing. Ten cm accommodates recorded height quantization. A detected
ring requests neutral RC immediately; stable detection still requires three distinct
frames over 0.4 seconds. Detection pauses reset the consecutive-climb progress
window, but never the 16-second ascent deadline or 30-second total search cap.
Yaw/downward phases, zero horizontal startup RC and existing stop checks remain.
150 cm ceiling unchanged. No motor operation or physical validation here.

66 offline tests pass. Tests include replay of the four measured startup traces
plus a synthetic slow climb at approximately the observed 6 cm/s from 40 to
120 cm, quantized in 10 cm increments. The synthetic continuation is NOT proof
of physical acquisition. Stalled height and insufficient progress land; absolute
timeouts, interrupted inference, recording errors, stale inputs, operator stop
and learned handoff remain covered. Detector thresholds and all three policy
components are unchanged.

Reproduce using the environment installed by policies/lab-runner/1-Setup.command.
From that runner folder execute:

    python -m unittest -v test_recorder test_live_preview test_calibration test_compare_session test_flight_control test_supervised_flight test_pc_policy test_startup_search

Audit original exports using analyse.py EXPORT_DIRECTORY OUTPUT_DIRECTORY.
Recorded numeric regression fixture is in lab-runner/testdata; no videos published.
