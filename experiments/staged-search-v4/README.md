# Staged search v4 - option 1

User authorized bounded higher-altitude and wider-yaw acquisition, requesting
190 cm maximum in the 205 cm cage. The [geometry audit](GEOMETRY.md) distinguishes
that request from the old training distribution and documents the still-ambiguous
roof-to-hoop measurement reference.

Search scans at telemetry heights 120, 150 and 180 cm. At each stage it settles,
turns toward +75 degrees, -75 degrees, then its initial heading. Targets use actual
wrapped yaw telemetry rather than assuming a fixed angular rate. Each leg allows
18 seconds, with at least10 degrees progress per4 seconds; a90-degree accumulated
excursion guard remains. RC up/yaw remain15, horizontal startup RC remainszero.

Upward search stops at180cm. A190cm telemetry ceiling applies to acquisition and
the selected policy after handoff. This10cm stopping allowance is experimental,
not a measured guarantee of clearance in the205cm cage. Do not confuse zero
horizontal command with physical position hold. A10cm gain per4seconds is required
during continuous climb, with32seconds cumulative upward-command time. Total
search cap120seconds; slow or stalled scans can end before every stage completes.

Detection immediately requests neutral commands, requires three distinct fresh
frames across0.4seconds, then hands off on a subsequent visible frame. Actual
camera/issued-action/GRU history continues. Search movements are scripted and
logged separately; navigation after handoff remains the unchanged learned model.
No detector threshold reduction or invented actor observations. No endless retry.

72 offline tests pass in the checkout. Tests cover full three-stage coverage under an explicitly synthetic quantized
6cm/s and12degrees/s plant, plus recorded climb-trace replay, stalls, bounds,
operator stop, stale input, send/log failure and handoff. Synthetic movement is
only control-logic validation; physical flight remains unvalidated.

Reproduce from policies/lab-runner after setup:

    python -m unittest -v test_recorder test_live_preview test_calibration test_compare_session test_flight_control test_supervised_flight test_pc_policy test_startup_search

All model components and original training source/evaluation archives are unchanged.
The existing full_horizon-round-05 candidate remains unqualified (52/100 randomized
simulation crashes). Higher startup reach does not demonstrate high-hoop policy
reliability; a confirmed-height simulator configuration and new evaluation are needed.
