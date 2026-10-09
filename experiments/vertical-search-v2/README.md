# Vertical-only acquisition commands — 9 October 2026

The user explicitly requested up/down movement and left/right turns without
sideways translation before the learned policy starts. This extends startup v1
with a downward search after the upper-height sweeps. No policy weights changed.

Sequence: settle after takeoff, rise slowly toward measured 120 cm if the ring
is unseen, make the two existing bounded yaw sweeps, then lower slowly toward
80 cm if still unseen. Stop the search motion and confirm the ring before
learned-policy handoff. First two RC components are always zero (right/left and
forward/back). The flight command boundary separately rejects any search proposal
with a nonzero horizontal component, including an accidental future regression.

This does **not** implement measured XY position hold: no trustworthy global XY
position feedback was added, and real drift remains possible. The commanded
search is vertical/yaw only; do not describe it as guaranteed fixed position.

Upward/downward RC magnitude is 15. Each ascent/descent attempt has a five-second
limit. Upper-height yaw legs remain three seconds each with the existing measured
60-degree excursion bound. Total acquisition cap is now 20 seconds to allow the
new descent; the learned policy still has no ten-second flight-duration cap.
At 80 cm with no ring, a stalled descent, expiry or any safety/stop condition,
request landing. Existing 150 cm ceiling, 30 cm floor, thermal, battery, freshness
and operator checks are unchanged. Never descend below the lower search target
intentionally; physical stopping accuracy remains uncalibrated.

All startup movement remains explicitly scripted and recorded separately from
learned navigation. Camera and actually issued action/GRU history continue across
handoff. 62 offline tests passed, including downward acquisition/handoff, lower
bound and descent timeout, horizontal-command rejection and all prior regressions.
No physical drone was operated or flight qualification inferred.

Current package: PC-Policy-Lab-Vertical-Search.zip. UI label: Vertical search v2.
The preceding v1 package remains recoverable in Git at 51029b5.
