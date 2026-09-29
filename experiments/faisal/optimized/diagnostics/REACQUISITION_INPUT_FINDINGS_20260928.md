# Matched-history error at camera reacquisition

Audit uses existing nine FIT teacher flights and learned baseline500 rollouts
from the same natural starts. No reserved or acceptance layouts were used.

At reacquisition all9 teachers and all7 students that passed show current marker0
(back view) and front-memory-seen1. Mean heights relative to ring centre are
-.279m teacher and -.296m student. Thus neither missing front-memory flag nor
descent alone explains the observed difference.

Mean normalized commands (right, forward, up, clockwise yaw):

- Teacher: [-.682, .628, .995, -.964].
- Student on its own7 histories: [.079, .151, .315, -.573].
- Student on the exact9 teacher histories: [.009, .158, .304, -.618].

Matched-history action MSE at these9 events is .34848, versus approximately
.10197 across all5400 FIT steps. Per-axis event MSE .49627/.29112/.48364/.12289.
This demonstrates substantial local imitation error even before rollout drift;
it does not prove that copying these9 actions guarantees safe return generally.

Evidence: audit_reacquisition_inputs.py/reacquisition_inputs_audit.json. True
geometry appears only in reports; the actor reads unchanged119D public inputs.

Bounded follow-up: same MLP+head anchor10 trial, but weight20steps after each
camera-visible0->1 transition by5 (overlapping windows capped at5). These windows
are derived from existing public observations, not hidden pass/phase flags.
Full600-step recurrent prefixes remain intact, LSTM/critic frozen, maximum100
updates, retention checks25/50/100. It is supervised fitting, not a new RL result.
No change to simulation rewards, success criteria or physical flight.

Trial completed100updates: centred25/25 throughout, varied21/22/21 at25/50/100, lesson fullmissions1/1/0 outof25, general/delayed0throughout. No sustained completion; no promotion or unchanged extension. All artifacts in lap-reacquisition-weighted-20260928; no active trainer. This does not resolve recurrent representation or scarce successful-history coverage.
