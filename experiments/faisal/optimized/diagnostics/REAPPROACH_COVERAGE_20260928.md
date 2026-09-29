# Authentic re-approach coverage

Collected all seven live transitions into `align_front` across the 32 saved
training continuations, without selecting by outcome. One flight contributed
two transitions. Each trusted local snapshot includes its complete real
observation and issued-command prefix, current observation, full environment
state and source references. Prefix lengths equal the original episode step.
All seven restored next transitions match recorded observations, rewards and
termination exactly. Hashes and source copies are preserved in
`authentic-reapproaches-20260928`.

No stored hidden state is trusted after changing policy weights. Actor and
critic memory must be rebuilt from these prefixes before use.

At entry, signed front distance is 0.154–0.185 m, while absolute lateral error
is 0.195–0.803 m. Four states see the ring and three do not. Two have especially
large vertical offsets (-0.651 m and +0.337 m). These are geometric front-side
states, not proof of visual reacquisition or flight alignment.

The original 32 trajectories' collisions were classified with existing
simulator geometry: 13 noodle contacts and four hanger contacts. There were
also 14 cage exits and one timeout. This classification is diagnostic truth;
it is not supplied to actor observations.

## Preserved-policy continuation check

Evaluated every snapshot with one deterministic continuation and three
stochastic continuations, rebuilding the original policy's memory from the
actual prefix. Seeds and complete learned action records are in `policy_audit`.
All 28 continuations crashed before the original mission endpoint.

Three of seven deterministic continuations and seven of 21 stochastic
continuations achieved an additional passage before crashing. Thus a useful
re-approach skill exists in some states, but it does not sustain the next lap.
The high/low off-height snapshots produced no additional passages in any mode.
The second-return state at step 380 hit the noodle in all four continuations.
These small selected-phase diagnostics do not establish general success rates.

Next use this coverage to distinguish (1) returning to a usable approach,
(2) collision-free passage and (3) surviving the subsequent return. Any bounded
training proposal must retain authentic history and both early-return and
re-approach coverage, with natural-start validation and single-pass retention.
Do not optimize only for one additional passage or weaken mission criteria.

No training was performed, model weights were unchanged, and independent
acceptance seeds were not used. Physical flight remains unvalidated.
