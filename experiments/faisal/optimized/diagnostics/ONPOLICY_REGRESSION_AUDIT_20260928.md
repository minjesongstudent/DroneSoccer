# On-policy regression: evidence before another run

The stopped trial completed4096 critic-only transitions and4096 actor-enabled
transitions. The actor hash was unchanged during calibration; fixed validation
counts were unchanged at4096. This rules out actor modification during that
phase as the source of regression.

Logged critic explained variance rises about.300 -> .385 -> .512. The logged
actor-update approximate KL is .00235, clip fraction .098, learned std about.200.
These are rollout diagnostics, not calibrated held-out critic-error guarantees.
The SB3 log rows may reflect the preceding update, so they do not capture every
final optimizer statistic. Modest average KL does not guarantee skill retention.

Training outcome counters at8192 contain49 completed stage4 episodes:33hits and
16cage exits, no full mission. Thus this trial provided no completed lap episodes
in its recorded full-task training outcomes.

A separate diagnostic used the unchanged4096 checkpoint on64 paired training-
domain natural layouts, seeds610100..610163 (not fixed validation or acceptance).
Both inference modes reset recurrent state at genuine episode start:

| Mode | Flights with a pass | Full missions | Hits | Cage exits | Timeouts |
|---|---:|---:|---:|---:|---:|
| Deterministic |41/64|0/64|36|25|3|
| Learned stochastic distribution |32/64|0/64|46|16|2|

Stochastic sampling reduces first-pass retention on this finite comparison, but
deterministic inference also completes zero full missions. Lower exploration
alone is therefore not an established fix. The isolated3/25 fixed-lesson result
does not translate into reliable natural-layout behaviour here.

Artifacts: audit_onpolicy_exploration.py/onpolicy_exploration_audit.json, including
checkpoint hash and episode records. No optimizer steps, source-model writes,
reserved examples, acceptance seeds, physical hardware or publication.

Decision: do not resume8192 or merely lower noise/raise training duration. Need a
training strategy that can acquire successful complete returns while retaining
basic approach skill; investigate dataset coverage and policy/critic learning
signals before another run. No active trainer remains.
