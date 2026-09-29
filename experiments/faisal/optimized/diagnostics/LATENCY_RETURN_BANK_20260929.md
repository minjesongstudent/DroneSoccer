# Latency return-history collection

A bounded source-policy collection uses the actual HoopEnv(level=1) full task,
64 new paired layouts2990000..2990063 under delays0 and2,128 flights total.
Every fourth layout is held out in BOTH delay conditions and all descendants;
96 flights FIT and32 heldout, declared before collection. No outcome filtering.
Development6920010 and reserved acceptance seeds excluded.

Source model is acquisition-return-mixture-20260928/model_0.zip. Actual
learned deterministic actions generate all flights, with no scripted movement.
At the first arrival at each return_side/return_front/align_front phase after
a pass and by step400, snapshot the real environment and complete authentic
public observation/action prefix. Capture is not conditional on later success.
Each snapshot must reproduce recurrent state and next action/observation/
reward/termination exactly. Dynamics are never swapped under a saved history.
All source flights are saved, including those without any snapshot.

Provenance correction: the older collect_recurrent_return_histories.py starts
from GuidedLessonEnv(stage=4). It generated genuine learned trajectories, but
it was not the same natural full-task start distribution as HoopEnv(level=1).
The new collector explicitly uses HoopEnv and retains paired delays. Avoid
calling the older guided bank natural full-task coverage in future reports.

Runner diagnostics/collect_latency_return_histories.py; output
 diagnostics/latency-return-histories-20260929 with protocol, source snapshot,
collector, all flight NPZs and hash-addressed snapshot manifest. No optimization,
physical operation, model promotion or publication. Physical flight unvalidated.

## Completed collection

Session68349 exited0. Source hash unchanged; zero optimization steps.
128flights:99left_cage,28hit,1timeout,zero full-mission successes.
All42snapshots passed exact recurrent-state and next-transition checks.

| Split | Delay | Flights | First-pass flights | Crashes | Return-side snapshots | Align-front snapshots |
|---|---:|---:|---:|---:|---:|---:|
|FIT|0|48|15|47|15|0|
|FIT|2|48|14|48|14|1|
|Heldout|0|16|5|16|5|0|
|Heldout|2|16|6|16|6|1|

The only FIT delayed align-front snapshot is seed2990002 step255; heldout
seed2990060 step257 must remain excluded. All other snapshots are return_side;
no return_front snapshot was captured under the predeclared first-arrival/
remaining-horizon rule. There are30 FIT and12 heldout snapshots, with related
phases grouped by their original parent layout. Snapshot count is not the
number of independent layouts. No flight improvement or acceptance claimed.

Next measure bounded learned-policy continuation outcomes from the30 FIT
snapshots, stratified by nominal/delayed dynamics, before configuring a new
training batch. Keep the12heldout snapshots out of fitting and sampling. The
single FIT delayed reapproach cannot establish broad coverage. Current source
and viewer remain unchanged; preserve this full dataset and provenance.
