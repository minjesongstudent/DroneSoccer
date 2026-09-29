# Return timing replay: development seed6920010

Six diagnostic episodes reproduced the saved evaluation outcomes, pass counts
and terminal steps exactly. All movements came from each checkpoint's learned
deterministic policy. Public119D observations/actions are saved separately
from simulator-only post-step geometry. No optimization or actor modification.

| Model | Delay steps | Passes | End | Step | Minimum ring clearance |
|---|---:|---:|---|---:|---:|
| Source0 |0|2|ring hit|472|-4.62mm|
| Source0 |2|2|timeout|600|13.60mm|
| Update1 |0|3|timeout|600|22.97mm|
| Update1 |2|1|ring hit|284|-8.16mm|
| Update2 |0|2|ring hit|492|-7.78mm|
| Update2 |2|1|ring hit|285|-11.95mm|

Clearance is minimum distance from sampled ring centreline to drone collision
box minus ring tube radius, matching the simulator noodle collision geometry;
it is not total clearance from every obstacle. No terminal hanger collisions.

Source nominal second reapproach entered align_front at432 and hit at472,
0.335m lateral from ring centre, after63 consecutive actions without a visible
ring. Update1 nominal reapproaches at211/473 led to passages at262/524.
Update2 nominal second reapproach began455 and hit492,11 actions after loss
of visibility. Phase labels describe simulator progress, not proof of actual
visual alignment.

Both updated delayed runs entered align_front at236. Update1 hit284 at
along=-.039m,lateral=-.022m,vertical=-.055m; update2 hit285 at
along=-.058m,lateral=-.050m,vertical=-.066m. Both had16 final actions without
visible ring (1.6s at the simulated .1s step). This local evidence points to
latency-sensitive blind reapproach, but does not establish a universal cause.
The source delayed run's second passage occurred291; it survived to600 while
remaining in return_side and blind for the final323 actions. Success under
the existing criterion does not mean indefinite stable laps.

Artifacts: return-focused-actor-20260929/timing_replay contains protocol,
checkpoint hashes, copied runner, six public observation/action NPZs, six
post-step diagnostic traces, result.json and visibility_tail.json. Runner
replay_return_timing.py refuses overwrite. No truth fields entered actor input.
This selected development case must not be fed into training or promoted as
independent acceptance. All historical models/viewer retained.

Next justified data change: collect authentic return/reapproach histories from
new predeclared FIT natural starts under BOTH nominal and delayed dynamics,
using the preserved learned source actor and actual public prefixes. Preserve
all source flights; snapshot phase arrivals without selecting on later success.
Existing authentic histories mainly supply nominal dynamics; nominal and delayed
natural starts alone yielded too few return experiences in recent trials.
Check how many delayed returns are actually available before training, and
verify snapshot/prefix replay parity. Do not fabricate states or swap dynamics
beneath a saved history. Keep development and acceptance seeds excluded.
Simulation acceptance remains incomplete; physical flight unvalidated.
