# Policy sampling diagnostic launch - 8 October 2026

The bounded CUDA inference comparison completed and failed its diagnostic gate. See [the result](POLICY_SAMPLING_RESULT_20261008.md). Details below describe the historical launch at
`D:/DroneSoccerData/runs/policy-sampling-comparison-20261008` (managed session34095).
This is not a trainer or a candidate promotion. The original85 SAC actor is unchanged.
The [frozen plan](POLICY_SAMPLING_COMPARISON_20261008.md) specifies 64 current FIT
layouts, one deterministic and two normally sampled learned-policy flights per
layout. Maximum192 episodes and600seconds, same600-frame mission and passage/crash
rules. Public85 only; truth is confined to reset identity and diagnostics.

Ten focused tests passed. CUDA smoke policy-sampling-smoke-20261008 completed six
flights in3.719seconds; all56 evidence hashes verified, unchanged actor confirmed,
and its software-only result correctly cannot support a learning intervention.
The current main allocation was checked before launch and is rescanned under the
project lock. Only one diagnostic process holds that lock, allocator fraction .15,
strictfloat32/noTF32, one CPU thread. Prior failed memory trials stay closed.

The reason for this test is the current training/evaluation action-mode difference
and repeated noisy outcome evidence. The September RecurrentPPO sampling test was
on a different actor with zero clean missions in both modes; its failed conclusion
is preserved. This comparison concerns the retained SAC source with demonstrated
deterministic repeat flights. It is not a parameter-noise or scale sweep.

Passing requires both repeat gain and crash reduction at least8/64 versus the
average of two stochastic replicas, with positive differences in both halves and
both reset strata. Passing only supports preparing a separately bounded learning
experiment. Failure closes this diagnostic without more seeds, repetitions or a
noise-scale sweep. Neither outcome changes the final policy qualification gates.

Next heartbeat: read compact status.json first. If healthy/running, finish quietly.
Only after completion or failure read the saved result and evidence once, then
record the next justified action. Do not change bound sources during execution.
No physical flight, publication or master-branch changes. Simulation incomplete.
