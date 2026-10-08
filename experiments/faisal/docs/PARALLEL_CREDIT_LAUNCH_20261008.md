# Parallel comparison launch and resource admission

Faisal requested concurrent variable comparisons. The frozen experiment is
`PARALLEL_CREDIT_COMPARISON_20261008.md`: three credit constructions with the
same frozen original actor and estimated-motion residual. This is not a new
checkpoint-reset experiment and does not change any policy success criterion.

Implementation checks: 90 tests passed on Windows, covering the existing
worker/on-policy logic, private worker registration, the project lock,
kill-on-close process lifetime, fixed schedules, the barrier before any
optimizer update, legacy allocation formats and smoke-to-main admission.
Each arm has its own process, model, optimizer, RNG and output directory.
The source and estimator stay frozen; only the declared residual is trained.

The first software launch at
`D:/DroneSoccerData/runs/parallel-credit-smoke-20261008` failed in 0.531 seconds
while reading historical protocols whose `arms` field was a list of names.
No worker, CUDA training, seed allocation or cohort manifest was created.
The failure and pre-fix sources remain preserved. The scanner now supports
the historical list format while retaining root seed reservations, and rejects
unknown structures. A complete rescan found no collision in 628 episode logs,
22,174 episode records and 88 protocols. This software repair is not an
experimental policy retry or an altered performance gate.

During preparation, Rainbow Six began using the GPU. The reviewed launch
requires 7,574 MiB free (three allocator limits of .15 of total VRAM plus
2,048 MiB for context and other overhead). Observed free memory fell below
that threshold. No user application was closed and limits were not increased.

The bounded queue is
`D:/DroneSoccerData/runs/parallel-credit-launch-20261008`, initially launched
as PID49672 with actual interpreter PID70660. Read `status.json`, live process
identity and `result.json` before deciding whether it is still active.
It snapshots and checks the reviewed source hashes. Each GPU admission waits
at most 900 seconds, requires two adequate samples ten seconds apart, and
does not automatically retry a timeout or a failed experiment.

If admitted, it runs exactly this sequence:

1. `parallel-credit-smoke-20261008-v2`: three simultaneous CUDA workers,
   software only, using seeds11244900 and11244901. Each worker is limited to
   600 seconds; the coordinator to900 seconds. Exact first-rollout features,
   actions, lengths, rewards, old log probabilities and endpoints must match
   before any value or actor update. Different returns are intentional.
2. Verify all three completed software-only closures, frozen artifacts,
   evidence hashes, and matched rollouts. Any failure prevents MAIN.
3. Wait for GPU headroom again, then run `parallel-credit-20261008` with
   five rounds of48 complete episodes per arm, fixed FIT seeds11244000 through
   11244239, and algorithm seed11592000. Worker bounds remain3600 seconds and
   coordinator bound3900 seconds. The existing training lock excludes another
   coordinator or standalone trainer. At most one qualifying final candidate
   can be nominated for a separate fresh paired development screen.

The queue's `STOP` file cancels the remaining workflow and its owned process
job; an active cohort has its own global and per-worker STOP checks. Do not
edit bound source files or start another cohort while this queue is active.
If it times out waiting, inspect current resources and preserve that result;
do not claim that training occurred. A completed software smoke is never a
policy improvement result.

Original85 remains selected. Simulation acceptance is incomplete and physical
flight remains unvalidated. No hardware operation, publication or master
branch change occurred.

For the separate checkpoint question: historical authentic full-state capture
is documented in `../optimized/diagnostics/LATENCY_RETURN_BANK_20260929.md`.
All42 snapshots reproduced recurrent state and the next transition exactly;
that collector used an older policy and demonstrated no full-mission success.
It proves the reset mechanism, not that the current policy has mastered the
continuation. Full-task validation must still begin from normal starts.

## Admission timeout, 8 October 02:52 Sydney

The queue has closed with `phase=stopped`, `training_started=false`, and
`GPU admission timed out after900seconds; no automatic retry`.
Elapsed wall time was908.656 seconds including checks and process overhead.
Its final GPU sample recorded6,666MiB free, below the unchanged7,574MiB
requirement. At the subsequent heartbeat the GPU had6,855MiB free and
Rainbow Six was still active. No Python process remained; neither the v2
CUDA smoke nor MAIN directory had been created.

No model update, policy evaluation, FIT episode or seed reservation occurred
in this queue. The prior original85 policy and every historical model remain
unchanged. No new wait, experiment retry, relaxed resource limit or stopped
user application was issued. The prepared comparison remains pending a real
change in available GPU resources. Subsequent unchanged checks should remain
quiet; do not create repeated admission timers while memory stays insufficient.

Independent read-only closure checked all40 current source files and all40
snapshots against the launch protocol, with no differences. The queue's
`admission_closure.json` records the untouched protocol/result/status hashes
and absence of the planned stage outputs. No new inference or rollout was
performed for this closure.

## User resumed after closing the game

Faisal explicitly resumed work and the scheduled task after closing Rainbow
Six. The GPU had8,404MiB free, no Python or game process remained, and all40
reviewed source hashes still matched. The heartbeat was restored to ACTIVE
with its existing20-minute interval. A new admission workflow was launched
in `parallel-credit-launch-20261008-r2` (launcher86304); the expired workflow
remains preserved. Algorithm, seeds, resource limits and gates are unchanged.

The queue's compact status subsequently reported `running_main` for
`parallel-credit-20261008`, coordinator launcher107884. It had first verified
the complete three-worker v2 software smoke and recorded its decision SHA256
`e4fc9b93716be584cd3b0ebeaf4f5f8fdc7ca5ce684e936c80871434af3e6a6e`.
This is software readiness and launch evidence, not a policy performance
result. Detailed result review is deferred until the bounded batch finishes.

At Faisal's request, scheduled checks now read compact status first and end
quietly while a healthy run remains active. Detailed logs, trajectories,
audits and further tests are examined after completion or a meaningful
failure, rather than repeatedly during unattended computation.

## Completed comparison

The resumed queue completed in 988.14 seconds and the main coordinator in
940.015 seconds. All three decisions rejected the variant; no candidate
qualified. See [the final result](PARALLEL_CREDIT_RESULT_20261008.md) for all
outcomes, one saved-data diagnosis and independent closure evidence.
