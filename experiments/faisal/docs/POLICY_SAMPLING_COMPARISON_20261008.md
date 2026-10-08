# Original SAC sampling comparison - frozen protocol, 8 October 2026

Trainable-memory fitting completed 240 accepted updates but lost nominal repeat
retention and gained only one randomized clean repeat. Training collects sampled
commands whereas deployment screens use deterministic commands. Earlier family
and gradient studies found noisy outcome contrasts. A separate 28 September
RecurrentPPO sampling comparison found no clean missions in either mode; that
failed experiment remains closed. The current original85 SAC model already has
48/100 randomized deterministic clean missions, so the old different-model
comparison does not establish its sampling behavior.

This test asks whether unchanged original85 policy sampling loses substantial
clean-repeat experience relative to its own deterministic commands on matched
current FIT resets. It does not retest parameter perturbations or sweep noise
scales, and cannot establish a better learned policy or guarantee that changing
exploration improves training.

- Exactly 64 unused FIT layouts, 11244720..11244783, alternate the existing
  approach/full-task reset curriculum. One deterministic and two stochastic
  learned-policy flights each: 192 episodes, at most 600 frames each.
- Original hash-bound seenflag-s1-level1 SAC, no optimizer, unchanged means and
  variances. Actions come from its standard predict path. Public85 observations
  contain camera and actual command history only. Truth records reset identity
  and full-body passage/crash/return diagnostics, never actor inputs.
- Separate policy seeds 11605000 + 2*layout_index + replica_index. Identical
  initial public observations, pose, ring and dynamics required per layout.
- Freeze a positive diagnostic gate: deterministic clean-repeat gain AND crash
  reduction must both reach eight out of 64 compared with the mean of the two
  stochastic replicas. Both outcomes must improve in each 32-layout half and
  in both 32-layout reset strata. No claim of independent statistical significance.
- Passing only supports preparing a separately bounded learning intervention.
  Failing closes this diagnostic: no added layouts, repetitions, noise sweep or
  automatic optimizer. Final policy qualification criteria remain unchanged.
- One standalone project-lock process, CUDA RTX4070Ti, float32/noTF32, allocator
  fraction .15, one CPU thread, total 600 seconds including preparation. STOP
  checked every frame. Incomplete runs cannot pass; preserve partial trajectories.
- Smoke uses only 11244788/11244789, six flights, same pipeline; never qualifies.
  An initially considered interval starting11244700 overlapped older authentic
  reset data and was rejected before execution. The actual allocation scanner
  found no conflicts across903 episode logs/34570rows/140protocols before launch;
  recheck inside the project lock and reserve in protocol before any flight.

Preserve source snapshots, hashes, all learned actions/observations/trajectories,
paired records, status and final gate. Earlier failed models remain unselected.
No development/acceptance layouts or hardware are used.
