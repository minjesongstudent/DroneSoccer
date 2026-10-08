# Trainable control-memory launch — 8 October 2026

The [bounded plan](TRAINABLE_MEMORY_ACTOR_TRIAL_20261008.md) completed and failed its final gate. See the [closure report](TRAINABLE_MEMORY_ACTOR_RESULT_20261008.md). The launch details below are historical; no trainer remains active.
Earlier frozen-memory actor variants remain closed. This policy starts again from
original85 plus the pretrained GRU and zero correction; control loss can now update
GRU memory across complete flight histories. It is not a continuation of a failed
checkpoint. Original controller, action variance and input normalization stay frozen.
No physical estimate is asserted for the learned hidden state.

## Completed checks

- 13 new component tests,6 initialization orchestration tests and50 runner tests
  passed (69 focused checks in their respective runs). They cover genuine recurrent
  gradients, reset/causality/padding, safe head+GRU checkpoints and re-evaluation of
  current memory for the source trust guard after each update.
- Initial fused CUDA sequence discrepancy7.56979e-6 exceeded the fixed1e-6 bound.
  Shared explicit cell arithmetic fixed the issue, with the same bound. The failed
  implementation and original test source are preserved under
  `D:/DroneSoccerData/runs/trainable-memory-integration-20261008`.
- Initialization75 episodes/31634 frames exactly reproduced original source
  actions and preserved trajectories in30.031seconds. Decision SHA256
  `659704f72cb0a613e385760312e4f008d9e339f4cb9d9e844a4f22f10dba3d30`.
- CUDA smoke completed9.766seconds,2 episodes/755 transitions,2 accepted/0 rejected
  proposals, confirmed changed GRU weights and unchanged base/normalizers. Full
  software pipeline completed; its decision intentionally cannot qualify outcomes.
  All102 evidence hashes and44 source hashes verified. Decision SHA256
  `265f439b0ff354420cb83a87f9f34a21b0151ac0c6dc85552c1d83497e8048dc`.
  Peak GPU allocation32,118,784 bytes; reservation52,428,800 bytes.
- The initially considered FIT interval overlapped earlier reservations and was
  rejected before execution. The actual unused interval and two smoke seeds passed
  the scan. No guard or seed interval was widened.

## Main run

`D:/DroneSoccerData/runs/trainable-memory-actor-20261008`, managed session41132;
stdout sibling `trainable-memory-actor-20261008.stdout.log`. Initial status:
preparing_public_features. One project-lock CUDA worker,0.15 allocator,1 CPU intra/inter-op thread,
strictFP32/noTF32. GPU admission8875MiB free/0% utilization; only UI Python processes
before launch. No other training cohort.

Five48-episode rounds, FIT11243760..11243999, algorithm11604000. Four epochs of
whole-four-episode minibatches (240 proposals maximum), total3600seconds including
preparation/evaluation. Hard source trust checks regenerate memory from original
whole FIT histories after every proposal. Final probes also regenerate current
memory. All final first-pass/return/clean-repeat/crash and retention criteria are
unchanged; a learned-memory-change check is additionally required.

Read only compact status/decision first at the next heartbeat. Let a healthy batch
finish; no repeated logs/tests/audits or edits to bound code/tests/plan. Review once
on completion/meaningful failure. Failure closes this architecture/budget; no
intermediate promotion or extension. Passing allows only a later fresh paired
screen, followed by the still-required sustained validation and independent acceptance.

Original85 remains selected; simulation incomplete; physical flight unvalidated.
No hardware, master changes, publication or outside messages. UI8794 still displays
the earlier three-arm replays. Heartbeat remains ACTIVE.
