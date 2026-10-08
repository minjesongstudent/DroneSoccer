# Reward alignment ignores relevant guard geometry

Audited the original policy's preserved warmup only, stopping before any adapted-policy transitions. Completed 558 episodes (447 FIT, 111 held out), preserving every-fifth-family separation. Simulator source hashes match the original run; recovered resets match saved ring centers, normals and drone boxes exactly.

Among pre-action states within 40 cm in front of the ring whose centers satisfy the current direct-center reward branch:

| Split | Eligible states | Projected guard contacts | Fraction |
|---|---:|---:|---:|
| FIT | 10,907 | 2,081 | 19.1% |
| Held out | 2,463 | 562 | 22.8% |

The diagnostic translates the current box along the ring normal to the plane without changing yaw or lateral/vertical offset. It then applies the same 256-point noodle-to-box distance criterion used by the simulator. This is geometry analysis, not flight, actor input, or a predicted inevitable crash. It omits hanger contacts and assumes unchanged pose at the plane. Many current states can still recover before crossing. States within each episode are correlated.

Two focused tests verify agreement with native noodle collision geometry for 100 poses and invariance under normal-axis translation. All actual dataset movements remain those of the learned source policy. No actor or critic updates occurred.

This supports a bounded reward-only experiment, not a dynamics change or relaxed passage definition. Before training, implement the proposed alignment cost explicitly and reconstruct both previous and next geometry for replay rewards; verify online/offline reward parity. Keep public actor observations identical, retain original models, and use existing first-pass, return, repeat and crash checks. A hard geometry-dependent branch may introduce reward discontinuities, so inspect its deltas and avoid treating this diagnostic predicate as an automatically correct reward.

Reproduction: `python -m tools.audit_guard_alignment`. Evidence and input hashes: `D:/DroneSoccerData/runs/guard-alignment-20261007/report.json`. Original policy remains selected. Acceptance outstanding; physical flight unvalidated.
