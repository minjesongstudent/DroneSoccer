# Alignment separates successful and failed first approaches

Offline comparison of the preserved paired development trajectories at first front-to-back crossings of the 30, 15 and 8 cm pre-ring distances. Wall-contact episodes were excluded; successful first passages and ring collisions before a credited passage were compared. Features use the position preceding the selected action, the current actor observation and the command actually issued. Two focused tests verify that timing and exclusions.

At 30 cm, original-policy median absolute lateral error was:

| Suite | Successful first passage | Ring collision |
|---|---:|---:|
| Aligned | 6.85 cm (175) | 15.14 cm (25) |
| Nominal | 4.43 cm (134) | 16.62 cm (26) |
| Randomized | 3.81 cm (120) | 13.93 cm (35) |

Counts differ from all clean-first-pass counts because episodes with any later wall contact were excluded. This retrospective selection is a limitation. Groups share distance thresholds, not necessarily identical layouts or approach angles. The 8 cm failure subset is especially selected because many collisions happen before that threshold.

The earlier hypothesis that blindness or strong yaw alone explains crashes is not supported: successful approaches also have zero median current visibility and nearly saturated yaw at 30 cm. Median blind durations are three steps for successful approaches across suites; failed aligned/nominal approaches have only one step, randomized three. At 15 cm, successful lateral error is about 2.8–5.9 cm, versus 9.2–12.1 cm in failures. These are associations, not causal intervention results.

Review of `src/sim.py::_to_go` found that the current progress reward switches to direct center distance when the drone center lies inside the ring opening ellipse. That predicate does not include the yawed guard-box extent, unlike collision checks. This is a concrete candidate mismatch between the reward's alignment criterion and whole-body clearance, not proof that it caused the failures.

Next: quantify this mismatch on saved FIT trajectories, then consider a bounded reward-only experiment using guard-aware alignment. Preserve collision and success criteria, actor inputs and original policy; do not clamp actions or change dynamics. If implementing changed rewards, reconstruct them consistently for replay and online transitions before any updates. No such training was launched by this audit.

Reproduce: `python -m tools.audit_approach_distances`. Output and input hashes: `D:/DroneSoccerData/runs/approach-distance-20261007/report.json`. Original policy retained, no model updates, no acceptance claim or physical-flight validation.
