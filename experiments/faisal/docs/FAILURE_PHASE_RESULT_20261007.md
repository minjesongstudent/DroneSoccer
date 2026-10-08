# Paired deterministic failure phases

Audited all 1,200 saved development trajectories from `twelve-adapter-fresh-20261007` without additional flight or training. Recreated resets solely to recover randomized ring thickness; simulator, parameter and contract source hashes match the preserved evaluation sources, and each reset's ring center, normal and drone box match its trajectory exactly.

## Findings

Each suite contains 200 episodes per policy. The original and adapter both have 25 aligned crashes, all ring contacts before a credited first full passage. During the ten commands before these collisions, mean absolute right/forward/yaw commands are approximately 0.96/0.99/0.93. Median terminal blindness is four observation steps. Saturation is an association, not yet a causal explanation; successful passages need a matched comparison before imposing any action penalty.

In randomized episodes, original versus adapter:

| Terminal failure phase | Original | Adapter |
|---|---:|---:|
| Ring contact before credited first passage | 36 | 38 |
| Ring contact after returning to front | 22 | 22 |
| Ring contact during departure or return | 1 | 3 |
| Net contact after returning to front | 19 | 14 |
| Other net contacts | 23 | 20 |

Ring-contact median terminal blindness is 9–10 steps on approaches, versus zero for the main net-contact groups. The adapter's reduced net contacts do not remove the ring-contact failure mode. Nominal crashes decrease from 66 to 61; aligned crashes remain 25; randomized crashes decrease from 101 to 97. These are the existing development results, not new evidence of acceptance.

The phase labels are geometric diagnostics based on saved full-body passage steps and front-side reach. A plane-slab location is not a legal crossing. First wall-contact timestamps are absent, so this audit does not claim to identify the first unsafe event in episodes with wall contact. “First approach” means before a credited passage, which may differ from an attempted crossing after contact.

## Next justified analysis

Compare successful and failed approaches at matched distances from the ring, using pre-action camera history and actual issued commands. Establish whether lateral/yaw timing or visibility history distinguishes failures before designing an intervention. Do not assume that reducing speed, suppressing yaw or substituting a controller will fix them. No actor update is supported by this association alone.

Reproduce with `python -m tools.audit_failure_phases`. Output: `D:/DroneSoccerData/runs/failure-phase-20261007/report.json`, containing trajectory hashes, all episode classifications and paired gain/loss records. The tool and report remain local. Original policy retained; simulation acceptance outstanding; physical flight unvalidated.
