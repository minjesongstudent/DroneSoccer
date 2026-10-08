# Fresh paired result: small gain with collision tradeoff

The frozen comparison completed 1,200 episodes on 600 new development layouts. The candidate is not promoted and its training is not extended unchanged.

| Suite (200 layouts each) | Original / candidate clean repeat | Original / candidate crashes | Candidate-only / original-only clean-repeat successes |
| --- | --- | --- | --- |
| Aligned | 175 / 175 | 25 / 25 | 0 / 0 |
| Nominal | 133 / 138 | 66 / 61 | 6 / 1 |
| Randomized | 98 / 102 | 101 / 97 | 18 / 14 |

First-pass counts were unchanged at 175 in aligned conditions, 149 versus 151 nominal and 142 versus 142 randomized. Returns were 175/175, 149/151 and 140/140. The candidate gains nine clean-repeat episodes across 600 layouts, but randomized paired results include fourteen regressions. These modest development differences do not demonstrate sustained qualification, a robust improvement or physical readiness.

The saved-trajectory audit further distinguishes collision types. Randomized net collisions dropped from 42 to 34, while ring collisions rose from 59 to 63. Before the first passage, ring hits increased 36 to 38; after a passage they increased 23 to 25. Net collisions after a passage decreased 20 to 14. This is a tradeoff, not a general solution to collision avoidance.

At most of these collisions, some ring information remained in the camera history. Randomized post-pass ring hits with entirely empty camera histories numbered one for the original and three for the candidate. Median terminal blind durations were nine steps for both policies' post-pass ring collisions. Such associations do not prove that visibility caused or prevented a collision.

Evidence is preserved in `D:/DroneSoccerData/runs/twelve-adapter-fresh-20261007` (frozen source/model hashes, protocol, every actual trajectory, paired results and DONE), and `D:/DroneSoccerData/runs/twelve-adapter-failure-audit-20261007` (read-only classification, input-log hashes and audit source).

```text
python -m tools.audit_boundary_failures --run D:/DroneSoccerData/runs/twelve-adapter-fresh-20261007 --out D:/DroneSoccerData/runs/twelve-adapter-failure-audit-20261007 --candidate-label adapter
```

No trainer remains active. The original source model remains selected. Before another actor update, the next diagnostic should check whether critic action preferences agree with measured learned-policy continuation outcomes. Use actual actions sampled from the learned policy and FIT layouts, preserve complete trajectories and RNG, and distinguish the critic's discounted entropy-adjusted target from the binary mission metric. Do not use a scripted controller, fabricate observations, use reserved acceptance seeds or treat a single noisy continuation as ground-truth expected value. This diagnostic is proposed, not yet executed.

Acceptance remains false. There were no physical operations, publications or master-branch modifications.
