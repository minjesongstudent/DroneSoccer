# Fixed continuation diagnostic completed: value accuracy did not resolve ranking

The planned 24 replicas per action are complete: 12 states, four learned-policy actions per state and 1,152 unique continuations. The original six and additional eighteen replicas were merged into a new analysis directory after checking complete replica IDs, unique state/action/replica keys, identical model hashes and consistent Q scores. Original outputs remain preserved. Offline rescoring generated no extra flights.

| Quantity | Pre-calibration critic | Prespecified calibrated critic |
| --- | --- | --- |
| Bias against sampled bootstrapped means | -35.14 | -5.92 |
| RMSE against sampled bootstrapped means | 38.41 | 15.61 |
| States where highest-Q action outperformed lowest-Q action | 5/12 | 6/12 |

Finite-return and bootstrapped ranking counts agree. Some reversals persist despite additional replicas: in state 2, the preferred action's mean bootstrapped return is 22.65 lower than the lowest-ranked action, paired standard error 6.79, for both critics. In state 1, calibration reverses a correct ranking: preferred-minus-lowest mean -0.315 with standard error .109. These are descriptive diagnostics on a small selected state set, not multiple-comparison-adjusted significance claims or global ranking estimates.

The result supports retaining the distinction between value calibration and useful policy gradients. Matching average returns more closely did not establish a reliable action preference. No actor update from the calibrated critic is authorized by this experiment's evidence. Stop the precision extension at its declared budget; do not add replicas or select a different checkpoint until the result looks favorable.

Evidence:

- `D:/DroneSoccerData/runs/critic-ranking-combined-20261007`: original critic and all 1,152 outcomes.
- `D:/DroneSoccerData/runs/critic-ranking-combined-calibrated-20261007`: offline calibrated scores on those identical outcomes.
- `tools/merge_critic_replicas.py`, `tools/rescore_critic_rankings.py` and frozen copies in the output directories reproduce assembly and scoring.

All actor, target-critic, entropy and discount equality checks passed before outcome reuse. The original learned trajectories and input/model hashes remain available in the two collection runs. No reserved acceptance seeds were used.

The next useful diagnosis is to distinguish hidden-state ambiguity from deficient action-value learning. A simulator-state-informed critic may be examined as a training-only diagnostic, compared against the public-history critic on separate reset families. Its simulator features must be explicit, excluded from every actor and deployment input, and never presented as measured hardware observations. This diagnostic has not been implemented or run. The current evidence does not prove that hidden-state ambiguity is the cause.

The original flight policy remains selected. No trainer or evaluation is active after this work. Simulation acceptance is false; physical flight remains unvalidated. The heartbeat remains active for further justified work.
