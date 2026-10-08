# Critic rankings versus paired learned-policy continuations

A bounded diagnostic completed 288 continuations: twelve surviving FIT states, four actions sampled from the current twelve-command actor at each state, and six stochastic continuations per sampled action. All prefix actions, tested actions and subsequent movements came from that learned actor. No actor or critic was trained during this diagnostic.

The highest-Q sampled action had a higher mean realized return than the lowest-Q action in six of twelve states. The count was the same for finite returns and timeout-bootstrapped returns. Critic estimates were on average 24.25 return units below the sampled bootstrapped means, with RMSE 28.94 across the 48 state/action means. This is evidence that the current critic is not yet demonstrated to provide reliable action preferences. It is not proof that it caused all observed training failures.

The diagnostic preserves the SAC target distinction: the fixed first action has no entropy subtraction in Q(s,a); subsequent learned actions contribute discounted reward minus alpha times log probability. True terminations have no tail. Timeout continuations also report a tail estimated from the target critic and a learned next action. Unbootstrapped finite return and binary full-episode mission success are recorded separately. Approximately 39.2% of continuations reached timeout, so the bootstrapped statistic is partly dependent on the critic being examined.

Six paired continuations are a noisy estimate. Several apparent ranking reversals are smaller than their paired standard errors. The prefixes select surviving states after 20, 60 or 120 learned steps rather than representing all states uniformly. These public observations can also alias different hidden simulator states. The diagnostic therefore cannot establish globally unbiased Q values or a precise ranking accuracy. It does justify checking value calibration before another otherwise unchanged critic-driven actor update.

## Reproduction

```text
python -m pytest tests/test_critic_rankings.py -q
python -m tools.audit_critic_rankings --out D:/DroneSoccerData/runs/critic-ranking-20261007
python -m tools.summarize_critic_rankings --run D:/DroneSoccerData/runs/critic-ranking-20261007
```

Use a new output directory for a repeated run. Two focused return-accounting tests and a separate eight-continuation smoke passed before the full diagnostic. The full output contains frozen model/source hashes, selected and rejected prefix logs, actual prefix trajectories, pickled environment states, action-sampling RNG, per-replica RNG seeds, full observation/action/reward/log-probability/position sequences, outcomes, ranking summaries and a completion marker. Common future environment and policy randomness is paired across candidate actions within each replica; actors still receive only real public camera features and issued commands.

The model examined is the saved restricted twelve-command candidate. It remains unpromoted; the original 85-input source remains selected. Future value-fitting experiments should separate complete reset families, evaluate calibration and ranking on distinct states, and keep the actor frozen until a better value signal is demonstrated. Do not fit and claim improvement on these same twelve diagnostic states. Any value target change must explicitly preserve the distinction between finite mission outcomes and entropy-adjusted critic returns.

No trainer is active after this diagnostic. No hardware operation, publishing, group-master modification or reserved 7M acceptance-seed use occurred. Simulation acceptance remains false and physical flight remains unvalidated.
