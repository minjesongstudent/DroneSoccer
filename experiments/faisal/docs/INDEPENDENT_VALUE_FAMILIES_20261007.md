# Independent state families with repeated learned continuations

The single-trajectory value diagnostic overfit even with simulator truth. The next bounded dataset distinguishes noise within a state/action continuation from variation between independent starting states. It is not a continuation of failed actor training.

Collect 64 surviving stochastic learned-policy prefix states from new FIT reset seeds starting at 11230000. Prefix lengths cycle through 20, 60 and 120 steps. At each selected state, sample four actions from the fixed twelve-command actor and run six paired stochastic continuations per action. The fixed maximum is 1,536 continuations, 384 prefix attempts and 30 minutes. Record skipped terminated prefixes as well as selected states. No previous ranking-audit states are reused.

Every movement, including the prefix, first branch action and continuation, comes from the learned policy. The actor receives only its public history and issued commands. Simulator snapshots support branching and diagnostics, not actor inputs. The actor is unchanged; no neural network training is included in this collection job.

The intended analysis is declared before inspecting outcomes:

1. Keep each complete selected state/reset family together. Use selected state index modulo five equal to four as held out; never split candidate actions, replicas or trajectory steps from one family between fitting and holdout.
2. Estimate within-state/action return variation from the six replicas, and compare action contrasts using paired replicas. Keep finite soft returns, timeout-bootstrapped returns and mission outcomes distinct.
3. Check agreement between replica halves before treating their mean as a precise expected-value target. Six samples can remain insufficient; do not silently describe means as ground truth.
4. Do not fit on the earlier diagnostic states or promote a flight policy from this data alone. Decide whether a subsequent value-fitting experiment is justified only after examining noise and family diversity. No automatic actor update follows collection.

```text
python -m tools.audit_critic_rankings --out D:/DroneSoccerData/runs/critic-independent-families-20261007 --seed-base 11230000 --states 64
```

Five focused return-accounting and nonmutating-truth tests passed. The existing audited collector now has explicit state-count and FIT-cohort arguments; its learned sampling, target accounting and trajectory preservation remain unchanged. The run preserves model/source hashes, prefix selection records, actual observations/actions/rewards/log probabilities, simulator states and RNG seeds.

The original flight policy remains selected. This is a bounded diagnostic collection on the PC, not physical flight, reserved acceptance or evidence of completion. No publishing or master changes are involved.
