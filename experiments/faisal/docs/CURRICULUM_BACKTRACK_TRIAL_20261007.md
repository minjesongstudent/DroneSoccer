# Curriculum optimizer step correction

The original curriculum trial stopped after 100 rejected updates, with only six accepted. Final randomized first passage/repeat/crash counts were 72/47/53 per 100, versus original 73/48/52. It is rejected; no fresh performance comparison is warranted.

A bounded offline check of 16 discarded proposals from the saved first checkpoint found finite gradients but proposed maximum KL around 0.14–0.15, beyond the unchanged 0.1 limit. The accepted checkpoint maximum was only 0.0126. This supports trying smaller proposals rather than relaxing the guard. This check uses sequential saved batches, not an exact recreation of historical shuffled update order.

The authoritative check is `rejection_check_independent.json`. The earlier `rejection_check.json` is preserved but superseded: its optimizer-state template was shared across loads and mutated. The corrected check deep-copies that template for every independent proposal. No diagnostic actor was saved or promoted.

`guarded_optimizer_step` now optionally retries a rejected proposal at half the learning rate, at most four times. Each retry starts from the original weights and Adam state with the same computed gradient; a successful step retains its Adam state and restores the nominal learning rate for the next batch. Exhausted attempts roll back completely. Default behavior remains zero retries. The original KL and performance limits remain unchanged.

Thirteen focused tests passed, including a test that backtracking accepts a smaller step under the same guard with only one Adam step recorded. The CUDA smoke completed. A new bounded run starts from the original policy, with the same curriculum, source guard, seeds, 48 episodes per iteration, one actor epoch, five-iteration maximum and one-hour cap:

```text
python -m tools.run_onpolicy_trial --out D:/DroneSoccerData/runs/curriculum-backtrack-20261007 --approach-curriculum --backtracking --actor-lr 1e-5 --episodes-per-iteration 48 --actor-epochs 1
```

This is an optimizer correction prompted by measured rejected steps, not an unchanged extension. Judge by standard full-task first passages, returns, repeats and crashes. Reject regressions or no meaningful randomized improvement; do not weaken the KL guard if this also fails. Original model remains selected, simulation acceptance outstanding, physical flight unvalidated.

## Completed result: candidate not promoted

All five iterations completed on CUDA, with 240 complete learned-policy episodes and 267 accepted updates, zero rejected updates. The backtracking correction resolved the step-rejection problem within the existing KL limits. It did not establish a useful flight improvement.

Final fixed development results, 100 episodes per suite:

| Suite | First passage | Returned | Clean repeat | Crashed | Original clean repeat / crash |
|---|---:|---:|---:|---:|---:|
| Aligned | 84 | 84 | 84 | 16 | 84 / 16 |
| Nominal | 79 | 79 | 75 | 25 | 74 / 26 |
| Randomized | 75 | 74 | 49 | 51 | 48 / 52 |

The randomized gain is one clean repeat in 100, with one fewer crash. This is insufficient evidence to promote the candidate or extend the same run. The original model remains selected. No extra fresh cohort will be used to search for a favorable result from this variant. The reserved acceptance seeds remain unused.

Evidence: `D:/DroneSoccerData/runs/curriculum-backtrack-20261007/result.json`, `policy_05.zip`, saved rollouts and source snapshots. The result records `source_unchanged: true`. This closes the optimizer/curriculum trial, not the outstanding reliability work. Simulation acceptance and physical flight remain unvalidated.
