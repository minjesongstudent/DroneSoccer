# Parallel credit comparison: completed, no qualifying candidate

The three-arm CUDA comparison completed on 8 October Sydney. Full horizon retained the original clean-repeat and crash counts; local credit regressed; mission stopped at its second retention check. None met the frozen improvement gate. The original `seenflag-s1-level1` remains selected. No intermediate checkpoint, best failing arm, fresh development cohort or acceptance test was selected.

The [frozen plan](PARALLEL_CREDIT_COMPARISON_20261008.md) and [launch history](PARALLEL_CREDIT_LAUNCH_20261008.md) preserve the hypothesis, source identities, seeds, budget and selection rule.

## Final development outcomes

All counts below are episodes out of 100 per suite. Clean repeat requires at least two full-body passages and survival to the 600-step horizon without crash or wall contact. Return to front is a separate diagnostic.

| Policy | Suite | First passage | Return to front | Clean repeat | Crash |
|---|---|---:|---:|---:|---:|
| Full horizon | Aligned | 84 | 84 | 84 | 16 |
| Full horizon | Nominal | 80 | 80 | 74 | 26 |
| Full horizon | Randomized | 73 | 71 | 48 | 52 |
| Local credit | Aligned | 84 | 84 | 84 | 16 |
| Local credit | Nominal | 79 | 79 | 74 | 26 |
| Local credit | Randomized | 74 | 71 | 46 | 54 |

Original randomized reference: 73 first passages, 48 clean repeats and 52 crashes. Advancement required at least 53 clean repeats and at most 47 crashes, plus suite retention and feature-use checks. This is a development improvement screen, not sustained validation or independent acceptance. Both completed arms passed offline feature-use but failed improvement.

Matching aggregate counts do not mean identical behavior: Full horizon gained four randomized clean episodes and lost four; nominal gained one and lost one. Local credit gained five randomized clean episodes and lost seven; nominal gained two and lost two. All 1,032 residual weights changed, and no optimizer updates were fully rejected. This does not justify simply increasing training time or update size.

Mission has **no final 100-case result**. At round two its randomized 25-case guard fell from 13 to 9 clean repeats and rose from 12 to 16 crashes, exceeding both unchanged regression limits of three. First passages stayed 18/25 and returns 17/25. The remaining 144 FIT episodes and final evaluation were correctly skipped. The four lost cases match those lost by the preceding estimated-motion residual.

## Why a good replay can belong to a rejected experiment

Four saved passages demonstrate learned-policy ability in that episode. Reliability is evaluated across all starts and ring positions. The viewer replays saved training episodes; the final development table is a separate fixed evaluation of checkpoint five.

For example, Full horizon randomized evaluation seed 9430000 completed full-body passages at steps 50, 158, 277 and 398, then crashed at step 510. All four passages count, but that episode is not a clean completed mission. There are also successful flights; all 100 contribute to the decision.

## Execution and preserved evidence

| Arm | FIT episodes | Transitions | Accepted / rejected updates | Worker seconds |
|---|---:|---:|---:|---:|
| Full horizon | 240 | 69,593 | 274 / 0 | 849.313 |
| Mission | 96 | 31,875 | 125 / 0 | 338.828 |
| Local credit | 240 | 69,836 | 274 / 0 | 931.203 |

Coordinator duration: 940.015 seconds, all workers closed, no coordinator errors. Queue duration including smoke and admission: 988.14 seconds. This does not establish speedup against a matched serial run. Each worker recorded under 69 MB peak PyTorch allocation, excluding contexts and other GPU users.

One independent saved-artifact closure verified 3,639 unique file hashes, original actor/GRU identities, 38 coordinator and 38 per-worker source bindings, seed budgets, final outcomes and selection. The five saved first-round arrays and endpoints matched exactly before updates: 48 episodes and 16,592 transitions per arm. No additional inference or rollouts were run for closure or diagnosis.

Evidence root: `D:/DroneSoccerData/runs/parallel-credit-20261008`. Preserve `source`, manifests, three `arms`, all checkpoints/trajectories, `closure_review.py/json`, and `saved_data_diagnosis.py/json`. The audit scripts use saved arrays and hashes; their existing outputs must not be overwritten. The queue's launch protocol and source snapshots preserve exact configuration.

- Parent decision SHA-256: `133f275c4f0fbf99b7e69ea8d4b7fb0f028dc5c6fd73a2f9d11dc57f1b753050`.
- Closure report SHA-256: `7e333c96fb7184af756241e71a4830b70e21f0d5d0bd7a6c66cd5cf5c24709d4`.
- Saved-data diagnosis SHA-256: `847e9fca85de7f1e9f1e5cac1a5bd914cfc2b06af4733e3df3fb769d000feae4`.
- Selected original SHA-256: `8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef`.

These credit variants are closed. Do not extend budgets, retry seeds or select an intermediate result. A new trial requires a distinct mechanism supported by saved failures and a frozen go/no-go protocol. This negative result does not prove a hardware-data blocker or impossibility of the simulation task. Simulation acceptance and physical flight remain unvalidated.
