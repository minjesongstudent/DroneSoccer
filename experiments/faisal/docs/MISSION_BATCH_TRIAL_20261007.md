# Mission-target result and episode-gradient audit

The direct mission-target trial completed five iterations and 321 accepted updates. Final first/crash/return/repeat counts per 100 were aligned 83/17/83/83, nominal 79/26/79/74, randomized 74/50/71/50. These remain mixed versus original repeats 84/74/48. No promotion and no unchanged extension.

A CUDA diagnostic recomputed source-policy action likelihoods on the first complete rollout and matched the saved likelihoods. It then computed one score-function gradient per episode using the binary mission outcome and a leave-that-episode-out constant baseline. No actor update occurred. Sixteen original-policy episodes included seven successes. Mean gradient norm was 502.85 versus estimated mean sampling-noise norm 516.80. Across 200 reused random half-splits, median gradient cosine was -0.0055, 10th–90th percentiles -0.2133 to 0.1433, with 51% negative. These reused splits are diagnostics, not independent confidence intervals. They indicate weak direction consistency in this small batch, supporting a sample-size test rather than another optimizer change.

Next run starts from original policy with 48 fresh complete learned episodes per iteration and one actor epoch, replacing 16 episodes reused for three epochs. This roughly holds the amount of actor minibatch work per iteration while increasing distinct episode evidence threefold. Exact counts differ with episode length. Five iterations maximum, at most 144,000 actual transitions, same one-hour training cap and bounded evaluations. No failed candidate continuation.

Binary mission target, original 85D actor, learning rate 1e-5, fitted value baseline, source KL bounds/rollback, retention guards and final 100-per-suite evaluation stay unchanged. All FIT seeds, episodes, actions, old likelihoods and targets are preserved. Initial source likelihood matching and saved gradient arrays/audit source are under `D:/DroneSoccerData/runs/episode-gradient-audit-20261007`.

```text
python -m tools.run_onpolicy_trial --out D:/DroneSoccerData/runs/mission-batch-20261007 --actor-lr 0.00001 --objective mission --episodes-per-iteration 48 --actor-epochs 1
```

No acceptance claim, weakened metric, simulator-truth actor inputs, scripted action, physical operation, publication or group-master change. Original policy remains retained.
