# Fresh comparison and collision-approach replay trial

The fresh 600-episode comparison completed. Clean-repeat counts (original / boundary) were aligned 76/76, nominal 71/70, randomised 54/54. Randomised paired successes changed in both directions: four candidate-only and four original-only. There is no reliable improvement and no promotion. Original remains retained.

Read-only trajectory audit: original aligned failures were 24 hoop hits before the first full passage; all retained some ring observation in their 24-step history, with median four immediately preceding blind steps. Original randomised failures included 24 before-first-passage crashes and 22 after-passage crashes. Only three of the latter had empty ring history at the final action. This association does not establish causality, but does not support choosing longer memory alone as the next fix.

Next bounded experiment changes replay sampling, starting again from the unchanged original model and original learned-policy warmup. Half of each training batch is sampled uniformly from FIT replay, half from the last 16 actual transitions before FIT non-timeout episode termination. Windows never cross episode boundaries. Holdout episodes remain excluded. This changes the training distribution for both critic and actor; it is not an unbiased estimate of the uniform-replay objective. It tests whether emphasizing collision approaches yields a useful policy change under the same reference constraints.

No rewards, terminal flags, camera observations, actions, simulation parameters or acceptance rules change. All online movement remains learned. Fixed risk indices cannot be overwritten within the bounded replay budget; the sampler fails closed if the buffer fills. Source-anchored mean/max KL guards, boundary weight 0.1, learning rates, frozen entropy, 5,000 critic updates, at most 5,000 online updates, development retention guards and one-hour training cap are unchanged. No extension of the previous candidate.

27 focused checks passed, and a CUDA smoke completed eight critic and sixteen actor updates with checkpoint reload and evaluation. Smoke outcomes are not performance evidence.

```text
python -m tools.run_constrained_clearance --out D:/DroneSoccerData/runs/risk-replay-20261007 --guard-boundary-weight 0.1 --risk-replay
```

Preserved evidence: `D:/DroneSoccerData/runs/boundary-fresh-development-20261007/results.json`, `failure_audit/audit.json`, all 600 trajectories, frozen sources/models/protocol; and `D:/DroneSoccerData/runs/risk-replay-smoke-20261007`. New trial saves risk FIT indices, replay partitions, checkpoints, full learned trajectories and source hashes. Reserved acceptance seeds remain unused. No physical operation or publication.
