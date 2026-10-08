# Self-imitation result and dynamics sensitivity diagnostic

The successful-FIT imitation trial completed 2,000 updates without rollback or a retention stop. It did not establish useful improvement. Final aligned/nominal/randomised counts per 100 episodes were first passage 84/80/75, crashes 16/27/51, return to front 84/80/72 and clean repeat 84/73/49. Original clean repeat was 84/74/48. No promotion and no unchanged extension.

Several bounded local update methods have now retained or degraded the original behavior without resolving robustness. Before another training change, a paired sensitivity diagnostic separates command-delay variation from response-time variation.

Use the original unchanged learned policy on the same 100 randomized development layouts (9530000–9530099) under four conditions: normal randomization, delay fixed at the existing nominal two steps, response time fixed at existing nominal tau=0.55, and both fixed. All other reset draws, parameters, camera inputs, collisions, rewards and full-body passage metrics are unchanged. Dynamics overrides occur after reset so the random layout and observation draws match. All movement remains produced by the learned policy.

These fixed-dynamics conditions are diagnostic counterfactuals, not mission validation. They must never replace the original randomization range or be counted as acceptance. Improvements under them would identify sensitivity, not a better policy. The normal baseline must reproduce every saved array of all 100 previous original-model trajectories exactly; the runner stops if any differ. Source/model hashes, protocol, complete trajectories and paired repeat gains/losses are saved.

```text
python -m tools.probe_dynamics --out D:/DroneSoccerData/runs/dynamics-probe-20261007
```

Compilation passed. Completion requires the `DONE` file. This bounded 400-episode diagnostic uses CPU inference; no trainer or physical drone runs during it. Reserved acceptance seeds remain unused. Group master and remote branches remain unchanged.
