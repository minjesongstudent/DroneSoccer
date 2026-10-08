# Four-command reward result and on-policy test

The 89D reward trial completed 5,000 accepted updates, zero rejected updates and no guard stop. Final first/crash/return/repeat counts per 100 were aligned 85/15/85/85, nominal 81/25/81/75, randomised 72/53/70/47. This does not improve overall reliability against original repeat counts 84/74/48. No promotion; original 85D policy retained. Adding the fourth command alone has not established a benefit under either tested update method.

Several critic-driven and successful-trajectory imitation trials now show small mixed changes or regressions. The next test changes the update estimator rather than extending them. Collect complete new stochastic learned-policy episodes, compute discounted reward-to-go, and use clipped likelihood-ratio actor updates with a separate fitted state-value baseline. SAC Q action gradients and old replay rewards do not enter the update. Old FIT replay supplies only the existing reference guard observations.

Start from original 85D source. At most five iterations of sixteen complete episodes (at most 48,000 actual learned transitions); each iteration uses three actor epochs, batch size up to 256, ratio clip 0.1, actor learning rate 1e-6. Separate 85→128→128→1 value network, learning rate 3e-4, 100 minibatches per iteration. Advantages are detached and normalized. Full-clearance rewards unchanged. Finite 600-step reward-to-go ends at timeout without bootstrap: this deliberately tests the finite-episode objective used for mission evaluation, unlike SAC's timeout bootstrapping. It does not change evaluation criteria.

Maintain source KL mean/max 0.01/0.1, boundary penalty 0.1, exact actor/optimizer rollback, 100 consecutive rejection stop, one-hour training cap, and unchanged 25-per-suite regression guards after each iteration. Final 100 per suite. FIT reset seeds begin 11150000, algorithm seed 11540000; reserved acceptance seeds and development trajectories excluded from updates.

Each complete rollout saves actual observations/actions, old log probabilities, episode boundaries and returns, plus original geometry/reward traces. Source/model hashes, model/optimizer checkpoints, value network and RNG are preserved. No automatic resume or promotion. Tests cover discounted return boundaries and both signs of clipped advantages; a CUDA smoke collected actual learned flights, updated the actor and reloaded saved policies for evaluation. No physical operation or publication.

```text
python -m tools.run_onpolicy_trial --out D:/DroneSoccerData/runs/onpolicy-20261007
```
