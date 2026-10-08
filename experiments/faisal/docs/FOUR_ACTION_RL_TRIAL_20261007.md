# Four-command imitation result and reward-based test

The split-input implementation reproduced the original 75-episode initial guard exactly and completed 2,000 imitation updates. Final first/crash/return/repeat counts were aligned 84/16/84/84, nominal 80/28/80/72, randomised 74/49/72/51, each out of 100. These are mixed changes against original repeats 84/74/48. No promotion or acceptance.

The fourth command is available, but success-filtered imitation did not reliably improve behavior. The next bounded test uses full-clearance rewards with the existing uniform-replay, constrained SAC method on the explicit 89-input profile. Unlike imitation, this includes failed FIT trajectories and new learned-policy transitions. Collision-biased replay remains off.

Start again from the original source through the bit-preserving split-input transfer, not the imitation candidate. Replay is expanded from actual prior commands before complete-episode FIT/holdout partitioning. Actor and both critics receive the explicit 89D public history. The online wrapper derives the fourth command from the preceding public 85D history; no simulator state enters actor inputs. Base trajectory logs retain all 85D observations and issued actions, from which exact 89D observations are reconstructible. Replay checkpoints directly store 89D observations.

Preserve original gamma/tau, fixed source entropy, critic learning rate 3e-4, actor rate 1e-6, mean/max source KL limits 0.01/0.1, boundary penalty 0.1 and exact optimizer rollback. Initial parity required; 5,000 frozen-actor critic updates, at most 5,000 online transitions/updates, one-hour training cap plus bounded evaluation. Same 25-per-suite retention guards and final 100-per-suite evaluation. No weakened randomization or success metric.

29 focused tests passed, including online fourth-command reconstruction on actual learned steps and resets. CUDA smoke exercised critic adaptation, actor training, saved checkpoints and explicit 89D evaluation. No performance claim from smoke.

```text
python -m tools.run_constrained_clearance --out D:/DroneSoccerData/runs/four-action-rl-20261007 --four-actions --guard-boundary-weight 0.1
```

All historical models retained. The original 85D preview remains unchanged. No physical flight, publication, group-master changes or reserved acceptance-seed use.
