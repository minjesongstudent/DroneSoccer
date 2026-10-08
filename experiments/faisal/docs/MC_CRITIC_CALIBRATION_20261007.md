# Actor-frozen Monte Carlo critic calibration

The preceding twelve-state continuation audit found pessimistic critic estimates and no clear action-ranking advantage. This experiment tests value calibration before any further actor training.

Collect 80 complete stochastic learned-policy episodes at FIT seeds 11210000-11210079 with the restricted twelve-command candidate's actor frozen. Fit only the two critic networks to recorded full-episode soft returns. Every fifth whole episode is held out, keeping reset families separate. Neither the previous twelve diagnostic states nor reserved acceptance layouts are fitting data.

Targets exclude the current action's entropy and include subsequent discounted entropy-adjusted rewards. True terminal states have no tail. At timeout, a tail is sampled once from the unchanged original target critic and actor, then frozen for all regression updates. Finite, unbootstrapped returns are also retained. This reduces repeated bootstrap dependence but does not remove observation aliasing or make timeout targets independent of the old critic. The actor and target critic are checked bitwise unchanged at every checkpoint.

The run uses CUDA, 3,000 Adam critic updates, learning rate .0003, batch size 256, gradient norm cap 10, and a 30-minute wall cap. Checks at 0/500/1500/3000 report fitting and held-out RMSE and bias against fixed targets. This is critic-only calibration, with no automatic policy update or promotion. Saved model/optimizer states, target data, complete trajectories, source hashes, split and RNG are retained.

```text
python -m pytest tests/test_mc_critic.py tests/test_critic_rankings.py -q
python -m tools.calibrate_mc_critic --out D:/DroneSoccerData/runs/mc-critic-20261007
```

Four focused tests passed. A numerical cancellation discrepancy in the reverse-return unit test was fixed by accumulating in double precision before storing float32 targets; the initial smoke and corrected second smoke are both preserved. The corrected smoke completed collection, critic fitting, and actor/target equality checks.

Lower prediction error alone will not authorize actor updates. The prespecified final 3,000-update critic must also be compared with the pre-calibration critic on new paired continuation states starting at FIT seed 11220000, using identical actors, actions, prefixes and random streams. These states are distinct from both calibration and the earlier diagnostic. Retain finite and bootstrapped results separately. No checkpoint selection based on that new ranking set is allowed in this experiment.

The original flight policy remains selected. No physical operations, publishing, master changes or simulation acceptance claim are included.
