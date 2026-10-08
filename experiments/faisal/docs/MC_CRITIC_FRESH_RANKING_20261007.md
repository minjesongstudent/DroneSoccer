# Calibrated critic: better values, unresolved action ranking

The fresh diagnostic completed 288 learned-policy continuations on twelve new FIT states at seed base 11220000. The prespecified 3,000-update calibrated critic was scored offline on the identical saved states/actions. Actor parameters, target-critic parameters, entropy coefficient and discount were verified equal before reusing outcomes. No additional flight or training was performed by rescoring.

| Diagnostic | Before calibration | After calibration |
| --- | --- | --- |
| Q minus sampled bootstrapped mean bias | -36.09 | -6.88 |
| RMSE against sampled means | 40.02 | 17.22 |
| States where highest-Q action beat lowest-Q action | 4/12 | 6/12 |

Finite-return and timeout-bootstrapped ranking counts agreed. Calibration generalized better in value magnitude, but these twelve states do not demonstrate reliable action rankings. Many top-minus-bottom return differences are smaller than their paired standard errors. No further actor update is justified by these results alone.

To reduce continuation noise, collect eighteen additional paired replicas for each of the same four actions in all twelve states: replica IDs 6-23, supplementing the original 0-5. This is 864 additional learned-policy continuations and a fixed total of 24 replicas per action. Do not repeatedly extend this diagnostic until a desired result appears. Stop at the declared total and interpret remaining uncertainty candidly.

The extra run checks the frozen model hash and exact prefix observations/actions, selected observation and each sampled first action against the original run before executing additional continuations. All subsequent actions still come from the unchanged learned policy. Shared per-replica future simulator and policy randomness remains paired across candidate actions. There are no fabricated observations, actor updates or reserved acceptance layouts.

```text
python -m tools.rescore_critic_rankings --run D:/DroneSoccerData/runs/critic-ranking-fresh-20261007 --out D:/DroneSoccerData/runs/critic-ranking-calibrated-20261007 --model D:/DroneSoccerData/runs/mc-critic-20261007/critic_003000.zip
python -m tools.audit_critic_rankings --out D:/DroneSoccerData/runs/critic-ranking-extra-20261007 --seed-base 11220000 --replicas 18 --replica-start 6 --reference-run D:/DroneSoccerData/runs/critic-ranking-fresh-20261007
```

The baseline and calibrated outputs are preserved separately with input/model hashes and summaries. Once the extra run completes, combine the two outcome sets using unique (state, candidate, replica) keys, require IDs 0-23 for each action, and apply both critics to the same combined outcomes. Preserve the original files. Four return-accounting tests passed after adding the bounded replica extension.

The original flight policy remains selected. No GPU trainer is active; only the bounded diagnostic runs. Simulation acceptance remains false and physical flight unvalidated.
