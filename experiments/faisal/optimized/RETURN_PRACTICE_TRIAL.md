# Authentic return practice

The 40k route-reward trial ended without a repeat passage or a front-waypoint return. Its final centered/varied scores were 24/25 and 21/25, and full/delayed completion stayed zero. Its selected 10k checkpoint retains the stronger source behaviour and is the source for this next experiment. Do not extend the failed final policy blindly.

## One controlled change

The next training distribution uses 20% centered single-pass, 20% varied single-pass, 20% front-start full laps, 20% corner-start laps, and 20% authentic mid-flight return starts. All use one 100-input policy. The actor and its original frozen retention reference are transferred separately, and reward estimates/replay are rebuilt as in the prior experiments.

Return starts were captured after actual saved-policy passages. The complete environment, camera history, command memory, velocity, delay queue, physical clock and RNG are restored. No script chooses training actions, and no true position or reward phase enters the actor. The 200-step continuation keeps the same lap rewards/collisions and ends with a bootstrapable time-limit truncation. Old passages from the snapshot never count as new training passages, and short practice cannot promote the 60-second mission curriculum.

The training bank has 93 snapshots from 64 attempted flights (56 side entries, 37 later return entries). The separate development bank has 190 snapshots from 128 attempted flights (110 side entries, 80 later return entries). Training source seeds start at 2M; development source seeds start at 8M. The independent final 7M test remains unused. Snapshot and source hashes are checked on loading.

## Evaluation and stopping

Every 5,000 steps, run the original five fixed validation suites and 50 distinct development-source flights. Freeze the explicit development snapshot schedule across checkpoints and preserve it in the report. A return-practice success requires at least one NEW fully cleared passage and survival through all 200 continuation steps. A safe front return uses the physical outside back-to-front crossing and full-body clearance, not merely a reward waypoint label.

Save `best_return_model.zip` separately from the full-mission `best_model.zip`. Return selection retains the original centered >=84% and varied >=80% gates, then ranks new-pass success, safe front returns, and fewer crashes. It needs an improvement of at least 5/50 sustained across two checks to proceed beyond 20k; stop by 40k without an eligible new-pass gain. These development rules do not certify full mission completion. The original sustained validation plus fresh 100-layout final test still applies.

## Storage and launch

C: has only approximately 1 GB free. New model/replay/checkpoint data goes to `runs`; D: had over 800 GB free when checked. Existing artifacts are unchanged. Use a unique run name and unused port.

```powershell
..\DroneSoccer\.venv\Scripts\python.exe -m src.live --steps 40000 --eval-every 5000 --name return-practice-overnight-20260927 --port 8778 --runs-dir runs --source-checkpoint runs/return-route-40k-20260927/best_model.zip --source-config runs/return-route-40k-20260927/config.json --return-library return-libraries/train-v1 --return-validation return-libraries/development-v1
```

The source banks are local trusted generated artifacts, not files to accept from an outside sender. Keep them with the experiment record. The project tests and source are snapshotted with each run. Use the live `/state` response and saved evaluations to determine current progress; successful launch alone establishes no learning result.
