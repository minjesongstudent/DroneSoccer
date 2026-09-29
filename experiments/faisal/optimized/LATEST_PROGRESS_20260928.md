# Latest results: 28 September 2026

The goal remains one policy that fully clears the hanging ring, returns around its outside, passes again and survives the600-step mission in the2.05m cage. Physical flight is unvalidated. No checkpoint has met simulation acceptance.

| Experiment | Centered /25 | Varied /25 | Front returns /25 | Full mission /25 |
|---|---:|---:|---:|---:|
| Recurrent supervised basic transfer,300 updates |25|20|0|0|
| Recurrent RL fine-tuning,32,768 steps |25|23|0|0|
| Whole-lap supervised transfer,300 updates |25|22|2|0|
| Whole-lap supervised transfer,500 updates |25|21|3|0|

Front returns are recorded by the front-start lesson; they are intermediate events, not successful missions. At500 updates the front-start crash rate was22/25; general and delayed full-task success also remained0/25. These25-layout comparisons are small fixed development tests, not independent acceptance results. Three sustained validation checkpoints and a fresh100-layout-per-suite acceptance test remain required; independent7M seeds have not been used.

## Development since the previous export

1. Continued controlled SAC comparisons: balanced replay, source anchoring, geometric return recognition and reset coverage did not solve repeated laps. Failed trials and regression stops were preserved rather than presented as improvements.
2. Collected authentic learned flights and separated fit layouts from reserved checks. A2048-flight collection produced11 complete repeated-lap successes:9 fit and2 reserved. This low success rate is the denominator for the selected successful examples, not evidence of a reliable producer.
3. Investigated finite-return critic fits and bounded on-policy updates. Generalization and policy-change checks rejected treatments that did not establish improvement. Historical scripts, reports and protocols are exported as evidence.
4. Tested maintained RecurrentPPO with119 public observation values and correctly carried LSTM state. A fresh mixed-task run and subsequent close-start runs learned no validated passes. One launch failed due to memory allocation; historical replay servers were closed and the bounded run restarted. The failure was preserved.
5. Diagnosed the close lesson: the camera detector saw the ring in0/100 close resets,33/100 near resets and100/100 centered/varied resets. The preserved SAC policy passed25/25 centered diagnostic flights but0/25 close starts. Camera geometry was not changed to hide this limitation. Centered-start recurrent training alone still did not learn passes within its bounded trial.
6. Initialized the recurrent actor by supervised imitation of a previously learned SAC policy. All actions came from that learned policy.80 fit and20 held-out complete flights were retained, including failures; teacher passes were72/80 and18/20. The recurrent student reached25/25 centered and20/25 varied validation passes at300 updates. This is supervised initialization, not pure reinforcement learning.
7. Fine-tuned with PPO using centered/varied single-pass rehearsal and front-start laps. The first8192 steps trained only the critic; an actor hash verified unchanged movement weights. Subsequent RL retained25/25 centered and23/25 varied passing but stopped at32,768 steps for no front-return learning. Trajectories showed post-pass drift toward cage walls.
8. Verified all nine fit lap trajectories against the current environment: complete119-value observation histories matched exactly and each ended with two passes and600-step survival. Used those whole histories for supervised recurrent transfer with basic-skill rehearsal. Reserved trajectories were excluded. At500 updates, three front returns appeared in25 development flights, but no full mission succeeded. Basic passing declined slightly to21/25 varied; this checkpoint is an experimental artifact, not a finished controller.

## What is included

Updated source and tests, recurrent dependency pin, four recurrent checkpoints with SHA256 manifest, training datasets for supervised transfer, checkpoint evaluations, protocols and selected diagnostic trajectories under evidence/2026-09-28. Earlier evidence remains intact. Historical runners retain local archive paths and are provenance records; they are not portable one-command training recipes. Large replay buffers and all unsuccessful collection trajectories remain in local archives. Exported text replaces the workstation project-root path with PROJECT_ROOT; original and exported hashes identify that metadata transformation.

## Reproduce saved-policy evaluation

Run from this directory with Python3.12 and CUDA PyTorch, after installing requirements-tested.txt and requirements-recurrent.txt:

```powershell
python evaluate_recurrent.py --model models/recurrent-lap-transfer-500.zip --output evaluation-500.json
```

This uses the unchanged fixed development suites and authentic recurrent episode resets. It does not start training or connect to a physical drone. The model's stored RL step count and its supervised update count are different quantities; use the checkpoint manifest and experiment protocol for provenance.

## Next work

Inspect the three front-return trajectories to determine why the second passage fails, compare the300- and500-update checkpoints, and test any justified change with single-pass retention and full-flight checks. Keep actual camera/telemetry availability, camera calibration, latency and motion-response measurements on the lab checklist. The PC is intended to run inference and transmit commands; this does not supply unseen position or wall measurements.
