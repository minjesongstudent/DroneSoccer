# Full-clearance reward trial: candidates rejected

Both bounded CUDA arms completed their predefined regression stops. Neither
improved the retained policy. No model was promoted, no simulation acceptance
was achieved, and no physical flight was attempted. No trainer remains running.

The original `seenflag-s1-level1` checkpoint remains unchanged, SHA256
`8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef`.

## Final development results

Each row uses the same 100 development episodes for that suite, up to 600 steps
per episode. Passage means collision-free whole-body clearance. Crashes after a
successful first passage are still counted. Clean repeat means at least two full
passages and survival without contact to the episode limit.

| Policy | Suite | First full passage | Crashes | Clean repeat episodes |
|---|---|---:|---:|---:|
| Original, unchanged | Aligned | 84/100 | 16/100 | 84/100 |
| Original, unchanged | Nominal | 80/100 | 26/100 | 74/100 |
| Original, unchanged | Randomized | 73/100 | 52/100 | 48/100 |
| Control, 10,000 new updates | Aligned | 88/100 | 98/100 | 2/100 |
| Control, 10,000 new updates | Nominal | 66/100 | 90/100 | 10/100 |
| Control, 10,000 new updates | Randomized | 55/100 | 72/100 | 26/100 |
| Full-clearance reward, 1,000 new updates | Aligned | 84/100 | 100/100 | 0/100 |
| Full-clearance reward, 1,000 new updates | Nominal | 46/100 | 100/100 | 0/100 |
| Full-clearance reward, 1,000 new updates | Randomized | 38/100 | 100/100 | 0/100 |

These final checkpoints have different update counts because the retention stops
triggered at different times. They must not be presented as equal-budget endpoints.
The original 300-episode results are from 6 October; this run exactly reproduced
all 75 guard episodes, including every saved observation, action and trajectory
array. All trials use the unchanged group simulator and success criteria.

## Matched early check and interpretation

At the same 1,000-update checkpoint, control first-pass counts were 19/18/18
and crashes 6/9/12 out of 25 aligned/nominal/randomized episodes. Treatment
first-pass counts were 20/12/6 and crashes 25/25/25. Both arms had exactly the
same 200,000 warmup transitions and unchanged source weights before learning;
hashes verified equal observations, next observations, actions and terminal flags.
Only reward accounting differed during that shared warmup.

Thus, moving the reward alone did not solve the problem with this fine-tuning
setup. It caused a much earlier regression in this seeded comparison. The control
also regressed by 10,000 updates, showing that simply extending existing SAC
training is unsuitable here. This single comparison does not establish that
full-clearance rewards are intrinsically inferior with other training methods.

A read-only actor audit used 31,634 saved source-development observations. Warmup
models had zero output drift. After 1,000 updates the control's mean Gaussian KL
from the source was about 0.46-0.49 per suite; treatment was about 13.2-13.7. Mean
absolute deterministic action changes were about 0.054-0.057 for control and
0.247-0.257 for treatment on the normalized [-1,1] action scale. Large policy drift
accompanied the regression; this audit alone does not identify its cause.

The next justified work is to limit actor changes while adapting the critic to
the revised reward, using separate FIT data and explicit rollback checks before
long training. Development histories used for this audit must stay out of
optimization. Critic adaptation and constrained actor updates are proposed next
experiments, not completed fixes. No additional lab recording is needed to
investigate this learning failure.

## Reproducibility and preservation

Run: `D:/DroneSoccerData/runs/clearance-reward-20261007`, from
2026-10-06 16:09:04 UTC to 16:20:48 UTC (7 October Sydney). One active CUDA trainer;
control and treatment ran sequentially. Control collected 210,000 transitions
with 10,000 gradient updates; treatment collected 201,000 with 1,000 updates.
The 200,000-transition warmups used stochastic learned actions, not random
control or scripted flight. Every actor observation used the documented 85-input
camera/action history; truth was used only for reward and diagnostics.

The run directory preserves the frozen protocol and source/model hashes,
checkpoints with optimizers, replay buffers, RNG/environment/collector states,
all training and evaluation trajectories, reference reproduction audit, final
results, and `analysis/actor_drift.json` with its standalone audit script.
The software smoke trials and the first failed setup attempt remain preserved.
120 relevant repository checks passed; two existing Gymnasium observation-bound
warnings were unchanged. Final CUDA smoke verified actual gradient updates and
checkpoint reloads.

Reproduce the trial from its `source` folder with the recorded Python/dependency
versions, using a new output directory:

```text
python -m tools.run_clearance_experiment --out <new-directory>
```

Reproduce the diagnostic audit from the publication checkout:

```text
python -m tools.audit_clearance_drift D:/DroneSoccerData/runs/clearance-reward-20261007
```

The group master, remote branches, Mac packages and original policy were not
modified or published by this trial. The recurring heartbeat remains paused.
Physical flight remains unvalidated.
