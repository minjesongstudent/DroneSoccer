# Boundary penalty trial and fresh development comparison

The bounded trial finished all 5,000 online updates with 5,000 accepted updates and zero rollbacks. The original constrained trial stopped after 100 consecutive rejected proposals. A differentiable penalty on the existing FIT observation guard prevented that stall in this run without changing the hard KL limits, actor inputs, physics or passage criteria. This is an optimizer result, not mission acceptance.

After identical 5,000 frozen-actor critic updates, the candidate used boundary penalty weight 0.1, source-anchored Gaussian KL limits (mean 0.01, maximum 0.1), and the same development retention guards. No retention guard failed. The original model is unchanged and remains retained.

## Existing development outcomes

Each suite contains 100 episodes; crashes count any crash during the episode, including after a first passage. Clean repeat requires at least two full-body passages, survival through the episode, and no wall contact.

| Suite | Model | First passage | Crashes | Return to front | Clean repeat |
|---|---|---:|---:|---:|---:|
| Aligned | Original | 84 | 16 | 84 | 84 |
| Aligned | Boundary | 85 | 15 | 85 | 85 |
| Nominal | Original | 80 | 26 | 80 | 74 |
| Nominal | Boundary | 81 | 24 | 81 | 76 |
| Randomised | Original | 73 | 52 | 70 | 48 |
| Randomised | Boundary | 72 | 49 | 70 | 51 |

The differences are small and mixed. In particular, randomised first-passage performance fell by one episode while clean repeat improved by three. These reused development cases do not establish a reliable improvement. The model is not promoted and training is not extended unchanged.

## Next diagnostic

A paired comparison is frozen before evaluation: original versus final boundary checkpoint, 100 identical layouts per suite, seeds 9510000–9510099 aligned, 9520000–9520099 nominal, and 9530000–9530099 randomised. These are fresh development seeds, not the reserved 7M acceptance seeds. Primary outcome is paired clean repeat; first passage, crash and return are secondary. Discordant episode counts and complete trajectories support inspecting exactly which failures changed. There is no automatic promotion, optimization or physical command in this evaluation.

Reproduce from the local repository using its Python environment:

```text
python -m tools.compare_boundary_candidate --out D:/DroneSoccerData/runs/boundary-fresh-development-20261007
```

The output directory must be new. The runner freezes models, source hashes and protocol before the first episode. Completion creates `DONE`; partial results alone do not mean completion.

Training evidence: `D:/DroneSoccerData/runs/constrained-boundary-20261007`, including source snapshots, partitions, checkpoints, replay, RNG and trajectories. Candidate: `online_005000/model.zip`. Training command: `python -m tools.run_constrained_clearance --out <new-directory> --guard-boundary-weight 0.1` (requires the preserved parent warmup replay).

The updated boundary implementation passed 25 focused checks and a CUDA smoke trial; the earlier implementation passed 124 relevant checks. The new comparison runner passed compilation. Simulation acceptance and physical flight remain unvalidated. Group master and remote branches were not changed.
