# Matched PC training comparison

| Arm | Completed training episodes | Last checkpoint | Final randomized clean repeats | Final randomized crashes | Decision |
|---|---:|---|---:|---:|---|
| Full horizon | 240 | round 05 | 48/100 | 52/100 | Not qualified |
| Local credit | 240 | round 05 | 46/100 | 54/100 | Not qualified |
| Mission outcome | 96 | round 02 | Not run | Not run | Stopped for retention regression; intermediate only |

Full horizon final single passes: 73/100; return-to-front: 71/100. The 52% branch name describes a measured failure rate, not a deployment qualification. Acceptance and physical-flight flags remain false.

Evidence:

- `cohort_manifest.json`, `seed_allocation.json`: CUDA hardware, coordinator/worker budgets, source identities and reserved seeds.
- `decision.json`, `closure_review.json`: original no-promotion decisions, unchanged.
- Per-arm `result.json`: optimizer iterations, transitions, accepted/rejected updates, guard metrics and diagnostics. Per-arm protocols and decisions retain original PC paths as historical evidence.
- `training-trajectories.zip`: 576 training records and their complete saved trajectories, including successful passages and failures.
- `fitting-and-evaluation-records.zip`: rollout arrays, advantage inputs, optimizer states, probes and per-round guard evidence. Preserve seed roles when reproducing; do not fit on development or acceptance seeds.
- `source-snapshots.zip`: exact recorded trainer/runtime/source snapshots. Extract to a separate directory before reproducing; adapt only artifact locations from the original protocols. Source hashes in the manifests identify the originals.
- [All correction checkpoints](../../policies/CHECKPOINTS.md) and [shared models](../../models/README.md).

These archives preserve evidence for this comparison, not a claim that a different machine will produce bit-identical GPU training. Original absolute paths and input references are intentionally retained for provenance. Earlier prerequisite experiments remain in the pre-cleanup Git tree and PC run archive. Final screens are not the independent acceptance test. None of these results certifies physical flight.
