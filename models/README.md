# Shared model components

| Model | Attribution | Purpose |
|---|---|---|
| [seenflag-s1-level1.zip](team-baseline/seenflag-s1-level1.zip) | Tom Storey / team, original Git commit `9d89c26` | Frozen 85-input SAC baseline |
| [gru64.pt](motion-estimator/gru64.pt) | PC recurrent-estimator experiment, 2026-10-08 | Frozen GRU, learned previous-interval motion estimate from public observation history |

The trainable PC correction weights are individually preserved under [policies/checkpoints](../policies/checkpoints). The [registry](../policies/registry.json) binds each correction to the exact two shared model hashes. These three components together constitute the PC policy.

No simulator position, true heading, or hidden ring location is supplied to the lab actor. The added motion values are network estimates, not measured ground truth. Telemetry is used by the flight safety checks and recorder, not silently added to actor inputs.

The lab adapter uses the same frozen estimated-motion actor/GRU implementation as the PC source snapshot, float32 CPU inference, and actual previously issued quantized RC history. Its receipt-based camera timing and team-default intrinsics have not been physically calibrated by this release.
