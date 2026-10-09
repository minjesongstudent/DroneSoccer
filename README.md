# DroneSoccer — PC policies and lab experiments

This research branch contains Faisal's PC-trained policy checkpoints, their shared model components, and the evidence behind them. **No checkpoint is flight-qualified.** Group master is unchanged.

| Section | Contents |
|---|---|
| [Policies](policies/README.md) | Every saved checkpoint, measured results, recorded simulator replay, and the real-flight recorder |
| [Models](models/README.md) | Clearly attributed team baseline and PC motion estimator required by the composite policy |
| [Experiments](experiments/README.md) | Training, evaluation, immutable source snapshots, and provenance audit |

## Important correction

The older `Mac-Single-Click-Policy-Flight` ZIP contained Tom Storey's unchanged `seenflag-s1-level1` baseline. It did **not** include the PC-trained correction or motion estimator. The earlier claim that the lab package was the PC-trained policy was incorrect. [Evidence and exact hashes](experiments/provenance/IDENTITY.md).

The actual PC policy is a composite: **team baseline + learned motion estimator + PC-trained correction checkpoint**, with 89 actor inputs. A residual `.npz` by itself is not an SB3 model and cannot replace a `.zip` in `SAC.load`.

## Group lab package

[Download PC-Policy-Lab-Search.zip](policies/PC-Policy-Lab-Search.zip). Extract the whole folder. Inside `policies/lab-runner`, run **1-Setup.command** on normal internet, then read **START-HERE.txt**. `2-Supervised-Flight.command` opens the real flight controls; it sends takeoff, a separately recorded bounded upward/turning search, then actual policy navigation commands and landing requests after the local operator starts it. It records video, telemetry, 89-input observations, raw policy outputs, issued commands, timestamps, and all three model identities.

The supplied selection is **full_horizon-round-05** for the explicitly requested experimental lab test. This is **not a promoted winner**: its saved randomized simulation screen has **48/100 clean repeats and 52/100 crashes**. Physical success is unvalidated. The prior baseline flights failed; replacing their model identity does not establish that this composite will fly well.

Old duplicate packages and unrelated historical trees were removed from the current branch. They remain recoverable at commit `405b9390b97e25fd29c428d9010e63ea3fa0d255`. [Recovery inventory](experiments/provenance/prior-tree.json). No old PC run or source snapshot was deleted from the PC.
