# Actor and critic history reconstruction

Implemented `rebuild_actor_critic_memory` in `src/recurrent_burnin.py`.
It replays authentic public observation prefixes using the current policy's
forward method, discarding predicted actions. Recorded commands in those
observations are preserved. Both hidden and cell states are rebuilt, without
gradient history; the current observation is deliberately excluded.

CUDA validation on the first hashed authentic return snapshot passed:

- Actor state exactly matches the existing inference reconstruction.
- Resumed action, value, log probability and both recurrent states agree with
  processing the entire prefix plus current observation in one sequence.
- Repeating after independent in-memory actor and critic bias changes matches
  the sequence reference and differs from stale memory in both networks.
- States have no gradients; original training mode is restored; invalid
  prefixes are rejected. No model file was changed.

Run using the project Python environment:

    python DroneSoccer-Optimized/diagnostics/test_dual_recurrent_burnin.py

## Remaining integration

This helper alone does not make snapshot training correct. The installed
RecurrentPPO collector copies `_last_lstm_states` at rollout start, stores
pre-action states, and sets `_last_episode_starts` to vector-environment dones.
A normal reset would therefore erase a restored snapshot's prefix memory.
Integration must explicitly handle snapshot resets, keep true trajectory
boundaries for advantages separate from recurrent reset masks, and reconstruct
current-weight memory at subsequent update boundaries from the original prefix
plus actually collected observations. Recurrent minibatch updates must also
account for prefix states becoming stale after optimizer steps; merely fixing
collection-time memory does not solve that issue.

Before any bounded GPU training, verify reset isolation, timeout bootstrapping,
pre-action buffer state alignment, and current-weight prefix handling through
an actual optimizer update. Keep natural-start full-mission validation and
single-pass retention unchanged. Snapshot continuations are training diagnostics,
not independent full-mission acceptance.

No trainer was running during this check; only the existing viewer process pair
was present. This change establishes infrastructure correctness, not improved
flight performance. No reserved acceptance seeds, hardware or publishing used.
