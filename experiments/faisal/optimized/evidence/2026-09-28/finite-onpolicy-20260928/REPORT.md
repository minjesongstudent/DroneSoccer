# Finite on-policy trial: no update accepted

The bounded trial collected96 complete learned-policy flights (13,935 transitions) then rejected its first epoch. Mean collection-policy KL0.000819754 was within0.001, but maximum0.176734507 exceeded0.02. Clipped surrogate gain0.00904794 was positive. Rollback restored actor weights AND optimizer state. The saved candidate actor is exactly equal to the source actor; no improved or degraded candidate flight result exists. Training is stopped, not silently continued.

Exact replay of the update computation reproduced these values.126/13935states exceeded the per-state bound, across return, single-pass and full-flight groups. The largest violation occurred on a hidden-ring return observation with minimum Gaussian std0.010796 and maximum deterministic command shift0.004032. Other large violations also involved narrow distributions; several affect passing rehearsal. These small normalized mean-command differences do not establish harmless stochastic flight consequences. Do not raise the guard merely to accept this proposal.

All data and diagnostic results are preserved:rollouts_0.npz,episodes_0.json,updates_0.json,candidate_0.pt,baseline.json,rejection_audit.json. audit_rejection.py reruns the proposal only in memory, restores the source afterward, and executes no flights. No reserved-layout or7M evaluation. No critic changes or simulation acceptance.

Next implementation question is whether proposal construction can respect the existing KL limits before committing an entire epoch, while retaining meaningful finite-return updates. Any changed update rule needs explicit versioning, transactional tests and a fresh bounded protocol; do not resume the failed run, relax bounds or claim these diagnostics prove policy improvement.
