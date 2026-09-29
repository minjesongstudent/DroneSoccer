# Frozen-feature capacity audit

The success collector's saved protocol and code both use deterministic learned
teacher actions. They are not sampled exploration actions. The rare successes
are selected outcomes (9 FIT out of11 successes in2048 flights), which remains
a major coverage limitation; this audit does not establish robust teacher skill.

Using all5400 FIT steps and authentic recurrent histories from baseline500,
an SVD least-squares diagnostic found the best unconstrained affine action head
over its128 hidden features plus bias. Effective feature rank129. Baseline MSE
.101967; best affine unregularized MSE .078208, clipped predictions .077726.
Per-axis errors .06164/.10132/.07610/.07378 (right/forward/up/yaw).

This establishes substantial residual error for this frozen feature space even
without a basic-retention penalty. It does not bound nonlinear heads, other
recurrent representations, or the optimum of a clipped-output objective. MSE
alone does not prove a flight will fail. No fitted head was installed, no model
weights changed, and no held-out or acceptance histories were used.

Evidence: audit_head_capacity.py and head_capacity_audit.json, with model hash.

Next bounded comparison: unfreeze the existing policy hidden layer and action
head while retaining frozen LSTM/critic, anchor coefficient10 and the same
source500, seeds, data, learning rate,100-update limit and retention checks.
This changes which layers may learn, not the observation or success contract.
Results live in lap-mlp-anchor10-20260928. No physical flight or publication.

Comparison result: policy-MLP trial first failed at77 on a Windows status-file replace lock; partial artifacts preserved. Identical separate retry with bounded write retries completed100. At25/50/100: centred25/25 throughout; varied21/25,22/25,21/25; frontmission1/25,0/25,0/25; general/delayed0throughout. Frozen-parameter checks passed. No sustained mission gain or promotion. No active trainer. Runtime lock recovery changes no model settings. Further local head/MLP sweeps are not justified by current evidence.
