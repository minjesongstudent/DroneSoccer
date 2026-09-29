# Zero-initialized observation bypass

Implemented src/public_bypass_policy.py as an explicit architecture migration.
The new119->128 projection adds current public features to the actor's recurrent
output before its existing policy MLP. It is initialized to zero. It reads no
simulator state, phase label, waypoint or replacement action. Critic recurrence
has no bypass; all actions still come from the learned policy distribution.

The common recurrent sequence path serves get_distribution, forward and
evaluate_actions, preserving episode masks and the recurrent state. Construction
rebuilds the optimizer to include new weights. Migration checks missing keys:
only the new bypass weight may be absent from the old checkpoint.

CUDA test diagnostics/test_public_bypass.py passed: exact old/new predictions
and recurrent state on32 genuine observations with a mid-sequence reset;
forward/distribution/evaluate_actions parity; nonzero new-weight gradients;
optimizer inclusion; and exact save/load prediction roundtrip after an update.
No physical hardware was used. Test updates only temporary in-memory copies.

Bounded trial lap-public-bypass-20260928 starts from baseline500 and trains only
the new bypass. All original parameters are frozen and hash-checked. Same9FIT
complete histories, basic-output anchor10 on8 sampled basic histories, lr1e-4,
max100/check25/50/100 and retention stop. This is supervised fitting, not RL or
acceptance. Source/model/data hashes, migrated checkpoints and results preserved.
No export/viewer replacement, reserved examples,7M tests or physical flight.

Completed100: centred25/25 throughout, varied21/22/22 at25/50/100. All fullmission suites0/25 at everycheck. Final preupdate lapMSE .095521; anchorMSE .000142. Originalparameterhashesunchanged. Noactive trainer, extension orpromotion. Bypass-only learning does not establish a solution; keep code experimental and original exports unchanged.
