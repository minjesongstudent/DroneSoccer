# Why the isolated anchored success did not persist

On fixed lesson seed5000003, baseline500 passes at84 and collides at209.
Head-anchor100 checkpoint25 passes at82, returns fully to the front at309
(about85cm lateral offset), passes again at371, returns front again at522,
and survives to600. It ends misaligned with two passes; this satisfies the
existing two-pass horizon criterion on this one layout, not general acceptance.
Checkpoints50/100 follow almost the baseline path: pass84, collision209/210.

Evidence: anchor_success_audit.json from audit_anchor_success.py. All movement
is deterministic learned-policy inference using authentic natural resets and
recurrent histories. Simulation geometry was used only for reporting.

Hypothesis: the strong basic-output penalty pulls the shared output head back
toward baseline before lap fitting improves materially. This is an inference,
not an established cause. Compare anchor coefficient10 against100, keeping
source500, seed44/45, frozen layers, data, learning rate,100-update bound and
retention checks the same. The comparison must not select a policy from this
one successful seed or treat its trace as training data. Do not use7M acceptance
or reserved teacher examples. The trial's protocol/status/evaluations live in
lap-head-anchor10-20260928. No camera/physical-flight changes.

Comparison completed: coefficient10 retained25/25 centred throughout; varied21/25,21/25,22/25 at25/50/100. Lesson success0/25,1/25,0/25; general/delayed0/25 throughout. Frozen hashes passed. Final lap MSE .100513 and basic-output MSE 2.385e-5. No sustained return improvement; no further head-only extension or coefficient sweep justified by this result. No active trainer. Next inspect representation/objective limitations while retaining the existing baselines.
