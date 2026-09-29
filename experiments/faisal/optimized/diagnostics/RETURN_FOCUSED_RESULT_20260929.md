# Return-focused actor trial result

The bounded trial stopped after20632 learned-policy environment steps. Two
updates were accepted; proposal3 exceeded max KL(.0140068>.01) and restored
weights/optimizer. All update reports confirm zero basic reinforcement terms,
69 guard episodes, and exact replay log-probability agreement. No promotion.

Fixed validation retained centered25/25 and varied21/25 throughout. Nominal
full success stayed0/25. Checkpoint1 achieved1/25 delayed full success on
seed5000005: two passages and timeout at600steps, crash rate96%. Checkpoint2
returned to0/25 delayed success and100%crashes. Lesson success stayed0/25;
return-to-front counts source/1/2 were3/5/2. An isolated intermediate success
is not sustained validation and is not simulation acceptance.

A separate paired development comparison is evaluating source and BOTH
accepted checkpoints,25episodes per suite, seeds6920000..6920024. These are
not reserved acceptance seeds. All three models are included to avoid reporting
only the favorable fixed-validation checkpoint. Source hashes/protocol/runner
and per-model reports are retained under development_comparison.

Runner compare_return_focused.py uses the saved source snapshot and real
learned-policy deterministic actions. It performs no training. Retain all
models and the existing viewer source. Await the full comparison before
proposing another training change. Physical flight remains unvalidated.

## Completed paired development comparison

Session65197 exited0; all375 episodes saved. Per25episode suite:

| Checkpoint | Centered | Varied | Lesson | Nominal full | Delayed full |
|---|---:|---:|---:|---:|---:|
| Source0 |25|22|0|0|1|
| Update1 |25|23|0|1|0|
| Update2 |25|22|0|0|0|

Full crash rates source0: nominal100%, delayed96%; update1:96%/100%;
update2:100%/100%. No sustained improvement or promotion.

Both success changes occur at development seed6920010. Nominal source passes
 twice then hits at step472; update1 passes three times and survives600steps.
Delayed source passes twice and survives600steps; update1 passes once and hits
at284steps. Thus there is a real isolated learned-policy three-pass outcome,
but it trades off against the delayed version and remains far below acceptance.
This checkpoint was investigated because of its isolated fixed-suite result;
these are development findings, not unbiased acceptance estimates.

Next diagnose saved/replayed public-history trajectories for this seed under
nominal/delayed dynamics across source and accepted checkpoints, recording
clearance and failure timing as simulator-only diagnostics. Do not feed this
development seed into training or extend this trial unchanged. Determine how
small policy changes alter return timing before proposing another experiment.
All movements during any replay must still come from the selected learned
policy. No active trainer, viewer remains original source.
