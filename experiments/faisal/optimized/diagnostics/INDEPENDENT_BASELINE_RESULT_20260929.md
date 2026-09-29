# Independent-baseline actor trial result

The trial stopped after20118 learned-policy environment steps. Two updates
were accepted; the third proposal exceeded the unchanged maximum KL limit
(.010740 versus.01) and restored parameters and optimizer. Last accepted
checkpoint is model_2.zip. Source and viewer remain unchanged; no promotion.
All three replay likelihood checks had zero discrepancy.

Fixed validation at source,update1,update2 retained centered25/25 and varied
21/25. Lesson,nominal full and delayed full success remained0/25 throughout.
Nominal/delayed full suites each had6 total passes,zero two-pass episodes and
100% crashes at every check. Lesson source/final passages22/21,return-to-front
3/2,crash88%/88%. Delayed return-to-front3/2;nominal0/0. No sustained progress.

Accepted guard mean/max KL: update1 .00007171/.00455767;update2
.00017651/.00709873. Rejected update3 .00033453/.01074001. Limits and
success criteria were not relaxed. All checkpoints, trajectories and reports
are retained under independent-baseline-actor-20260929.

## Diagnostics

Across collected batches early-return new passages were0/18 each; reapproach
new-pass counts5/15,6/15,3/15 with nearly all crashing. Natural full first-pass
counts0/24,3/24,5/24, all crashed. Different starts/actions and updated policies
make those counts unsuitable as paired improvement evidence.

The peer baseline reduced scalar weight RMS in early-return/reapproach/full
categories. For batch0, early .478 vs uncentered .801, reapproach .494 vs .793,
full .347 vs .904. Basic weights increased1.014 vs .787. Scalar weight size is
not gradient variance and cannot establish a better learning direction.

CUDA inference on saved first-batch histories located the accepted checkpoint's
largest drift in basic trajectory68 at step22: KL.00709873. The largest
reapproach peak was.00214886. This suggests the shared update spends substantial
policy change on a basic task whose deterministic score was already strong.
The rejected proposal was rolled back and not saved, so its peak location is
unknown. Do not attribute its rejection to that same trajectory as a fact.

Reproduction scripts: audit_independent_baseline_trial.py and
 audit_baseline_kl_locations.py. Outputs audit.json and kl_locations.json.
No simulator/hardware movement was used by these two audits.

## Next justified comparison

Do not extend this stalled trial unchanged. Test a fresh-source bounded
comparison where basic flights are retention/behavior checks but do not add
reinforcement gradients, concentrating the update on actual full-mission and
authentic return outcomes. Preserve the basic histories in the same fixed KL
guard and retain centered/varied stop thresholds, along with all full-mission
validation. First make that gradient-mask behavior explicit and test it.
This is a proposed diagnosis-driven change, not evidence of success; the
existing global guard must not be weakened to permit more drift.

Simulation acceptance and physical flight remain incomplete. No publications,
reserved acceptance evaluation or physical commands were performed.
