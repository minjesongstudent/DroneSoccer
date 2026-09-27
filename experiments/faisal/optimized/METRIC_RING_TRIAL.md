# Camera-derived ring-centre trial

This trial adds an approximate ring-centre estimate to help the learned policy interpret its blind return after a passage. A frozen-policy audit found about 2-3 cm nominal position error, up to 5.18 cm in two contributing delayed flights; these are simulated, correlated samples, not a real-flight guarantee.

The 119D public input preserves all 115 existing camera/command/history values and appends body-frame forward/left/up displacement divided by 3 m plus a validity flag. Weak-perspective depth uses the last genuine front ellipse and fixed nominal diameter/camera calibration. Nominal issued-command odometry propagates it while blind. No simulator position, ring normal, wall coordinates or selected movement enters the actor. No scripted controller or demonstration is used.

The preserved 100D source actor and original frozen reference transfer with zero added columns and separate 100/15/4 linear calculations. Critics, replay and optimizers start fresh. An explicit archive/version check prevents accidental old-format loading. Authentic115D return banks remain byte-identical and are adapted only after validated restoration. All learned actions use the resulting119D policy.

Control: reset-lessons-20260927. Treatment retains original KL protection, reward, entropy, replay and mixture. Scheduler v2 additionally stops regression in a previously mastered reset lesson; this safety change is recorded separately from the input treatment. Failed anchor-repair behavior is not used.

Maximum40k transitions, evaluation every5k; 5k replay refill plus5k critic calibration precede actor updates. Retain centered>=.84 and varied>=.80; stop at20k without sustained >=5/50 return-development gain, and at40k without new-pass gain. Reset-lesson successes do not certify complete missions. Completion still requires three sustained fixed validation checkpoints and the independent100-layout acceptance test; reserved7M seeds remain unused until qualification.

Before launch: CUDA transfer/learning/save/resume tests and exact full-flight plus reset baseline comparison to the saved control. Reproduction artifacts: evidence/diagnostics/metric_baseline_v1.py and metric-baseline-v1. Training outputs belong on runs. Physical flight remains unvalidated.
