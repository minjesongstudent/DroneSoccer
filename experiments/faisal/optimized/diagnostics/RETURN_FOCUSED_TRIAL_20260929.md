# Return-focused policy comparison

Motivation: the preceding trial's largest accepted policy drift occurred on a
basic-pass trajectory, despite unchanged basic success and no full-mission
improvement. Test removal of the basic-task reinforcement term while preserving
all safety/retention constraints. This does not freeze shared actor parameters
on basic observations: full/return gradients can still affect them, hence the
guard and flight checks remain essential.

Same original source checkpoint, reset/action seeds and23starts x3rollouts as
independent-baseline-actor-20260929. Same independent-peer baseline,gamma.999,
fixedscale40,trajectory denominator69,Adam1e-5,gradient cap.5,max4updates and
165600steps. Natural seeds2980000+100update+group;Torch995000+1000update+index.
Fresh on-policy batch each update. Early trajectories are paired until policies
diverge; later identical random seeds do not imply identical state histories.

Only change to reinforcement loss: multiply basic-task terms by0; full and
return terms remain1. All69 first-batch histories, including12basic flights,
remain in the fixed source-reference KLguard (mean.001,max.01). Rejection
restores weights/optimizer and stops. Fixed centered>=24/25 and varied>=20/25
retention checks unchanged. Lesson/full/delayed results and crashes recorded
after each accepted update. Source/viewer unchanged; no automatic promotion.

Three test_return_objective_mask.py tests passed: zero basic gradient,
unchanged full/return gradient, unknown task rejected. Runner syntax checked.
Existing independent-baseline and recurrent replay tests passed preceding run;
each live batch again enforces log-probability replay error<.001 and identical
starting observations/prefixes across peers. Update reports expose zero
basic_gradient_count and guard_episodes69.

Runner diagnostics/run_return_focused_actor.py; output
 diagnostics/return-focused-actor-20260929. Source snapshot,protocol,trajectories,
checkpoints,reports and guard measurements retained. The trial refuses overwrite.
No reserved acceptance tests, physical commands, publication or master changes.
