# Development process through the metric-ring trial

## Starting point

The earlier work evaluated the included policy, established local CUDA training and a live simulator, compared A/B/C runs, selected useful single-pass checkpoints and began unified training. Those reports remain in the parent experiment directory. Subsequent work focused on a complete cycle: fully clear the ring, return around its exterior and pass again without crashing.

## Reward and task corrections

Full-body clearance was retained so passing the ring centre alone does not earn a passage. Reward diagnostics identified premature turn encouragement and a positive-reward loop without passages. Later route shaping used remaining route length across phase transitions and permitted either feasible return side. Tests separated diagnostic geometry probes from learned-policy outcomes. The actor continued to choose every training/evaluation action.

## Controlled experiments

1. Protected completion training preserved single-pass behaviour through a frozen-reference constraint. It still failed to return reliably. The route trial preserved a useful10k checkpoint, now packaged as route-source-10k.zip.
2. Authentic return practice captured complete learned flights at actual post-pass states, preserving camera history, dynamics and random state. Separate training/development banks prevented fitting the evaluation set. Historical passages did not count as new successes.
3. Critic audits found optimistic terminal values. A training-only asymmetric critic comparison worsened held-out error and was not selected. A short importance-corrected Retrace diagnostic improved value error, but its subsequent live trial did not improve return success.
4. Recent/all-lap replay improved access to current experience while keeping single-pass rehearsal. It produced safer stalling rather than successful returns.
5. Persistent front-sighting memory appended camera and issued-command history to the actor input, with exact source-policy transfer. It avoided overwriting front context when the ring back was seen, but return success remained zero.
6. Replay reconstruction showed no successful complete return transitions. Fresh reset lessons progressively shortened the return task, using real reset observations and learned actions. Advancement required repeated fixed validation and retention of earlier lessons.
7. A separate monotonic anchor-repair treatment allowed reduction of already excessive reference divergence. It accepted more actor updates but worsened flight. It was not adopted. A mastered-lesson regression stop was added after this failure.
8. The metric-ring treatment appended approximate ring-centre coordinates derived solely from camera fit, nominal calibration and command odometry. Separate100/15/4 input calculations preserved initial actions exactly. Full flight/reset baselines matched the control before learning;479 development tests passed.

## Latest outcome and diagnosis

Metric training stopped at15k through the regression guard. Centered/varied passing remained24/25 and23/25, but the easiest reset lesson fell from24/25 pass-and-survive successes to0/25. All25 flights still passed once, then left the cage4.4-6.7seconds later while blind. Authentic return validation showed0/50 new passages,2/50 safe front returns and32/50 crashes. The intermediate return selector ranked the two safe returns positively, but this did not override the stop or establish mission completion.

Frozen learned-flight diagnostics found metric error below3.7cm on the sampled failed nominal returns. In24 terminal stochastic flights, minimum twin-Q averaged-11.65 versus sampled soft return-31.97. This is terminal-selected, one continuation per layout; it suggests optimism but is not an unbiased expected-value estimate or proof of the only cause. Deterministic and stochastic outcomes, failures and limitations are kept separate.

## Next work

Repeat learned stochastic continuations from identical genuine post-pass states/actions before choosing another value-learning treatment. Any calibration dataset must remain separate from fixed validation. Do not add simulator truth to deployable actor inputs, substitute a scripted controller, weaken success criteria or merely extend a stalled trial.

Simulation completion still requires three sustained fixed validation checkpoints followed by an independent100-layout test. Reserved7M acceptance seeds have not been used. Camera calibration, sensing, real command response and physical clearance remain lab work. There is no claim of completed autonomous physical flight.
