# Authentic return variance experiment

Predeclared88 continuations:11 authentic FIT histories,8 stochastic learned-
policy repetitions per history. Entry indices1,2,4,5,6,7,10,11,12,13,14 from
acquisition-return-mixture-20260928/protocol.json. Parent families0/3 and their
derived histories remain excluded. Each local snapshot hash is verified before
loading. The environment, current public observation and actual observation
prefix are restored anew for every repetition. Actor memory is reconstructed
from the authentic prefix with current weights; no simulator state enters the
actor observations. All movement actions come from the unchanged source actor.

Torch seeds993000+100*entry_index+repeat are declared before collection.
Budget600 steps per continuation or earlier mission end. Gamma.999 raw-return
targets, all outcomes retained, no training. Frozen broad and balanced critics
are compared at the identical start after their own prefix reconstruction.
Their outputs are multiplied by40 to restore raw units. Eight-return averages
are noisy development estimates, not acceptance results. New passes here are
continuations from stored histories, not natural-start success rates.

Runner diagnostics/audit_return_variance.py; output
 diagnostics/authentic-return-variance-20260928, with snapshot, protocol,
per-repetition NPZs, progress and final result. The runner refuses overwrite.
Use the workspace project venv. No actor update, physical operation or publish.

## Completed results

Session71193 exited0. All88 continuations preserved, source hash unchanged,
zero optimization steps. Early returns:1/48 continuations made a new passage,
45/48 crashed. Reapproaches:13/40 made a new passage,39/40 crashed. Combined:
14/88 made a new passage,84/88 crashed. A passage can precede a later crash;
these are not successful full missions or natural-start acceptance results.

Start-value RMSE against eight-repeat averages:

| FIT history category | Broad critic | Balanced critic |
|---|---:|---:|
| Early return,6 starts |5.17|9.99|
| Reapproach,5 starts |9.83|14.33|

Balanced predictions cluster near-42 to-45, while sampled return means range
from-22.26 to-41.05. Broad predictions also miss some reapproach means, e.g.
index12 predicts-39.52 against-22.26 sampled mean (sample SD18.60). These small
repeat samples are uncertain; repeated descendants of one parent flight are
not independent layouts. Do not call the broader critic accepted from these
FIT results. No promotion and no new flight-policy capability.

Next justified experiment: use the recorded repeated FIT trajectories to
improve training-target coverage, with weighting by parent history so that
repetitions do not silently dominate other tasks. Preserve existing heldout
histories and seeds; include nominal/delayed FIT data and basic retention.
Compare per-category errors and repeated-start mean predictions with explicit
uncertainty. This changes data support; do not merely extend stalled prior
runs. Any subsequent actor trial still requires bounded updates, policy-change
guards and fixed flight validation before independent acceptance.
