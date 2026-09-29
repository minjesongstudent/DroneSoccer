# Balanced critic result and repeated-start diagnostic

The16-epoch experiment completed. Actor parameters stayed exactly unchanged;
there were zero actor updates. The trainer is no longer active.

| Metric | Epoch4 | Epoch8 | Epoch16 |
|---|---:|---:|---:|
| FIT RMSE |19.199|17.004|16.937|
| Heldout RMSE |13.245|13.644|14.760|

The global FIT-mean constant has heldout RMSE14.360. Final per-category heldout
critic/category-constant RMSE: centered43.04/39.27, early5.98/5.48,
delayed6.59/12.23, nominal9.95/12.68, reapproach11.09/7.78,
varied52.15/30.92. Basic FIT errors improved markedly (centered9.33,
varied10.41), while heldout performance did not support promotion. All
checkpoints retained. Do not select an earlier epoch using these heldout
results or treat critic changes as improved flight behavior.

## Repeated FIT starts

Ran96 new simulated flights: the12 existing FIT basic starts, each repeated8
times, using the unchanged source stochastic learned actor. Reset seeds stayed
fixed, Torch seeds991000+100*original_index+repeat were declared before
collection. All observations and actions are genuine, all outcomes retained.
No heldout start, reserved acceptance seed, actor optimization or hardware used.
All eight initial observations matched exactly for every reset seed.

Eleven starts produced both successful passes and crashes. Pass counts ranged
4/8 through7/8 for these11; the remaining varied start produced8/8 passes.
Initial-return sample standard deviations were22.75 to35.10 for the mixed
starts, versus0.18 for the8/8 start. These are small samples, not reliable
population success estimates. They measure stochastic policy sampling, not
deterministic-policy validation and not learning progress.

Initial critic RMSE against the12 eight-repeat return means was36.37 for the
broader critic and18.56 for the balanced critic. This FIT-only diagnostic shows
some improved mean prediction, despite poor single-rollout heldout metrics.
Eight samples still leave uncertainty; the return critic estimates expected
future return and should not be expected to reproduce each random outcome.
The previously observed successful-flight underprediction is real, but large
individual residuals alone cannot establish underfitting of expected values.

## Artifacts and next step

Balanced models/protocol/metrics/optimizer/source snapshot are under
recurrent-value-balanced-20260928. New rollouts, protocol, source snapshot,
result.json and value_comparison.json are under basic-return-variance-20260928.
Scripts: audit_basic_return_variance.py and compare_repeated_start_values.py.
Run with the workspace project venv. Collector refuses to overwrite output.

Before further actor work, use repeated learned-policy continuations from
identical authentic return histories to measure conditional return variability
and compare the frozen critics to those mean returns. Use FIT parent histories
only, fresh declared action seeds and current-weight prefix reconstruction;
retain all outcomes. This will determine whether return baseline errors reflect
systematic mean error as well as random outcomes. Do not extend either critic
run unchanged. No model promoted; simulation and physical flight acceptance
remain incomplete. Critic raw-value scale remains40.
