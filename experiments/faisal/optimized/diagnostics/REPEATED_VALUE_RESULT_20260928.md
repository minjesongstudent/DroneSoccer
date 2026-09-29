# Repeated-data critic result

The16-epoch CUDA run completed. An independent checkpoint comparison confirmed
all actor parameters exactly match the preserved source. All2304 sampled
trajectories were FIT records; each category's parent-family counts differ by
at most one within an epoch. No trainer remains active.

| Epoch | FIT RMSE | Heldout RMSE |
|---|---:|---:|
|4|20.586|14.002|
|8|18.554|14.390|
|16|18.839|11.298|

Final heldout category RMSE: centered36.07,early return5.82,delayed4.44,
nominal8.11,reapproach9.46,varied35.77. Early return explained variance is
-0.244 and reapproach-0.417. Pooled improvement does not support promoting
the critic as a reliable return predictor. Dataset-dependent constant errors
are in metrics.json; do not compare those constants as though data were fixed.
The24heldout records are unchanged but repeatedly inspected development data.

Final start-value RMSE against recorded eight-repeat means: centered13.85,
varied17.62,early return8.38,reapproach7.47. These repetitions are now part of
FIT, so this is training diagnosis rather than independent validation. Eight
samples have considerable uncertainty; audit.json includes standard errors.
No flight improvement was tested or claimed. The actor and viewer remain
unchanged. All artifacts and source snapshots retained, raw critic scale40.

Reproduction: diagnostics/calibrate_recurrent_value_repeated.py (requires new
OUT to repeat without overwriting), followed by diagnostics/audit_repeated_value.py.
Outputs under diagnostics/recurrent-value-repeated-20260928 include input hash
manifest, model0/4/8/16, metrics/result, optimizer, selections and audit.

## Decision and next step

Do not extend this critic-fitting family unchanged. Its weak return prediction
has persisted across changed weighting and greater repeated-data coverage.
A next bounded actor experiment can instead test a return baseline estimated
from separate sampled continuations of the same authentic history. This does
not require a well-fitted learned critic. For each update, collect a fresh
on-policy batch; exclude the evaluated trajectory from its baseline estimate,
and detach baseline values before differentiating. Use only FIT history IDs,
actual learned-policy actions and public observation prefixes. A baseline
must not depend on that trajectory's sampled action or future outcome.

Before launching, implement and test those independence constraints and
likelihood replay parity, then use the existing rollback guard, basic-pass
retention checks, and full/return/crash evaluations. Keep updates bounded and
preserve source/checkpoints. This is a proposed variance-reduction experiment,
not evidence that it will learn repeated passages. Fixed validation followed
by reserved fresh acceptance remains necessary; physical flight unvalidated.
