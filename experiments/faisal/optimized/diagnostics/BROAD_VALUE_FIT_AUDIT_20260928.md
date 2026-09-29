# FIT residual and critic numerical audit

The broader critic remains unpromoted. Both checks used FIT trajectories only;
there was no training, new rollout, actor change or acceptance-seed use.

## Residual findings

Successful single-pass FIT episodes are the largest errors. For example,
centered episode71 has mean target +30.24 and mean prediction -24.47, RMSE54.90.
Varied episode75 has target +30.71 and prediction -23.66, RMSE54.59.
The nominal 600-step episode18 has target +6.91 and prediction -32.43,
RMSE39.48. This is underprediction on known training trajectories, not simply
failure to generalize to held-out trajectories.

Across nominal FIT frames target standard deviation is19.07 versus prediction
3.92. Centered target standard deviation is23.70 versus prediction3.60.
The current critic mostly predicts negative outcomes even for successful basic
passes. Basic task identity is explicitly supplied through MissionObservation;
it is not absent by design. The dataset has24 nominal and24 delayed FIT
trajectories, versus6 centered and6 varied,6 early returns and5 reapproaches.
One optimizer update per trajectory gives unequal update counts by category.
This is a plausible contributor, not proof of the root cause.

## Numerical check

The strict raw-value sequence parity assertion failed: maximum absolute
CUDA batch-versus-step difference0.00187159 exceeded0.001. The failing result
and exit are preserved; the threshold was not relaxed.
The worst FIT case was index12, a reapproach continuation. An independent
CPU/CUDA comparison found:

- CPU batch-versus-step maximum0.00000668.
- CUDA batch-versus-step maximum0.00187159.
- CPU/CUDA batch maximum0.00187302; step maximum0.00005245.
- RMSE on that case: CUDA batch2.798763 versus step2.798531;
  CPU batch2.798529 versus step2.798529.

The evidence points to a small backend numerical discrepancy, not a large
sequence-boundary error on this case. It is far too small to explain the
50-unit basic-pass errors. No universal parity guarantee is claimed.

## Reproduction and next experiment

Run diagnostics/audit_broad_value_fit.py and diagnostics/check_value_numerics.py
with the project venv. The former intentionally retains its strict failing
assertion; its complete diagnostic JSON is written before the assertion.
Outputs fit_residual_audit.json and numerics_audit.json are inside
recurrent-value-broad-20260928. Both scripts perform inference only.

Next justified bounded experiment: start from the same preserved source actor
and critic, balance gradient contributions across the six FIT categories,
retain authentic prefix reconstruction and actor equality checks, and compare
FIT residuals before checking heldout categories. Declare the budget and
weighting before the experiment. Do not simply extend the prior trajectory-
uniform run. Success here would mean better baseline predictions, not flight
success; any later actor trial still needs retention and full-mission tests.
