# Constrained updates retained skill; continuation addresses stalled updates

The first constrained trial finished after 795 online transitions: 694 accepted
actor updates, then 100 consecutive rejected updates. The final step triggered
the stop before another optimizer update. All old models remain preserved.

| Suite | Original first / crashes / clean repeat | Candidate first / crashes / clean repeat |
|---|---|---|
| Aligned, 100 episodes | 84 / 16 / 84 | 86 / 14 / 86 |
| Nominal, 100 episodes | 80 / 26 / 74 | 79 / 26 / 74 |
| Randomized, 100 episodes | 73 / 52 / 48 | 71 / 51 / 49 |

This avoided the previous catastrophic regression, but the small mixed differences
do not establish a better policy. No candidate promotion or acceptance is claimed.
The original policy remains retained for deployment previews.

Critic adaptation used 163,255 FIT transitions and 36,745 episode-grouped heldout
transitions from unchanged-source warmup. Heldout moving-target TD RMSE fell from
4.927 to 3.873 while actor weights and entropy stayed exact. That is a Bellman
residual diagnostic, not proof of calibrated values. The final FIT guard mean KL
was 0.005213 and maximum KL 0.099879, near the unchanged maximum limit of 0.1.

The actor loss previously penalized average KL on sampled FIT minibatches; it did
not provide a gradient for the separate fixed guard's largest deviation. Repeated
proposals then exceeded that guard and were rolled back. This motivates a changed
loss rather than continuing the stalled trial or relaxing the hard guard.

## Next bounded comparison

`constrained-boundary-20261007` starts again from the same original model and
200,000-transition warmup dataset. It repeats the same critic adaptation and
online limits, adding only a FIT-guard boundary penalty with weight 0.1. The
penalty is the sum of squared normalized excesses above half the existing mean
and maximum KL limits. It gives a gradient away from a boundary before the hard
rollback check is reached. The hard KL, regression and acceptance limits remain
unchanged. No development or heldout observations enter the new penalty.

The boundary unit test checks zero penalty in the interior and a restoring
gradient for the largest near-boundary deviation. All 25 focused checks passed;
the preceding full suite passed 124 checks. A new CUDA smoke completed critic and
actor updates and model reload/evaluation with the new code. Smoke results are
not learning evidence.

Run data, sources and reproducibility protocols:

- `D:/DroneSoccerData/runs/constrained-clearance-20261007`
- `D:/DroneSoccerData/runs/constrained-boundary-20261007`

Reproduce the second run from its source snapshot with a new output directory:

```text
python -m tools.run_constrained_clearance --out <new-directory> --guard-boundary-weight 0.1
```

The project heartbeat remains active to inspect outcomes and continue justified
work. No physical flight or remote publication occurred. Simulation acceptance
still requires the project's sustained fixed checks and independent fresh-seed
test; the remaining crash rates are far too high to claim completion.
