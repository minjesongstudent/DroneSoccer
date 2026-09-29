# Guarded larger-update result

The trial collected 8,646 simulated steps. Three updates were accepted; the
fourth was rejected and parameters plus optimizer state were restored. Fixed
centered success stayed 25/25 and varied success 21/25. Lesson, general and
delayed full-mission successes were zero at all accepted checks. No promotion.

| Proposal | Mean KL | Maximum frame KL | Accepted |
|---|---:|---:|---|
| 1 | 0.0001131 | 0.0012847 | Yes |
| 2 | 0.0003161 | 0.0032158 | Yes |
| 3 | 0.0005897 | 0.0066463 | Yes |
| 4 | 0.0009279 | 0.0107528 | No |

The fourth proposal exceeded the predefined maximum-frame limit of 0.01.
The bound was not relaxed. Source weights and the existing viewer are unchanged.

## Exploration audit

First-batch action standard deviation was approximately 0.20 on each axis.
Fewer than 1% of sampled commands per axis required clipping. Saturation does
not appear to dominate this batch.

Paired CUDA diagnostics used the same eight early-return snapshots and two
shared random seeds each, with Gaussian noise scales 0.5, 1.0 and 1.5. Policy
means and stored checkpoints were unchanged; no training occurred.

| Noise scale | Crashes /16 | Reached front | Additional passages |
|---|---:|---:|---:|
| 0.5 | 16 | 1 | 1 |
| 1.0 | 16 | 3 | 0 |
| 1.5 | 16 | 6 | 1 |

Higher noise supplied more front-reaching trajectories in this small sample,
but no scale yielded a surviving continuation. This is insufficient evidence
to select a new training noise scale or claim improved policy performance.
Full protocol and learned-action traces are under `exploration_audit`.

## Next diagnostic

Do not continue a learning-rate or noise sweep on these results. Audit the
public ring-position estimate against simulator geometry along recorded
histories, especially while the ring is invisible. Simulator truth is solely
the diagnostic reference, never an actor input. Determine whether failure is
associated with inaccurate estimates or persists despite accurate inputs before
changing observation construction or the learning architecture. Preserve the
full-mission acceptance criteria and all historical models.
