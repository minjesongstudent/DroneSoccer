# Balanced critic experiment

Motivation: prior FIT audit found successful basic trajectories underpredicted
by roughly50 reward units. The previous training stream gave24 natural nominal
and24 delayed episodes versus6 centered,6 varied,6 early return and5 reapproach
per epoch. Test equal category gradient weighting with the same data split.

Predeclared budget:16 epochs,24 optimizer updates per epoch,384 total updates.
Each update accumulates the mean episode loss from one FIT trajectory in each
of six categories, dividing each by6 before one shared clipped Adam step.
Each category uses a shuffled cyclic pool, with fresh deterministic ordering
seed990000+epoch. All71FIT trajectories are retained;24heldout excluded.
Start from the original source model0, not the previous calibrated critic.
Actor frozen with exact tensor equality assertions each epoch. CUDA required.
Targets gamma.999 returns/40; lr.001; gradient norm cap1. Save/report0/4/8/16.

This experiment changes category weighting, gradient aggregation, number of
optimizer updates and trajectory presentations. It is an intervention test,
not an isolated causal attribution to category weighting. No best checkpoint
selection from heldout metrics. Inspect FIT errors and all six heldout metrics.
No flight improvement follows from critic prediction improvement alone.

Runner: diagnostics/calibrate_recurrent_value_balanced.py.
Output: diagnostics/recurrent-value-balanced-20260928; refuses overwrite.
Run with .\DroneSoccer\.venv\Scripts\python.exe from workspace root.
Source snapshot, runner copy and hashed input protocol accompany checkpoints.
Do not resume these normalized-value checkpoints as ordinary raw-value PPO.
