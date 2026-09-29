# Repeated-trajectory critic experiment

A fresh-source bounded critic fit tests the new184 recorded FIT repetitions:
96 basic and88 authentic return continuations. Combined with71 previous FIT
trajectories this gives255 FIT records; the24 heldout records are unchanged.
These are correlated trajectories, not255 independent layouts.

Budget16 epochs x24 optimizer updates,384 total. Each update averages six
category losses. Within category, parent families are selected by shuffled
cycles, then a history uniformly within family, then one trajectory uniformly
within that history. Thus repetitions and multiple derived histories do not
increase a parent's category weight. An episode's loss averages its frames.
Every record is eligible; finite sampling need not select every record.
The final selections.json records all draws for inspection.

Original source model0, frozen actor with exact equality every epoch; only
independent critic parameters train. Adam.001, gradient cap1, gamma.999 recorded
returns divided by40, seed994000+epoch. Current-weight authentic prefix burn-in.
Checks and saved models at0/4/8/16, no best checkpoint selection on heldout.
Reported RMSE remains frame weighted and is not the family-weighted objective.
FIT-derived constant baselines also change with this dataset; compare absolute
errors on the unchanged heldout categories as well as baseline comparisons.

Runner diagnostics/calibrate_recurrent_value_repeated.py. Output
 diagnostics/recurrent-value-repeated-20260928 refuses overwrite and saves input
hashes, source snapshot, runner, models and optimizer. Use the project venv.
The changed data and parent weighting are an intervention, not isolated proof
of one cause. No actor or flight improvement is claimed by critic fitting.
These normalized-value models are not ordinary raw-value PPO resumes.
