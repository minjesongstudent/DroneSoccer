# Acquisition and return coverage trial

Natural-start support was added to the KL guard without changing the existing
snapshot path. Each genuine reset receives its own zero recurrent state; only
its first input has an episode-start flag. CUDA tests passed repeated-segment
distribution equality, oversized-update rollback, original snapshot rollback,
and natural collector likelihood/value agreement. No test weights were saved.

The bounded experiment starts from the preserved recurrent source model.
Each update collects 27 learned-policy flights or continuations:

- Eight existing early-return snapshots.
- All seven authentic re-approach snapshots, including failed flights.
- Eight genuine full-task resets, four nominal and four with two-step delay.
- Four genuine basic-pass resets, two centered and two varied.

Fresh reset seeds are `2961000 + 100*update + entry_index`; these are distinct
from fixed validation and the reserved independent acceptance set. No artificial
front marker, ring estimate or history is supplied. Recorded continuations use
their complete authentic prefixes; natural flights start with empty history.

Maximum four updates and 64,800 simulation steps. Adam rate 1e-5, gamma 0.999,
unchanged rewards, normalized Monte Carlo reward-to-go, and original episode
termination rules. Gradient contributions remain timestep-weighted, not equally
weighted by task category; this mixture is a coverage test. Cumulative KL is
measured against the original actor on the fixed first-batch histories, with
each actor's own reconstructed memory. Reject above mean 0.001 or maximum 0.01,
restore parameters and optimizer, and stop. Fixed validation follows each
accepted update; stop below 24/25 centered or 20/25 varied successes.

Runner: `run_acquisition_return_mixture.py`. Output:
`acquisition-return-mixture-20260928`. This is one training process, with no
physical commands or publication. Promotion requires the unchanged sustained
validation and independent acceptance protocol, not training reward increases.
