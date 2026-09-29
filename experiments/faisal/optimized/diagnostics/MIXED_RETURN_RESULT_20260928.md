# Mixed authentic-state trial result

Four updates completed with 9,107 training steps. Fixed centered success stayed
25/25 and varied success 21/25. The front-start lesson rose from 0/25 to 1/25
at each of the four checks. Both general full-task suites remained at zero.
Final fixed lesson crash rate stayed 88%; general and delayed crash rates
remained 100%. The general suite gained one return-to-front, without completion.

Training batches had 15, 14, 14 and 14 crashes out of 15 continuations, with
1, 4, 3 and 2 additional passages. Historical passes are not included. These
training continuations do not establish natural-start mission success.

## Paired development comparison

Evaluated source and final mixed model on the same separate development seeds
6910000–6910024, all five existing suites. Neither model completed any lesson,
general or delayed full mission. Both scored 25/25 centered and 21/25 varied.
The small fixed-lesson gain therefore did not transfer to this diagnostic set.
This was a post-selection development check, not the reserved 7M acceptance
test. Detailed passes, returns, crash rates and per-seed records are preserved
under `authentic-return-pg-mixed-20260928/development_comparison`.

No candidate promoted. Existing source policy and viewer remain unchanged.
The trial and comparison have finished; no training continues from this run.

## Decision

Do not simply extend this low-rate, four-update reward-gradient family. Across
the early-return, longer-horizon and mixed-state versions, no general mission
improvement was demonstrated. Before another experiment, inspect whether the
updates produce a coherent change in action distributions on the saved actual
histories, and whether gradient contributions from different failed/successful
continuations conflict. This can distinguish a weak/noisy update from missing
behavioral coverage without another blind training extension. Retain all
histories, source checkpoints, protocols and fixed validation criteria.
