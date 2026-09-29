# Longer horizon result

The gamma 0.999 comparison completed four updates and 5,783 training steps.
Centered success remained 25/25 and varied success 21/25 throughout. The
front-start lesson scored 0, 1, 1, 1, 0 successes across checks 0–4. General
and delayed full-task success remained zero at every check. No model promoted.

At the final check, general and delayed suites both had 100% crashes, six
total passes each, and respectively zero and three return-to-front episodes.
These match the gamma 0.99 comparison. The lesson crash rate was 88% versus
92%, with the same 23 passes and four returns; this small difference does not
establish sustained improvement. Do not extend this run on that basis.

# Mixed authentic-state trial

A separate bounded trial starts from the preserved source model, not the
unsuccessful final checkpoint. It adds all seven collected live re-approach
entries to the same eight early-return snapshots. No outcome filtering is used.
Complete authentic prefixes are reconstructed using current model weights.

The trial keeps gamma 0.999, Adam 1e-6, the same reward-only Monte Carlo objective
and original remaining mission duration. It permits four batch updates with
15 continuations each, at most 36,000 simulated steps. Batch size and state
coverage change together, so this is a coverage experiment rather than a strict
equal-sample-budget comparison. Every update receives the unchanged five fixed
25-episode suites; stop below 24 centered or 20 varied successes.

Runner: `run_authentic_return_mixed.py`. Output:
`authentic-return-pg-mixed-20260928`. Source, protocols, checkpoints and complete
rollouts are saved. No new deployment observations, controller substitutions,
reward changes, hardware commands or independent acceptance seeds are used.
