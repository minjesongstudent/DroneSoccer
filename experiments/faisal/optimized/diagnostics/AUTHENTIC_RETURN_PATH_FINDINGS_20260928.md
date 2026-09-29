# Return trajectory and paired-source findings

Replayed all 32 recorded learned-action continuations from their hash-verified
snapshot states. Every observation, reward, endpoint and termination step
matched exactly. Simulator positions and phases were used for diagnostics only.
Paths are saved under the trial's `trajectory_audit` folder.

- 17 collisions (`hit`), 14 cage exits, one timeout survivor.
- Nine continuations never saw the ring. Six reached `align_front`.
- Seventeen crashes had seen the ring within ten steps; seven crashes ended
  after more than 50 steps without seeing it. Visibility loss alone does not
  explain all failures.
- Snapshot 5 in batch 3 returned to the front at step 195, passed at 246,
  returned again at 380, then collided at 428. One additional pass, no completed
  continuation. The `hit` label does not distinguish noodle from hanger here.
- Snapshot 6 in batch 3 returned at 216, passed at 361 and survived to step 600
  with two total passages. It remained in return-side phase afterward, ending
  with 256 consecutive steps without ring visibility. It did not pass a third
  time. This satisfies the existing two-pass continuation endpoint, but does
  not demonstrate sustained laps or natural-start acceptance.

## Paired source comparison

Ran the preserved source policy on all eight batch-3 snapshots with exactly the
same Torch seeds (983200 through 983207). The source also produced the same two
additional-passage cases: snapshot 5 collided and snapshot 6 survived to timeout.
Both source and updated policy had seven crashes, one timeout, two additional
passages. These are post-hoc diagnostic comparisons, not independent validation.
Results and actual policy trajectories are saved in `paired_source_batch3`.

Thus the two training-batch passages are not evidence of improvement due to the
updates. Fixed validation likewise showed no sustained improvement. Do not
extend the four-update trial or promote its final checkpoint.

## Next justified work

Current snapshot coverage is predominantly early return-side states. Preserve
authentic first-reacquisition and subsequent re-approach states from these exact
trajectories, including failures and the full preceding public-input history,
then compare collision and clearance behavior there. Selection should include
every reached state of the declared phase, not only successful flights. This
can distinguish return navigation from re-approach control before choosing
another training change. No fabricated history, hidden actor observations or
changes to acceptance criteria are needed.

No training, hardware commands, publication or reserved acceptance tests were
performed during this audit. Existing source model and viewer remain unchanged.
