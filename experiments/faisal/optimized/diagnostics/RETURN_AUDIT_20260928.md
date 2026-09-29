# Return audit:28 September

Replayed all25 fixed front-start flights from recurrent lap-transfer checkpoints300 and500 with learned actions and authentic LSTM episode resets. Detailed observations are in trajectories.json and summary.json beside the audit script.

The three500-update front-return events were geometric front clearance, not aligned/safe return completion:
- Seed5000015:front at step503; ring remained invisible afterwards; timed out600 with one pass and about60cm below ring.
- Seed5000021:front at264 roughly70cm below ring; exited cage at265.
- Seed5000023:front at256 roughly84cm lateral from ring; exited cage at262.

At300updates, two front events retained vision for most remaining steps but stayed30-33cm laterally displaced and collided with the ring region before a second pass.

These results qualify the earlier3/25 return count. No successful complete mission or reliable usable return has been demonstrated. Counting a geometric milestone is not a second passage. No success criteria or detector geometry were changed.

Next diagnostic:closed-loop reproduction on the nine FIT teacher layouts, keeping the two reserved success layouts and independent7M seeds untouched. Lower supervised action MSE alone cannot establish stable closed-loop behavior.

FIT closed-loop result:checkpoint300 achieved0/9 full missions (8 total first passages;5hits1exit3timeouts); checkpoint500 achieved0/9 (7 total first passages;3hits6exits). Thus the student cannot reproduce even its nine successful teacher layouts. Further work must address closed-loop imitation/stability before attributing failure solely to generalization. No new training was launched.
