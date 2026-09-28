# Current results

See [Latest progress and results](LATEST_PROGRESS.md) for the28 September recurrent experiments. Full mission success remains0/25; no simulation or physical completion is claimed.

# Measured experiment results

Final saved validation for each bounded trial. Passing suites use25 episodes; return development uses50. Full-task success remains0/25 throughout.

| Run | Steps | Centered | Varied | New passes | Safe returns | Return crashes |
|---|---:|---:|---:|---:|---:|---:|
| anchor-repair-20260927 | 20000 | 24/25 | 21/25 | 0/50 | 0/50 | 27/50 |
| front-memory-20260927 | 20000 | 24/25 | 22/25 | 0/50 | 0/50 | 11/50 |
| metric-ring-20260927 | 15000 | 24/25 | 23/25 | 0/50 | 2/50 | 32/50 |
| recent-retrace-20260927 | 20000 | 24/25 | 22/25 | 0/50 | 0/50 | 0/50 |
| reset-lessons-20260927 | 20000 | 24/25 | 23/25 | 0/50 | 0/50 | 0/50 |
| return-practice-overnight-20260927 | 20000 | 24/25 | 23/25 | 0/50 | 0/50 | 25/50 |
| return-retrace-20260927 | 20000 | 24/25 | 22/25 | 0/50 | 0/50 | 24/50 |

The six20k trials stopped for insufficient sustained return progress. The latest metric trial stopped at15k for mastered-lesson regression. Zero crashes can indicate stalling and must not be confused with task success. Reset lessons, initial passages and diagnostic tests cannot substitute for full-mission acceptance. Detailed per-episode results and intermediate checkpoints are under evidence/.
