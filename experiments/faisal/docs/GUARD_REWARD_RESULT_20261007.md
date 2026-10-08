# Guard reward trial completed without randomized improvement

The bounded CUDA trial completed 5,000 critic-only and 5,000 accepted actor updates, with no rejected updates or retention-stop violations. The original checkpoint is unchanged. No additional training is authorized by this result.

Fixed development results per 100 episodes (original / candidate):

| Suite | First full passage | Return to front | Clean repeats | Crashes |
|---|---:|---:|---:|---:|
| Aligned | 84 / 85 | 84 / 85 | 84 / 85 | 16 / 15 |
| Nominal | 80 / 81 | 80 / 81 | 74 / 76 | 26 / 24 |
| Randomized | 73 / 71 | 70 / 69 | 48 / 48 | 52 / 52 |

These small mixed changes do not establish improved robustness. Critic residual reduction is not flight improvement. Keep the original selected and do not extend the run.

## One final fresh comparison

The final 5,000-step checkpoint is frozen before evaluation. Compare it with the original on 200 paired layouts per suite, using development seed bases 9,810,000 / 9,820,000 / 9,830,000. Reserved acceptance seeds remain untouched. All commands come from the respective learned policies. This comparison ends after 1,200 total episodes; no repeated evaluation until a favorable result appears.

```text
python -m tools.compare_boundary_candidate --out D:/DroneSoccerData/runs/guard-reward-fresh-20261007 --candidate D:/DroneSoccerData/runs/guard-reward-20261007/online_005000/model.zip --candidate-label guard_reward --episodes 200 --cohort 981
```

Decision recorded before fresh results: only retain this candidate for additional validation if randomized clean repeats improve by at least five episodes, randomized paired gains exceed losses with an exact one-sided binomial discordance probability at most 0.05, and no suite loses more than two first passages or clean repeats or adds more than two crashes. Otherwise reject the candidate and close this reward variant without further unchanged training. This is a development screening rule, not simulation acceptance or proof after multiple prior experiments.

If rejected, the guard mismatch remains an observed geometric issue but this penalty has not solved it. Do not infer that simply increasing its coefficient will work. Preserve all models, trajectories and the failed result. Physical autonomous flight remains unvalidated.

## Final decision: rejected

All 1,200 fresh evaluation episodes completed. Original / candidate counts per 200:

| Suite | First passage | Return | Clean repeats | Crashes |
|---|---:|---:|---:|---:|
| Aligned | 177 / 177 | 177 / 177 | 177 / 177 | 23 / 23 |
| Nominal | 150 / 150 | 150 / 150 | 138 / 137 | 62 / 61 |
| Randomized | 143 / 144 | 140 / 141 | 104 / 106 | 94 / 92 |

Randomized paired repeat gains/losses are 11/9: net +2, one-sided exact binomial p=0.411901474. Both required improvement conditions fail. Existing retention conditions pass, but retention alone is insufficient for promotion. Randomized ring contacts are unchanged at 42 each; the two fewer crashes are net contacts (52 to 50).

`tools/screen_guard_candidate.py` checked 200 unique seeds per model/suite and recomputed first-passage, repeat and crash totals from episode logs. Its hash-bound `decision.json` records rejection. No continued training, additional selection of checkpoints, coefficient increase or repeated fresh cohorts for this variant. The original model remains selected. This closes the experiment with a negative result; it does not complete the project.

A distinct next training design must address the failure before the near-ring penalty activates. The distance audit already found approximately 14–17 cm lateral error at 30 cm on failed approaches. A curriculum using complete learned approaches from fresh FIT layouts, with retained full-task episodes, is a candidate design; it is not implemented or validated. Any reset curriculum must preserve genuine camera/action history and may not use these development trajectories as training labels. Do not restart the exhausted reward or continuation-ranking experiments.
