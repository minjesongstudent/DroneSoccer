# Longer learned prefixes did not establish reliable mission preferences

The bounded run completed all 64 families with 256 learned prefixes and 1,512 actual learned continuations. Four prefixes in state 51 terminated before a continuation; their 24 repeated output rows are schema entries, not additional flights. No actor or critic was updated.

`tools/summarize_decision_blocks.py` verifies all 1,536 unique state/candidate/replica keys in both datasets, identical model hashes, finite outcomes and consistent terminal flags. It preserves whole-family FIT/held-out splits and reports all families and the 63 families where all prefixes survived. The latter is a descriptive subset selected using prefix outcomes.

## Results

For the survivor subset, timeout-bootstrapped return ranking agreement between replica halves was:

| Diagnostic | FIT | Held out |
|---|---:|---:|
| Single action | 53.9% | 47.2% |
| Twelve-step learned prefix | 56.5% | 65.3% |

Despite the held-out return agreement increase, the prefix median absolute return contrast was 2.261 versus a median paired standard error of 6.199. Only 19.4% of held-out contrasts exceeded two standard errors, a descriptive quantity without multiplicity correction. The two halves agreed on the best return candidate in 6 of 12 held-out families. Finite-return sign agreement was 62.5%.

Mission comparisons were less convincing: held-out sign agreement was 4 of 14 informative contrasts (28.6%); only one family had a unique best candidate in both halves, and its choices disagreed. Most mission contrasts tied. These results do not justify actor training on the estimated sequence rankings.

The two datasets use different future RNG streams. Prefix outcomes are conditional on a single realized learned trajectory. Six contrasts from each family are dependent. This is neither an isolated causal estimate of block length nor evidence of improved flight performance.

## Decision and next work

Stop this diagnostic at its declared bound. Do not add replicas, fit a critic to these rankings, select successful prefixes as a controller, or promote the adapter. The original 85-input policy remains selected.

Before another training design, return to existing deterministic development trajectories and classify failures by approach, passage clearance, departure and return. Compare commands and camera visibility around the first failure in paired original/adapter trajectories. This can identify a concrete behavioral defect instead of continuing to optimize noisy continuation labels. This next analysis is not yet implemented, and its outcome is not assumed.

Saved analysis: `D:/DroneSoccerData/runs/decision-block-analysis-20261007`, including report, arrays, hashes and source snapshots. Four focused tests for tie handling and soft-return accounting passed. The separate trace verifier checks all saved joins, action bounds and finite-return recomputations.

Simulation acceptance remains false; reserved acceptance seeds are unused. Physical flight remains unvalidated. No hardware, publication or group-master changes occurred.
