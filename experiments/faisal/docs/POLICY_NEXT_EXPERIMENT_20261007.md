# Next policy experiment - 7 October 2026

Original status at preparation: no training had started. The user subsequently
authorized execution. See `POLICY_CLEARANCE_TRIAL_20261007.md` for the implemented,
tested and launched comparison. The proposal below is retained as the planning
record; the frozen run protocol records the final execution settings.

## Evidence

The 6 October comparison found 84/100, 80/100 and 73/100 first full passages for
`seenflag-s1-level1` in aligned, nominal and randomized suites. Eventual crashes
were 16/100, 26/100 and 52/100. A successful first passage is not a safe complete
episode.

A read-only audit compared native centre-plane counts with saved full-body counts:

| Suite | Centre crossings | Full passages | Episodes with a discrepancy |
|---|---:|---:|---:|
| Aligned | 420 | 420 | 0 |
| Nominal | 369 | 367 | 2 |
| Randomized | 288 | 276 | 12 |

Both nominal discrepancies ended at the 600-step time limit. All twelve randomized
discrepancies ended in a hoop/hanger collision after an extra centre crossing but
before its full clearance. This supports testing reward timing; it does not prove
that reward timing caused the collision or explain the other 40 randomized crashes.

Reproduce by summing `centre_passes` and `full_body_passes` in
`lab-data/policy-validation-20261006/seenflag-s1-level1/*/episodes.jsonl`, comparing
the fields per episode, and grouping discrepancies by `outcome`. No new flight
or model prediction is needed for this audit.

## Proposed bounded comparison

Use two sequential CUDA arms, with only one active trainer at a time. Each begins
from the same preserved 85-input checkpoint. The control retains the existing
reward. The treatment moves the existing 40-point passage bonus from centre
crossing to collision-free full-body clearance. It must not add a second bonus,
change collision geometry, reduce crash penalties, or change evaluation scoring.
The rest of the route shaping and simulator remain identical in both arms.

- Source SHA256: `8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef`.
- Preserve its exact 85-input camera history: lags 24,20,16,12,8,4,2,1,0,
  three actually issued actions and the seen-hoop flag. Do not widen it using the
  current trainer's append-only weight expansion: the 109-input history is a
  different layout.
- Preserve level-1 starts, hanger, existing randomization and the full 600-step
  task. All movement comes from the learned policy, including replay-buffer
  warmup. Simulator truth may score rewards but never enters actor observations.
- Tentative cap per arm: 200,000 policy-generated warmup transitions, then
  50,000 learning transitions, learning rate 0.00003, with checks every 10,000
  learning transitions. Preserve the other source SAC settings. Limit each arm
  to two hours as well as the transition cap. These are experiment choices,
  not established optimal settings or a completion estimate.
- Use new training seeds beginning at 11,000,000. Exclude all development and
  reserved acceptance seeds. Freeze the exact schedule, dependency versions,
  sources and hashes before execution.
- Save the source, checked candidates, optimizers, replay buffer, outcomes and
  trajectories. Predeclare quantitative first-pass retention stop thresholds
  before execution; stop on regression or numerical errors and preserve evidence.

Before execution, implement and test the profile wrapper and reward accounting:
no bonus at centre crossing; exactly one bonus at full clearance; no bonus for
collision or side crossing; unchanged public observations and action history;
correct terminal/reset handling. Verify CUDA is actually used. The current group
`src/train.py` selects CPU and is not a ready launcher for this 85-input experiment.

Compare both arms with the frozen source on existing development suites, reporting
first full passage, return-to-front, clean repeat passages and all crashes
separately. Promising results need a separately predeclared new development
comparison to check for selection effects. This experiment cannot establish final
acceptance: sustained fixed validation and independent fresh-seed acceptance remain
required under the project's existing criteria.

## User preparation

No additional recording or re-entry of measurements is needed to prepare this
experiment on the home PC. No drone connection is needed. Physical testing comes
later, after policy qualification and validation of the actual flight runner and
operator controls. The Mac preview and loopback command tests do not qualify a
policy for flight.
