# Whole-sequence imitation audit — 28 September

No training was active; process inspection found only the saved-policy viewer.
This audit used the nine existing FIT successful teacher histories and previously
saved student rollouts. No reserved trajectories or independent acceptance seeds
were used. Predictions ran on the local CUDA GPU; no simulator flight was run.

## Findings

| Checkpoint | Mean action MSE on teacher histories | Right | Forward | Up | Yaw |
|---|---:|---:|---:|---:|---:|
| 300 updates | .11372 | .09834 | .13354 | .12257 | .10042 |
| 500 updates | .10194 | .08443 | .12589 | .10635 | .09111 |

These are normalized action errors, not success rates. All four outputs retain
substantial error. The worst average 100-step segment is steps 101–200: .20348
at 300 updates and .17981 at 500. This temporal window is not a phase label;
the nine flights need not reach the ring at the same time.

At 500 updates, the first student/teacher position difference exceeding 10 cm
occurs at steps 15, 70, 4, 47, 44, 79, 30, 57 and 16 (FIT seed order in JSON).
At 300 it occurs at 14, 60, 4, 12, 44, 61, 49, 78 and 69. At dt=.1 seconds,
these differences develop within .4–7.9 seconds. They compare positions at the
same elapsed step; a different valid path is possible in principle, but these
student rollouts all failed the full task. This does not isolate a single cause.

Whole-sequence distribution means and stepwise recurrent predictions differ by
at most .000378 after clipping to the action bounds. Initial strict 1e-5 audit
assertion stopped the first diagnostic. The rerun reported actual errors with a
1e-3 long-CUDA-sequence tolerance; no model, reward or success threshold changed.
This small discrepancy is far below the imitation error, so the evidence does
not indicate a gross sequence layout or hidden-state reset bug.

## Decision

Do not extend the 500-update trial unchanged. The policy cannot closely reproduce
even its selected teacher histories, and closed-loop errors appear early. More
work is needed than treating ring disappearance alone. Next compare a bounded
FIT-only capacity/optimization diagnostic against current mixed basic/lap fitting,
with authentic complete histories and fixed basic-skill retention checks. Such a
diagnostic must establish whether the actor can fit these nine sequences before
claiming a generalization solution. Any resulting overfit policy is diagnostic,
not a deployable winner. A subsequent training decision should use measured
closed-loop FIT reproduction and retention, not MSE alone.

Reproduce:
`python DroneSoccer-Optimized/diagnostics/audit_sequence_fit.py`

Machine-readable evidence: `sequence_fit_audit.json`, including dataset hashes,
per-sequence errors, variance and rollout outcomes. No acceptance achieved;
physical flight remains unvalidated.
