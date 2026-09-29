# Latency-bank continuation audit

Predeclared one deterministic and three stochastic continuations for each of
30 FIT snapshots,120 total. The original environment delay is preserved;
recurrent memory is rebuilt from each actual public prefix. Source actor
unchanged; all actions sampled or predicted by that learned actor. No training.
Seeds996000+100*entry_index+repeat. All12heldout snapshots excluded.

Deterministic continuation endpoints must exactly match the uninterrupted
source flights' outcome,pass count and step count. All observations, raw policy
actions, rewards and phase traces are preserved for every continuation.
Parent histories and delay pairs are correlated, not independent layouts.
Front-reached includes snapshots that already start in align_front; the single
FIT delayed align snapshot is therefore not a new return-to-front achievement.
Repeated passes from stored snapshots are not natural-start acceptance.

Runner diagnostics/audit_latency_continuations.py. Output under
 latency-return-histories-20260929/continuation_audit retains runner/protocol,
trajectory NPZs,progress and final result. No script/control substitution,
truth actor input, physical flight, publishing or acceptance test.

## Completed audit and bounded follow-up

Session46418 exited0;120 continuations saved, all30deterministic endpoints
exact. Nominal deterministic15:front1,newpass0,crash14,full0.
Nominal stochastic45:front12,newpass1,crash41,full1.
Delayed deterministic15:front2,newpass0,crash15,full0.
Delayed stochastic45:front7,newpass1,crash45,full0.
Front counts include the already-aligned delayed snapshot. The lone surviving
nominal repeat continuation is not natural-start acceptance or trained gain.

A fresh-source two-update policy trial now uses all30FIT snapshots (15nominal
and15delayed, including one delayed align),8new natural starts and4basic starts,
three independent learned rollouts each:126 per batch,max151200steps. Full/return
reinforcement only; all126first-batch histories remain in fixed KLguard including
12basic retention flights. Independent-peer baseline,lr1e-5,gamma.999,scale40,
clip.5,KLmean.001/max.01 and centered24/25,varied20/25 stops unchanged. New
resetseeds2991000+100update+group;Torch997000+1000update+entry. Noheldouthistories.
Phase snapshots from same parent remain correlated; no independent-layout claim.
Runner run_latency_bank_actor.py; output latency-bank-actor-20260929. Syntax
check passed; same tested baseline/mask/collector components. Each live update
asserts replay parity and same-start observations/prefixes. Evaluate every
accepted checkpoint; no automatic promotion,acceptance test,hardware or publish.
