# Analytic mean-head trial: retention failure, 8 October 2026

The new analytic loss changed the intended policy weights but reduced actual
randomized flight reliability. The fixed retention check stopped the trial after
round two. This variant is closed. The original policy remains selected; no final
development screen, fresh paired screen or acceptance test was run.

Run: `D:/DroneSoccerData/runs/analytic-actor-20261008`.
[Frozen protocol](ANALYTIC_ACTOR_PROTOCOL_20261008.md).
Runtime63.188seconds;64fresh FIT flights/16,247frames;16accepted updates and zero
rejected updates. Two updates required one backtrack. Only `actor.mu.weight` and
`actor.mu.bias` changed. The backbone, variance, critics and source model remained
unchanged. The training budget was intentionally incomplete because of its stop
rule, not a software crash or stalled optimizer.

## Retention outcomes

Each cell is first passage / return-to-front / clean repeat / crashes, out of25
fixed cases. Clean repeat requires the full mission without a crash or wall touch.

| Suite | Original | Round1 | Round2, stopped |
|---|---|---|---|
| Aligned | 19 / 19 / 19 / 6 | 19 / 19 / 19 / 6 | 20 / 20 / 20 / 5 |
| Nominal | 18 / 18 / 17 / 8 | 18 / 18 / 16 / 9 | 18 / 18 / 16 / 9 |
| Randomized | 18 / 17 / 13 / 12 | 18 / 17 / 11 / 14 | 19 / 18 / 9 / 16 |

Randomized round2 had no repeat rescues and four losses relative to the original:

| Seed | Full passages before crash | Ending step | Outcome |
|---|---:|---:|---|
| 9430009 | 4 | 497 | ring/hanger hit |
| 9430010 | 3 | 431 | cage/net contact |
| 9430012 | 0 | 68 | ring/hanger hit |
| 9430024 | 2 | 327 | ring/hanger hit |

Thus three added crashes followed multiple passages; the remaining one preceded
the first passage. Nominal round2 rescued one episode and lost two, both after two
passages. Aligned gained one repeat. The randomized four-repeat loss and four-crash
increase exceed the predeclared three-case retention allowances. Selecting the
earlier checkpoint or proceeding to100case evaluation would violate the protocol.

## What this establishes

The numerical implementation worked: all training commands matched the loss's
per-frame actor calculations exactly. Maximum shadow replay position error was
6.8834e-15m; wrapped-heading error3.9968e-15rad. Every update had finite nonzero
mean-head gradients and changed weights. Final source KL mean was.004279,
maximum.05108, within the unchanged limits. There is no rejected-update stall to
justify loosening those limits or merely adding training time.

Accurate shadow motion and local loss derivatives did not establish a useful
closed-loop improvement. The loss holds camera/history fixed and does not
differentiate feedback, terminal decisions or passage progress. Those limitations
remain plausible explanations; this experiment does not isolate a causal one.
Loss values from different fresh episode batches are not an improvement curve.
Do not present lower clearance loss, successful individual passages, or the
higher first-pass count as successful qualification.

No further trial is automatically justified by this failure. A next proposal must
address a distinct measured mechanism, with a fixed test before further fitting.
Do not repeat this same clearance-only mean-head trial, change its rate/budget/KL
limits, or select its better-looking intermediate checkpoint.

## Reproduction and preserved evidence

Original invocation, from `DroneSoccer-Publish`:

```powershell
& '../DroneSoccer/.venv/Scripts/python.exe' -m tools.run_analytic_actor_trial --out D:/DroneSoccerData/runs/analytic-actor-20261008
```

The existing output is immutable experiment evidence; this command deliberately
refuses to overwrite it. Its protocol records all seeds and parameters, source
snapshot and input hashes. Initial/round model files, optimizer/RNG snapshots,
actual flight traces and separate loss-only geometry/dt inputs remain preserved.

One saved-data closure verified478 evidence entries,532unique files,43source
files, all64 public/issued-command histories, complete episode boundaries,
per-batch outcome recounts, all retention outcomes and exact changed parameters.
No inference, simulator stepping or optimizer updates were performed in closure.
Do not repeatedly rerun this completed review.

Closure script and report: `closure_review.py` / `closure_review.json` in the run.
Recorded decision SHA256:
`5af230968f4d906406156a746eac578dfbbc5befa697ed3bcdd43ba2ad6b10df`.

Simulation acceptance remains incomplete. Physical flight remains unvalidated.
No trainer remains active; only the earlier read-only simulator UI is running.
