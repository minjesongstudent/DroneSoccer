# Larger-batch result and repeated-layout comparison

The 48-distinct-episode mission trial stopped after iteration two on unchanged randomized retention guards: crashes 16/25 versus baseline 12/25 and clean repeat 9/25 versus 13/25. Final aligned/nominal/randomized clean repeat was 84/75/47 per 100. Randomized first pass 73, crashes 53 and returns 71. Candidate rejected; original retained.

Increasing unrelated episode evidence did not resolve the regression. The next test controls layout variation during outcome comparison: twelve FIT layouts per iteration, four independently sampled learned-policy attempts per layout. Every attempt resets the same simulator seed within its group; actor sampling continues independently. No state, seed or group outcome enters actor observations.

For each attempt, advantage is its exact binary mission outcome minus the mean outcome of the other attempts on that layout. Thus an easy layout where all succeed or a hard layout where all fail contributes zero outcome gradient. A lone incomplete group also contributes zero. No fitted state-value baseline is used, and no global advantage recentering changes the within-layout contrasts. This tests whether matched-layout outcome differences provide a better learning signal; it does not guarantee adequate exploration.

At most five iterations of 48 complete learned episodes, one actor epoch, original 85D source, learning rate 1e-5. Same source KL/rollback, one-hour cap, first-pass/crash/return/repeat guards and final evaluation. Stop rather than manufacture a gradient if an iteration has no mixed-outcome groups. Full group seed, trace, action, outcome, likelihood and advantage evidence is retained. No prior candidate continuation.

Ten focused tests passed, including correct leave-one-attempt-out arithmetic and homogeneous/partial groups. An eight-episode CUDA smoke correctly found zero mixed groups and performed zero actor updates; it is preserved as a no-signal test, not training success. A larger smoke checks whether naturally occurring mixed outcomes exercise updates. No artificial outcomes or diagnostic-controller actions are introduced.

```text
python -m tools.run_onpolicy_trial --out D:/DroneSoccerData/runs/grouped-mission-20261007 --actor-lr 0.00001 --objective mission --episodes-per-iteration 48 --actor-epochs 1 --group-size 4
```

No acceptance claim, physical operations, publishing, group-master changes or reserved acceptance-seed use.
The 16-episode CUDA smoke found one mixed group and completed 27 accepted updates, zero rejected updates, followed by saved-policy reload/evaluation. This validates execution only.
